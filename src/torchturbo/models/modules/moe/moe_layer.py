  # Adapted from NVIDIA Megatron-LM:
  # https://github.com/NVIDIA/Megatron-LM
  #
  # Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
  # Licensed under the BSD 3-Clause License.
  #
  # Modified by the TorchTurbo contributors.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .moe_config import MoEConfig
from .router import TopKRouter
from .token_dispatcher import MoETokenDispatcher, MoEAlltoAllTokenDispatcher
from .module import ModuleSpec, build_module


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


@dataclass
class MoERuntimeInputs:
    padding_mask: torch.Tensor | None = None
    # False while recomputation replays a forward that already counted its tokens.
    accumulate_tokens_per_expert: bool = True


class BaseMoELayer(torch.nn.Module, ABC):
    """Base class for a mixture of experts layer."""

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        layer_number: Optional[int] = None,
    ) -> None:
        super().__init__()

        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)

        self.config = config
        self.expert_parallel_size = ep_size
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = (
            self.config.num_moe_experts // self.expert_parallel_size
        )
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = (
            self.config.moe_shared_expert_intermediate_size is not None
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)
        )
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.shared_expert = None
        self.shared_expert_gate = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number
        self._runtime_inputs = MoERuntimeInputs()

    def set_runtime_inputs(self, inputs: MoERuntimeInputs) -> None:
        self._runtime_inputs = inputs

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method for the MoE layer."""
        pass


class MoELayer(BaseMoELayer):
    """
    Mixture of experts Layer **currently only supports no token dropping**.
    """

    def __init__(
        self,
        config: MoEConfig,
        ep_group: dist.ProcessGroup,
        submodules: Optional[MoESubmodules] = None,
        use_te: bool = False,
        layer_number: Optional[int] = None,
        sp_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.mega_moe_backend = None
        self.native_hot_replica = None
        self.use_te = use_te
        self.submodules = submodules
        super(MoELayer, self).__init__(
            config=config, ep_group=ep_group, layer_number=layer_number
        )

        # Initialize router
        self.router = TopKRouter(
            config=self.config,
            layer_number=self.layer_number,
            sp_group=sp_group,
        )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                ep_group=ep_group,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.use_te,
            self.config,
            layer_number=self.layer_number,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config
            )

    def _dispatch_experts(
        self, hidden_states, routing_map, probs, expert_indices, route=None, selected=None,
    ):
        dispatcher = self.token_dispatcher
        route_options = {}
        expert_options = {}
        cache = None
        if route is not None:
            cache = self.native_hot_replica.replay(route, hidden_states, selected)
            placement = route.placement
            route_options["counts_by_source"] = route.counts
            route_options["selected_probs"] = selected
            if placement is not None:
                dispatcher = self.native_hot_replica.dispatcher(placement.borrowed_capacity)
                route_options["counts_by_source"] = placement.counts_by_source
                expert_indices = placement.virtual_ids
                expert_options["hot_replica_route"] = self.native_hot_replica.replica, route
            if cache is not None:
                expert_options["hot_replica_cache"] = cache
                if route.replay:
                    tokens, expert_probs = cache.dispatch(hidden_states, selected)
                    # Enter the FSDP wrapper even when all expert forwards are cached.
                    expert_output, _ = self.experts(
                        tokens, cache.layout.tokens_per_expert, expert_probs, **expert_options,
                    )
                    return cache.combine(expert_output)

        tokens, expert_probs = dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs, expert_indices=expert_indices, **route_options,
        )
        tokens, expert_probs = dispatcher.token_dispatch(tokens, expert_probs)
        tokens, tokens_per_expert, expert_probs = dispatcher.dispatch_postprocess(tokens, expert_probs)
        if cache is not None:
            cache.capture_dispatch(dispatcher, tokens, expert_probs)
        tokens, bias = self.experts(tokens, tokens_per_expert, expert_probs, **expert_options)
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"
        tokens = dispatcher.combine_preprocess(tokens)
        tokens = dispatcher.token_combine(tokens)
        output = dispatcher.combine_postprocess(tokens).to(hidden_states.dtype)
        if cache is not None:
            cache.output = output.detach()
            route.record.activation = cache
        return output

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens, compute local experts, and combine their outputs."""
        probs, routing_map, expert_indices, logits = self.router(
            hidden_states, input_ids, self._runtime_inputs.padding_mask,
            self._runtime_inputs.accumulate_tokens_per_expert,
        )
        logits = logits.view(-1, logits.shape[-1])

        if self.mega_moe_backend is not None:
            output, _ = self.experts(
                hidden_states, None, probs, mega_moe_backend=self.mega_moe_backend, expert_indices=expert_indices,
            )
        elif self.native_hot_replica is not None:
            selected = probs.gather(1, expert_indices.long())
            route = self.native_hot_replica.prepare(hidden_states, expert_indices, selected, routing_map)
            output = self._dispatch_experts(hidden_states, routing_map, probs, expert_indices, route, selected)
        else:
            output = self._dispatch_experts(hidden_states, routing_map, probs, expert_indices)

        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            output = output + shared_expert_output
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output, logits
