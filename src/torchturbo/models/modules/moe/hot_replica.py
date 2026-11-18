"""Connect HMC placement to TorchTurbo's ordinary EP dispatcher."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ..comm.comm import gather_from_sequence_parallel_region
from .expert_layers import expert_layers
from .hot_replica_replay import ExpertReplay
from .token_dispatcher import MoEAlltoAllTokenDispatcher

if TYPE_CHECKING:
    from hyper_multicore.modules.hot_replica.runtime import HotReplica, HotReplicaRoute
    from .moe_layer import MoELayer


class NativeHotReplica:
    """Supply a placement and temporary routing width to the existing EP pipeline."""

    def __init__(self, replica: HotReplica, layer: MoELayer) -> None:
        self.replica = replica
        self.config = layer.config
        self.local_experts = layer.experts.num_local_experts
        self._dispatchers: dict[int, MoEAlltoAllTokenDispatcher] = {}
        self.retained_replays = 0

    def prepare(
        self, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
    ) -> HotReplicaRoute:
        def counts() -> torch.Tensor:
            local = routing_map.sum(dim=0).long()
            return gather_from_sequence_parallel_region(
                local, group=self.replica.ep_group,
            ).reshape(self.replica.ep_size, self.config.num_moe_experts)

        return self.replica.prepare(hidden, ids, probs, counts, self.config.moe_ffn_hidden_size)

    def replay(
        self, route: HotReplicaRoute, hidden: torch.Tensor, selected: torch.Tensor,
    ) -> ExpertReplay | None:
        """Get the agreed saved activations or reserve space during the no-grad forward."""
        record = route.record
        if record is None:
            return None
        if route.replay:
            if record.allow_activation:
                self.retained_replays += 1
                return record.activation
            return None

        from hyper_multicore.modules.hot_replica.checkpoint import ActivationReservation

        received = route.received if route.placement is None else route.placement.local_receive_capacity
        if not received or not selected.numel():
            return None
        # Grouped input, FC1 output, source-order output, and routing metadata.
        hidden_size = hidden.shape[-1]
        width = self.config.moe_ffn_hidden_size
        required = (received * (hidden_size + 2 * width) + hidden.numel()) * hidden.element_size()
        required += selected.numel() * (24 + selected.element_size()) + received * selected.element_size()
        reservation = ActivationReservation.reserve(hidden, required, self.replica.ep_group, native=True)
        if reservation is None:
            return None
        flat = hidden.reshape(-1, hidden_size)
        record.inputs = flat.detach(), selected.detach()
        record.input_versions = flat._version, selected._version
        return ExpertReplay(reservation)

    def dispatcher(self, extra: int) -> MoEAlltoAllTokenDispatcher:
        """Reuse an ordinary dispatcher for each logical placement width."""
        result = self._dispatchers.get(extra)
        if result is None:
            width = self.local_experts + extra
            begin = self.replica.rank * width
            config = replace(self.config, num_moe_experts=self.replica.ep_size * width)
            result = MoEAlltoAllTokenDispatcher(width, list(range(begin, begin + width)), config, self.replica.ep_group)
            self._dispatchers[extra] = result
        return result


def bind_hot_replica(model: torch.nn.Module, ep_group: dist.ProcessGroup) -> None:
    """Enable independent HotReplica after replacing the model's MoE layers."""
    from hyper_multicore.modules.hot_replica.runtime import HotReplica

    layers = expert_layers(model, ep_group, "HotReplica")
    geometries = tuple(
        (layer.experts.num_local_experts, layer.config.hidden_size, layer.config.moe_ffn_hidden_size)
        for layer in layers
    )
    replica = HotReplica(ep_group, geometries)
    for layer in layers:
        layer.native_hot_replica = NativeHotReplica(replica, layer)
