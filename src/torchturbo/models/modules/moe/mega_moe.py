"""Bridge between TorchTurbo's FSDP-owned experts and HyperMulticore on Ascend."""

from __future__ import annotations

from weakref import ref

import torch
import torch.distributed as dist

from torchturbo.ops.moe import MegaMoeContext

from .expert_layers import expert_layers


def bind_mega_moe(
    model: torch.nn.Module, ep_group: dist.ProcessGroup | None,
    *, enable_hot_replica: bool = False, source_token_bound: int | None = None,
) -> None:
    """Bind serial MoE layers after replacement and before checkpointing/FSDP.

    Args:
        model: Model whose routed experts are TorchTurbo GroupedMLP modules.
        ep_group: Expert-parallel group with more than one rank.
        enable_hot_replica: Enable HMC expert placement within the MegaMoE executor.
        source_token_bound: Expected input rows per EP rank for workspace reservation.
    """
    layers = expert_layers(model, ep_group, "MegaMoE")

    contexts: dict[tuple, MegaMoeContext] = {}
    reservations = []
    for layer in layers:
        cfg = layer.config
        limit = layer.experts.swiglu_limit or None
        clamp_after = bool(limit and cfg.moe_swiglu_clamp_gate_after_activation)
        key = (
            cfg.hidden_size, cfg.moe_ffn_hidden_size, cfg.num_moe_experts,
            cfg.moe_router_topk, limit, clamp_after,
        )
        if key not in contexts:
            contexts[key] = MegaMoeContext(
                ep_group=ep_group,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.moe_ffn_hidden_size,
                num_experts=cfg.num_moe_experts,
                top_k=cfg.moe_router_topk,
                swiglu_limit=limit,
                clamp_after_silu=clamp_after,
                enable_hot_replica=enable_hot_replica,
                source_token_bound=source_token_bound,
                reservation_contexts=reservations,
            )
            if source_token_bound is not None:
                reservations.append(ref(contexts[key]))
        layer.mega_moe_backend = contexts[key]
        layer.router.return_selected_probs = True
