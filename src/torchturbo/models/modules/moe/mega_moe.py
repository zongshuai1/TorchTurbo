"""Bridge between TorchTurbo's FSDP-owned experts and HyperMulticore on Ascend."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchturbo.ops.moe import MegaMoeContext

from .experts import GroupedMLP
from .moe_layer import MoELayer


def _expert_layers(model, ep_group):
    if ep_group is None or dist.get_world_size(ep_group) <= 1:
        raise ValueError("MegaMoE requires expert parallelism with ep_size > 1.")
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError("MegaMoE requires a model using TorchTurbo MoELayer.")
    ep_size = dist.get_world_size(ep_group)
    for layer in layers:
        cfg = layer.config
        if not isinstance(layer.experts, GroupedMLP) or layer.experts.use_te:
            raise ValueError("MegaMoE requires TorchTurbo GroupedMLP without Transformer Engine.")
        if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
            raise ValueError("MegaMoE requires SwiGLU experts with routing probabilities applied before FC2.")
        if layer.experts.num_local_experts * ep_size != cfg.num_moe_experts:
            raise ValueError("MegaMoE expert partition does not match the EP process group.")
    return layers


def bind_mega_moe(
    model: torch.nn.Module, ep_group: dist.ProcessGroup | None,
) -> None:
    """Bind serial MoE layers after replacement and before checkpointing/FSDP.

    Args:
        model: Model whose routed experts are TorchTurbo GroupedMLP modules.
        ep_group: Expert-parallel group with more than one rank.
    """
    layers = _expert_layers(model, ep_group)

    contexts: dict[tuple, MegaMoeContext] = {}
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
            )
        layer.mega_moe_backend = contexts[key]
        layer.router.return_selected_probs = True
