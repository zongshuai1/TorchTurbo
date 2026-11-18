"""Expert layer checks shared by the HMC backends."""

import torch.distributed as dist
import torch.nn.functional as F

from .experts import GroupedMLP
from .moe_layer import MoELayer


def expert_layers(model, ep_group, backend):
    """Find routed expert layers supported by the selected HMC backend."""
    if ep_group is None or dist.get_world_size(ep_group) <= 1:
        raise ValueError(f"{backend} requires expert parallelism with ep_size > 1.")
    layers = [module for module in model.modules() if isinstance(module, MoELayer)]
    if not layers:
        raise ValueError(f"{backend} requires a model using TorchTurbo MoELayer.")
    ep_size = dist.get_world_size(ep_group)
    for layer in layers:
        cfg = layer.config
        if not isinstance(layer.experts, GroupedMLP) or layer.experts.use_te:
            raise ValueError(f"{backend} requires TorchTurbo GroupedMLP without Transformer Engine.")
        if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
            raise ValueError(f"{backend} requires SwiGLU experts with routing probabilities applied before FC2.")
        if layer.experts.num_local_experts * ep_size != cfg.num_moe_experts:
            raise ValueError(f"{backend} expert partition does not match the EP process group.")
    return layers
