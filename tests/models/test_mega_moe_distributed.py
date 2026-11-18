"""Launch with torchrun -m pytest on two A3 ranks (EP) or four (EP plus FSDP)."""

import os
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.mega_moe import bind_mega_moe
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import get_ops, get_or_create_ops, OpPolicy
from torchturbo.runtime import get_device_runtime


@pytest.fixture(scope="module")
def expert_mesh():
    """Keep one communicator generation for all parameterized device cases."""
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        get_or_create_ops(device_runtime=get_device_runtime(), default_policy=OpPolicy.AUTO)
        mesh = init_device_mesh("npu", (dist.get_world_size() // 2, 2), mesh_dim_names=("edp", "ep"))
        yield mesh
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", 1)) not in (2, 4), reason="requires two or four torchrun ranks")
@pytest.mark.parametrize("enable_hot_replica", [False, True])
@pytest.mark.parametrize("hot_route", [False, True])
def test_mega_moe_matches_native_expert_parallel(expert_mesh, enable_hot_replica, hot_route):
    from torch.utils.checkpoint import checkpoint

    mesh = expert_mesh
    ep_group = mesh["ep"].get_group()
    cfg = MoEConfig(
        hidden_size=512, moe_ffn_hidden_size=128, num_moe_experts=8,
        expert_model_parallel_size=dist.get_world_size(ep_group),
        moe_router_topk=2, moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        activation_func=F.silu, gated_linear_unit=True, params_dtype=torch.bfloat16,
        moe_swiglu_limit=[0.5], moe_swiglu_clamp_gate_after_activation=True,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    native = MoELayer(cfg, ep_group, submodules, layer_number=0).npu()
    candidate = MoELayer(cfg, ep_group, submodules, layer_number=0).npu()
    torch.manual_seed(124 + dist.get_rank(ep_group))
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.035)
        if hot_route:
            # Positive inputs select experts 0 and 1 on the same owner.
            native.router.weight.fill_(-0.01)
            native.router.weight[:2].fill_(0.01)
    candidate.load_state_dict(native.state_dict())
    bind_mega_moe(candidate, ep_group, enable_hot_replica=enable_hot_replica)
    if dist.get_world_size() == 4:
        fully_shard(candidate.experts, mesh=mesh["edp"], mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
        ))
    assert list(native.state_dict()) == list(candidate.state_dict())

    for rows, recompute in ((128, False), (256, True)):
        x = torch.randn(rows, 1, 512, device="npu", dtype=torch.bfloat16, requires_grad=True)
        if hot_route:
            x = x.detach().abs().requires_grad_()
        other = x.detach().clone().requires_grad_()
        native.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        expected, _ = native(x)
        # Small correctness cases force placement; performance tests use
        # the unmodified cost model.
        policy = (
            patch("hyper_multicore.modules.hot_replica.planner.placement_is_profitable", return_value=True)
            if hot_route and enable_hot_replica else nullcontext()
        )
        capacity = (
            patch("hyper_multicore.modules.hot_replica.planner.placement_receive_limit",
                  side_effect=lambda average, *_: average)
            if hot_route and enable_hot_replica else nullcontext()
        )
        with policy, capacity:
            actual, _ = checkpoint(candidate, other, use_reentrant=False) if recompute else candidate(other)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        with policy, capacity:
            actual.backward(upstream)
        if hot_route and enable_hot_replica:
            provider = get_ops()._impls["hmc_mega_moe"]
            executor = provider._executors[candidate.mega_moe_backend]
            placement = executor._resource_group.resources.placement
            assert placement.statistics["placements"] > 0
            assert placement.pool.statistics["gradient_returns"] > 0
        pairs = [(actual, expected), (other.grad, x.grad)]
        for name, param in candidate.named_parameters():
            gradient = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            pairs.append((gradient, native.get_parameter(name).grad))
        for result, reference in pairs:
            assert result is not None and reference is not None
            torch.testing.assert_close(result, reference, atol=0.012, rtol=0.04)
