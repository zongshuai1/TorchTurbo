"""CPU checks for HMC provider selection and live expert parameter ownership."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, OpRegistry
from torchturbo.ops.moe import MegaMoeContext
from torchturbo.ops.providers.npu import mega_moe as hmc_provider
from torchturbo.runtime import TurboDevice


@pytest.fixture
def hmc(monkeypatch):
    constructed = []

    class Executor(torch.nn.Module):
        def __init__(self, **configuration):
            super().__init__()
            self.configuration = configuration
            self.weights = []
            constructed.append(self)

        def forward(self, inputs, ids, probs, *, expert_weights):
            assert ids.dtype == torch.int32
            self.weights.append(expert_weights)
            scale = sum(weight.float().sum() for weight in expert_weights)
            return inputs * (scale * probs.sum(dim=-1, keepdim=True)).to(inputs.dtype)

    monkeypatch.setitem(
        sys.modules, "hyper_multicore",
        SimpleNamespace(MegaMoeExperts=Executor, MegaMoeNumericsOptions=SimpleNamespace),
    )
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: object())
    monkeypatch.setattr(hmc_provider.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        OpDispatcher, "_load_provider_impl_classes",
        lambda self, device: {"hmc_mega_moe": hmc_provider.HmcMegaMoe},
    )
    return constructed


def dispatcher(policy=OpPolicy.AUTO, *, overrides=None, device=TurboDevice.NPU):
    registry = OpRegistry()
    registry.register(hmc_provider.HmcMegaMoe.op_spec())
    return OpDispatcher(
        device_runtime=SimpleNamespace(turbo_device=device),
        registry=registry, default_policy=policy, overrides=overrides,
    )


def inputs():
    context = MegaMoeContext(object(), 2, 1, 4, 2, 0.5, True)
    x = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3]])
    probs = torch.full((2, 2), 0.5, requires_grad=True)
    w1 = torch.ones(4, 2, requires_grad=True)
    w2 = torch.ones(2, 2, requires_grad=True)
    return (x, ids, probs, w1, w2), context


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
def test_dispatch_keeps_live_weights_and_reuses_context(hmc, policy):
    op = dispatcher(policy)
    args, context = inputs()
    assert not hmc
    result = op.mega_moe(*args, context=context)
    result.float().sum().backward()
    assert all(args[i].grad is not None for i in (0, 2, 3, 4))
    assert len(hmc) == 1
    assert hmc[0].configuration["ep_group"] is context.ep_group
    assert vars(hmc[0].configuration["numerics_options"]) == {
        "swiglu_limit": 0.5, "clamp_after_silu": True, "apply_probs_before_fc2": True,
    }
    with torch.no_grad():
        args[3].fill_(2)
    second = op.mega_moe(*args, context=context, training=False)
    assert len(hmc) == 1 and not hmc[0].training
    torch.testing.assert_close(second, torch.full_like(second, 20))
    assert hmc[0].weights[-1][0].shape == (2, 2, 2)
    op.mega_moe(*args, context=replace(context), training=True)
    assert len(hmc) == 2


def test_native_policy_rejects_hmc(hmc):
    args, context = inputs()
    with pytest.raises(RuntimeError, match="policy=native"):
        dispatcher(OpPolicy.NATIVE).mega_moe(*args, context=context)
    assert not hmc


@pytest.mark.parametrize("device", [TurboDevice.CUDA, TurboDevice.CPU])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
@pytest.mark.parametrize("selection", ["default", "explicit", "override"])
def test_unsupported_device_rejects_mega_moe(hmc, device, policy, selection):
    args, context = inputs()
    overrides = {"mega_moe": "hmc_mega_moe"} if selection == "override" else None
    impl = "hmc_mega_moe" if selection == "explicit" else None
    with pytest.raises(RuntimeError, match=f"device={device.value}"):
        dispatcher(policy, overrides=overrides, device=device).mega_moe(
            *args, context=context, impl=impl,
        )
    assert not hmc


def test_override_and_explicit_selection(hmc):
    args, context = inputs()
    dispatcher(OpPolicy.NATIVE, overrides={"mega_moe": "hmc_mega_moe"}).mega_moe(
        *args, context=context,
    )
    dispatcher(OpPolicy.NATIVE).mega_moe(*args, context=context, impl="hmc_mega_moe")
    assert len(hmc) == 2
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "missing"})


def test_unavailable_hmc_fails_without_initializing(hmc, monkeypatch):
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: None)
    args, context = inputs()
    with pytest.raises(RuntimeError, match="No implementation"):
        dispatcher().mega_moe(*args, context=context)
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "hmc_mega_moe"})
    assert not hmc


@pytest.mark.parametrize("invalid", ["activation", "weights"])
def test_invalid_inputs_fail_before_workspace_creation(hmc, invalid):
    args, context = inputs()
    args = list(args)
    if invalid == "activation":
        args[0] = args[0].float()
    else:
        args[3] = args[3].double()
    with pytest.raises(ValueError):
        dispatcher().mega_moe(*args, context=context)
    assert not hmc
