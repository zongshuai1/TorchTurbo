from __future__ import annotations

import importlib

import pytest
import torch

from torchturbo.models.modules.moe.utils import reorder_expert_chunks
from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.registry import registry
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime


@pytest.mark.parametrize("runtime", [CpuRuntime, CudaRuntime])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.NATIVE])
def test_native_dispatch(runtime, policy):
    dispatcher = OpDispatcher(
        device_runtime=runtime(), registry=registry, default_policy=policy
    )
    selected = dispatcher._candidate_impl(
        "moe_chunk_reorder", dispatcher.device_runtime.turbo_device
    )
    assert selected.op_spec().impl_name == "torch_moe_chunk_reorder"


@pytest.mark.parametrize("with_probs", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_native_forward_backward(monkeypatch, with_probs, empty):
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", dispatcher)
    counts = torch.tensor([[2, 0, 1], [0, 3, 2]])
    if empty:
        counts.zero_()
    num_tokens = int(counts.sum())
    tokens = torch.randn(num_tokens, 32, requires_grad=True)
    probs = torch.randn(num_tokens, requires_grad=True) if with_probs else None
    # Chunk order [0, 3, 1, 4, 2, 5] gives these independent row indices.
    indices = torch.tensor([] if empty else [0, 1, 3, 4, 5, 2, 6, 7], dtype=torch.long)
    output, scales = reorder_expert_chunks(tokens, counts, probs)
    torch.testing.assert_close(output, tokens[indices], rtol=0, atol=0)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected_grad = torch.empty_like(tokens)
    expected_grad[indices] = grad
    torch.testing.assert_close(tokens.grad, expected_grad, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(scales, probs[indices], rtol=0, atol=0)
        scales.backward(torch.arange(num_tokens, dtype=probs.dtype))
        expected_probs_grad = torch.empty_like(probs)
        expected_probs_grad[indices] = torch.arange(num_tokens, dtype=probs.dtype)
        torch.testing.assert_close(probs.grad, expected_probs_grad, rtol=0, atol=0)
    else:
        assert scales is None
