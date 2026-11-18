from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


_REPLAY_OUTPUTS = ContextVar("tt_npu_grouped_matmul_outputs", default=None)


@contextmanager
def replay_grouped_matmuls(*outputs: torch.Tensor):
    """Supply forward values to the existing NPU GroupedMatmul autograd nodes."""
    token = _REPLAY_OUTPUTS.set(iter(outputs))
    try:
        yield
    finally:
        _REPLAY_OUTPUTS.reset(token)


class _TorchNpuGroupedMatmul(torch.autograd.Function):
    """Differentiable grouped matmul backed by torch_npu."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        trans_b: bool,
        weight_grad_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_npu

        kernel_weight = weight.transpose(1, 2) if trans_b else weight
        counts = group_list.detach().to(
            device=x.device, dtype=torch.int64, non_blocking=True
        )
        ctx.save_for_backward(x, kernel_weight)
        ctx.counts = counts
        ctx.trans_b = trans_b
        ctx.fp32_weight_grad = weight_grad_target is not None
        replay = _REPLAY_OUTPUTS.get()
        if replay is not None:
            return next(replay).detach()
        if ctx.fp32_weight_grad and x.shape[0] == 0:
            return x.new_empty((0, kernel_weight.shape[-1]))
        output = torch_npu.npu_grouped_matmul(
            [x],
            [kernel_weight],
            group_list=counts,
            split_item=3,
            group_type=0,
            group_list_type=1,
        )
        return output[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch_npu

        x, kernel_weight = ctx.saved_tensors
        counts = ctx.counts
        grad_x = None
        grad_kernel_weight = None

        if ctx.fp32_weight_grad and x.shape[0] == 0:
            return torch.zeros_like(x), None, None, None, None

        if ctx.needs_input_grad[0]:
            grad_x = torch_npu.npu_grouped_matmul(
                [grad_output],
                [kernel_weight.transpose(1, 2)],
                group_list=counts,
                split_item=3,
                group_type=0,
                group_list_type=1,
            )[0]

        if ctx.fp32_weight_grad:
            if ctx.needs_input_grad[4]:
                grad_kernel_weight = torch.zeros_like(kernel_weight, dtype=torch.float32)
                torch_npu.npu_grouped_matmul_add_(
                    grad_kernel_weight, x, grad_output, counts.cumsum(0), group_list_type=0,
                )
            return grad_x, None, None, None, grad_kernel_weight

        if ctx.needs_input_grad[1]:
            grad_kernel_weight = torch_npu.npu_grouped_matmul(
                [x.transpose(0, 1)],
                [grad_output],
                group_list=counts,
                split_item=3,
                group_type=2,
                group_list_type=1,
            )[0]
            if ctx.trans_b:
                grad_kernel_weight = grad_kernel_weight.transpose(1, 2)

        return grad_x, grad_kernel_weight, None, None


def grouped_matmul_with_fp32_weight_grad(
    x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor, weight_grad_target: torch.Tensor,
) -> torch.Tensor:
    """Run grouped matmul and send FP32 weight gradients to the provided target."""
    return _TorchNpuGroupedMatmul.apply(x, weight, group_list, False, weight_grad_target)


@register_op(
    op_name="grouped_matmul",
    impl_name="torch_npu_grouped_matmul",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TorchNpuGroupedMatmul(OpImpl):
    def available(self) -> bool:
        try:
            import torch_npu

            torch_npu.npu_grouped_matmul  # noqa: B018
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        *,
        trans_b: bool = False,
    ):
        return _TorchNpuGroupedMatmul.apply(x, weight, group_list, trans_b)
