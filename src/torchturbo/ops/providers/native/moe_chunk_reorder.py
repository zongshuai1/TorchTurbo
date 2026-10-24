# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# Modified by the TorchTurbo contributors.

from __future__ import annotations

from typing import Optional

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        probs = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([probs[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


@register_op(
    op_name="moe_chunk_reorder",
    impl_name="torch_moe_chunk_reorder",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchMoeChunkReorder(OpImpl):
    def __call__(
        self,
        tokens: torch.Tensor,
        chunk_counts: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transpose rank/expert chunks with PyTorch split and concatenate."""
        order = torch.arange(chunk_counts.numel(), device="cpu")
        order = order.reshape(chunk_counts.shape).T.reshape(-1)
        return sort_chunks_by_idxs(
            tokens, chunk_counts.reshape(-1), order, probs=probs
        )
