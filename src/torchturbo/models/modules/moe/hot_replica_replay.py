"""Replay cached routed-expert forward values with their native gradient paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchturbo.ops import ops
from torchturbo.ops.moe import MoEPermutation

from ..comm.comm import all_to_all

if TYPE_CHECKING:
    from hyper_multicore.modules.hot_replica.checkpoint import ActivationReservation
    from .token_dispatcher import MoEAlltoAllTokenDispatcher


@dataclass(frozen=True, slots=True)
class _DispatchLayout:
    """One invocation's permutation and exchange layout."""

    dispatcher: MoEAlltoAllTokenDispatcher
    send: tuple[int, ...]
    receive: tuple[int, ...]
    counts: torch.Tensor
    tokens_per_expert: torch.Tensor
    row_indices: torch.Tensor
    expert_indices: torch.Tensor
    source_rows: torch.Tensor

    def sort(self, tokens, probs=None, *, to_experts: bool):
        if self.dispatcher.num_local_experts <= 1:
            return tokens, probs
        sizes = self.counts if to_experts else self.counts.T
        indices = (self.dispatcher.sort_input_by_local_experts if to_experts
                   else self.dispatcher.restore_output_by_local_experts)
        return ops.moe_sort_chunks(tokens, sizes.ravel(), indices, probs=probs)

    def exchange(self, tensor: torch.Tensor, *, to_experts: bool) -> torch.Tensor:
        output, input_ = (self.receive, self.send) if to_experts else (self.send, self.receive)
        return all_to_all(self.dispatcher.ep_group, tensor, output, input_)


@dataclass(slots=True)
class ExpertReplay:
    """Owned activations of one routed forward and its admission charge."""

    reservation: ActivationReservation
    layout: _DispatchLayout | None = None
    tokens: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    fc1: torch.Tensor | None = None
    output: torch.Tensor | None = None

    def capture_dispatch(
        self, dispatcher: MoEAlltoAllTokenDispatcher, tokens: torch.Tensor, probs: torch.Tensor,
    ) -> None:
        """Capture only expert-major inputs and the layout needed by both adjoints."""
        indices = dispatcher.permutation_row_indices
        rows = torch.arange(indices.shape[0], device=indices.device).repeat_interleave(indices.shape[1])
        source_rows = torch.empty_like(rows).scatter_(0, indices.flatten().long(), rows)
        self.layout = _DispatchLayout(
            dispatcher, tuple(dispatcher.input_splits), tuple(dispatcher.output_splits),
            dispatcher.num_global_tokens_per_local_expert, dispatcher.tokens_per_expert,
            indices, dispatcher.permutation_expert_indices, source_rows,
        )
        self.tokens, self.probs = tokens.detach(), probs.detach()

    def dispatch(self, hidden: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached expert-major inputs and attach the dispatch adjoint."""
        return _ReplayDispatch.apply(hidden, selected, self)

    def combine(self, experts: torch.Tensor) -> torch.Tensor:
        """Return the source-order output and attach the combine adjoint."""
        return _ReplayCombine.apply(experts, self)


class _ReplayDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, selected, cache):
        ctx.shape = hidden.shape
        ctx.layout = cache.layout
        return cache.tokens.detach(), cache.probs.detach()

    @staticmethod
    def backward(ctx, grad_tokens, grad_probs):
        layout = ctx.layout
        grad_tokens, grad_probs = layout.sort(grad_tokens, grad_probs, to_experts=False)
        grad_tokens = layout.exchange(grad_tokens, to_experts=False)
        grad_probs = layout.exchange(grad_probs, to_experts=False)
        indices = layout.row_indices
        permutation = MoEPermutation(grad_tokens, None, indices, layout.expert_indices)
        grad_hidden = ops.moe_unpermutation(
            permutation, restore_shape=torch.Size((indices.shape[0], ctx.shape[-1])),
        ).reshape(ctx.shape)
        grad_selected = grad_probs.index_select(0, indices.flatten().long()).view_as(indices)
        return grad_hidden, grad_selected, None


class _ReplayCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, cache):
        ctx.layout = cache.layout
        return cache.output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        layout = ctx.layout
        packed = grad_output.reshape(-1, grad_output.shape[-1]).index_select(0, layout.source_rows)
        gathered = layout.exchange(packed, to_experts=True)
        grad_experts, _ = layout.sort(gathered, to_experts=True)
        return grad_experts, None
