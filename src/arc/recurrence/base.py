from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from arc.models.base import ARCAdapter, ForwardContext, RecurrenceResult
from arc.recurrence.state import RecurrenceState


class RecurrentLM(torch.nn.Module, ABC):
    """Base for the fixed-recurrence models.

    Hidden state always chains forward across repeated executions; the final
    norm + LM head run exactly once after the last execution.
    """

    scale: str = "block"

    def __init__(self, adapter: ARCAdapter, recurrence: int):
        super().__init__()
        self.adapter = adapter
        if recurrence < 1:
            raise ValueError("recurrence must be at least 1")
        self.recurrence = recurrence
        self.transformer = adapter.hf_model

    @abstractmethod
    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def num_units(self) -> int: ...

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> RecurrenceResult:
        adapter = self.adapter
        hidden = adapter.embed(input_ids)
        ctx = adapter.prepare(hidden, attention_mask=attention_mask, position_ids=position_ids)
        seq_len = hidden.shape[1]
        batch_size = hidden.shape[0]

        state = RecurrenceState(scale=self.scale)
        with torch.no_grad():
            for unit_index in range(self.num_units()):
                for iteration in range(self.recurrence):
                    est = adapter.unit_flops(
                        self.scale, unit_index, seq_len, batch_size=batch_size
                    )
                    hidden = self.execute_unit(unit_index, hidden, ctx)
                    state.compute_used += est
                    state.record_execution(unit_index)

            final_hidden = adapter.normalize(hidden)
            logits = adapter.project_logits(final_hidden)
            state.compute_used += adapter.lm_head_flops_per_token() * seq_len * batch_size

        return RecurrenceResult(
            logits=logits,
            final_hidden=final_hidden,
            state=state,
        )


class BaseLM(torch.nn.Module):
    """Native one-pass model used as the Phase 1 control."""

    scale = "base"

    def __init__(self, adapter: ARCAdapter):
        super().__init__()
        self.adapter = adapter
        self.transformer = adapter.hf_model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> RecurrenceResult:
        with torch.no_grad():
            hidden, logits = self.adapter.forward_native(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        state = RecurrenceState(scale=self.scale)
        state.compute_used = (
            self.adapter.unit_flops("model", 0, seq_len, batch_size=batch_size)
            + self.adapter.lm_head_flops_per_token() * seq_len * batch_size
        )
        state.executions = 1
        return RecurrenceResult(
            logits=logits,
            final_hidden=hidden,
            state=state,
        )


class BlockRecurrenceLM(RecurrentLM):
    """h_{b,r+1} = B_b(h_{b,r}): block = contiguous segment of `block_size` layers."""

    scale = "block"

    def num_units(self) -> int:
        return self.adapter.num_blocks()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_block(unit_index, hidden, ctx)


class LayerRecurrenceLM(RecurrentLM):
    """h_{l,r+1} = F_l(h_{l,r}): repeat individual transformer layers."""

    scale = "layer"

    def num_units(self) -> int:
        return self.adapter.num_layers()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_layer(unit_index, hidden, ctx)


def build_recurrent_model(scale: str, adapter: ARCAdapter, recurrence: int) -> RecurrentLM:
    """Build one of the recurrent models with a uniform fixed recurrence value."""
    if not isinstance(recurrence, int):
        raise TypeError("recurrence must be an integer")
    if scale == "block":
        return BlockRecurrenceLM(adapter, recurrence)
    if scale == "layer":
        return LayerRecurrenceLM(adapter, recurrence)
    raise ValueError(f"unknown scale: {scale}")
