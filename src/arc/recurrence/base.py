from __future__ import annotations

import torch

from arc.models.base import ARCAdapter, RecurrenceResult
from arc.recurrence.state import RecurrenceState


class BaseLM(torch.nn.Module):
    """Native one-pass model: the equal-budget control substrate for CPT."""

    scale = "base"

    def __init__(self, adapter: ARCAdapter):
        super().__init__()
        self.adapter = adapter
        self.transformer = adapter.hf_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
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
