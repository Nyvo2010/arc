from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc.models.base import ARCAdapter, ForwardContext, RecurrenceResult
from arc.recurrence.state import RecurrenceState
from arc.recurrence.controller import RecurrenceController
import torch
from torch import Tensor


class AdaptiveRecurrentLM(torch.nn.Module):
    """Base adaptive recurrence with HALT/CONTINUE controller.

    Shares same adapter contract as fixed recurrence, producing identical
    RecurrenceResult shape for benchmarks.
    """

    scale: str = "block"
    granularity = "unit"  # unit = layer / block

    def __init__(self, adapter: ARCAdapter, controller: RecurrenceController):
        super().__init__()
        self.adapter = adapter
        self.controller = controller
        self.transformer = adapter.hf_model

    def num_units(self) -> int:
        raise NotImplementedError

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        raise NotImplementedError

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
                logits_prev = None
                hidden_prev = None
                rec_count = 0

                while True:
                    est = adapter.unit_flops(self.scale, unit_index, seq_len, batch_size=batch_size)
                    hidden = self.execute_unit(unit_index, hidden, ctx)

                    state.compute_used += est
                    state.record_execution(unit_index)
                    rec_count += 1

                    # Features for controller
                    hidden_for_logits = adapter.normalize(hidden)
                    logits_cur = adapter.project_logits(hidden_for_logits)

                    features = self.controller.build_features(
                        logits_prev=logits_prev,
                        logits_cur=logits_cur,
                        hidden_prev=hidden_prev,
                        hidden_cur=hidden,
                        recurrence_count=rec_count,
                        compute_used=state.compute_used,
                    )

                    logits_prev = logits_cur.detach()
                    hidden_prev = hidden.detach()

                    if not self.controller.decide(features, state):
                        break
                    if rec_count >= self.controller.max_loops:
                        break

            final_hidden = adapter.normalize(hidden)
            logits = adapter.project_logits(final_hidden)
            state.compute_used += adapter.lm_head_flops_per_token() * seq_len * batch_size

        return RecurrenceResult(logits=logits, final_hidden=final_hidden, state=state)


class BlockAdaptiveRecurrenceLM(AdaptiveRecurrentLM):
    scale = "block"

    def num_units(self) -> int:
        return self.adapter.num_blocks()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_block(unit_index, hidden, ctx)


class LayerAdaptiveRecurrenceLM(AdaptiveRecurrentLM):
    scale = "layer"

    def num_units(self) -> int:
        return self.adapter.num_layers()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_layer(unit_index, hidden, ctx)
