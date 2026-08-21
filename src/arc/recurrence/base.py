from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from arc.compute.flops import jetmoe_lm_head_flops_per_token
from arc.compute.latency import timer
from arc.models.base import ARCAdapter, ForwardContext, RecurrenceResult, TraceEvent
from arc.recurrence.scheduler import FixedSchedule
from arc.recurrence.state import RecurrenceState
from arc.routing.instrumentation import summarize_router_records


class RecurrentLM(torch.nn.Module, ABC):
    """Base for the three single-scale recurrence models.

    Hidden state always chains forward across repeated executions; the final
    norm + LM head run exactly once after the last execution. Hard limits are
    enforced here in the runtime, never by a controller.
    """

    scale: str = "model"

    def __init__(
        self,
        adapter: ARCAdapter,
        schedule: FixedSchedule,
        compute_budget_flops: float | None = None,
    ):
        super().__init__()
        self.adapter = adapter
        self.schedule = schedule
        self.compute_budget_flops = compute_budget_flops
        self.transformer = adapter.hf_model

    @abstractmethod
    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def num_units(self) -> int: ...

    def forward(self, input_ids: Tensor) -> RecurrenceResult:
        adapter = self.adapter
        adapter.begin_step()
        hidden = adapter.embed(input_ids)
        ctx = adapter.prepare(hidden)
        seq_len = hidden.shape[1]

        state = RecurrenceState(
            scale=self.scale,
            max_executions=self.schedule.total_executions,
            compute_budget=self.compute_budget_flops,
        )
        last_logits_history: list[Tensor] = []

        with torch.no_grad():
            last_logits_history.append(adapter.last_token_logits(hidden).squeeze(1))
            stop = False
            for unit_index in range(self.num_units()):
                if stop:
                    break
                for iteration in range(self.schedule.count_for(unit_index)):
                    est = adapter.unit_flops(self.scale, unit_index, seq_len)
                    if self.compute_budget_flops is not None and state.compute_used + est > self.compute_budget_flops:
                        state.truncated = True
                        stop = True
                        break
                    h_prev = hidden
                    lat: list[float] = []
                    with timer(lat):
                        hidden = self.execute_unit(unit_index, hidden, ctx)
                    sim = F.cosine_similarity(
                        hidden.float().reshape(-1, hidden.shape[-1]),
                        h_prev.float().reshape(-1, h_prev.shape[-1]),
                        dim=-1,
                    ).mean()
                    state.compute_used += est
                    state.record_execution(unit_index)
                    state.trace.append(
                        TraceEvent(
                            scale=self.scale,
                            unit_index=unit_index,
                            iteration=iteration + 1,
                            hidden_delta_cos=round(float(1.0 - sim), 6),
                            est_flops=est,
                            latency_s=lat[0],
                        )
                    )
                    last_logits_history.append(adapter.last_token_logits(hidden).squeeze(1))

            logits = adapter.final_logits(hidden)

        router_summary = summarize_router_records(adapter.get_router_records(), top_k=adapter.cfg.num_experts_per_tok)
        return RecurrenceResult(
            logits=logits,
            final_hidden=hidden,
            last_logits_history=last_logits_history,
            trace=state.trace,
            state=state,
            router_summary=router_summary,
            truncated=state.truncated,
        )

    def lm_head_flops_per_token(self) -> float:
        return jetmoe_lm_head_flops_per_token(self.adapter.cfg)


class ModelRecurrenceLM(RecurrentLM):
    """H_{t+1} = F(H_t): repeat complete model traversals."""

    scale = "model"

    def __init__(self, adapter: ARCAdapter, num_loops: int):
        super().__init__(adapter, FixedSchedule.uniform(1, num_loops))

    def num_units(self) -> int:
        return 1

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_model(hidden, ctx)


class BlockRecurrenceLM(RecurrentLM):
    """h_{b,r+1} = B_b(h_{b,r}): block = contiguous segment of `block_size` layers."""

    scale = "block"

    def __init__(self, adapter: ARCAdapter, schedule: FixedSchedule):
        expected = adapter.num_blocks()
        if len(schedule.counts) != expected:
            raise ValueError(f"block schedule needs {expected} counts, got {len(schedule.counts)}")
        super().__init__(adapter, schedule)

    def num_units(self) -> int:
        return self.adapter.num_blocks()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_block(unit_index, hidden, ctx)


class LayerRecurrenceLM(RecurrentLM):
    """h_{l,r+1} = F_l(h_{l,r}): repeat individual transformer layers."""

    scale = "layer"

    def __init__(self, adapter: ARCAdapter, schedule: FixedSchedule):
        expected = adapter.num_layers()
        if len(schedule.counts) != expected:
            raise ValueError(f"layer schedule needs {expected} counts, got {len(schedule.counts)}")
        super().__init__(adapter, schedule)

    def num_units(self) -> int:
        return self.adapter.num_layers()

    def execute_unit(self, unit_index: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        return self.adapter.forward_layer(unit_index, hidden, ctx)


def build_recurrent_model(scale: str, adapter: ARCAdapter, loops, block_size: int | None = None) -> RecurrentLM:
    """loops: int (uniform) or list[int] (heterogeneous, one count per unit)."""
    if isinstance(loops, int):
        if scale == "model":
            return ModelRecurrenceLM(adapter, loops)
        num_units = adapter.num_blocks() if scale == "block" else adapter.num_layers()
        return build_recurrent_model(scale, adapter, [loops] * num_units, block_size)
    schedule = FixedSchedule.heterogeneous(loops)
    if scale == "block":
        if block_size is not None:
            adapter.block_size = block_size
        return BlockRecurrenceLM(adapter, schedule)
    if scale == "layer":
        return LayerRecurrenceLM(adapter, schedule)
    raise ValueError(f"unknown scale: {scale}")
