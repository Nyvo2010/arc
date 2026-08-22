from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from arc.models.base import ARCAdapter, RecurrenceResult
from arc.recurrence.builder import build_model


class InferenceEngine:
    """Thin wrapper giving uniform inference for all 7 variants.

    All models expose:
        __call__(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult
    """

    def __init__(
        self,
        source: str,
        variant: str,
        block_size: int = 4,
        device_map: str | None = "auto",
        architecture: str = "jetmoe",
        recurrence: int = 1,
        max_loops: int = 4,
        controller_kwargs: dict | None = None,
        seed: int = 0,
    ):
        from arc.models.registry import MODEL_VARIANTS
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        cfg = MODEL_VARIANTS[variant]
        scale = cfg["scale"]
        adaptive = cfg["adaptive"]

        self.seed = seed
        self.adapter, self.model = self._build(
            source=source,
            scale=scale,
            block_size=block_size,
            device_map=device_map,
            architecture=architecture,
            recurrence=recurrence,
            adaptive=adaptive,
            max_loops=max_loops,
            controller_kwargs=controller_kwargs,
        )
        self.model.eval()
        if hasattr(self.adapter, "hf_model"):
            self.adapter.hf_model.eval()

    @staticmethod
    def _build(**kwargs):
        from arc.models.factory import build_arc_model
        model, adapter = build_arc_model(**kwargs)
        return adapter, model

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> RecurrenceResult:
        torch.manual_seed(self.seed)
        with torch.no_grad():
            return self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)

    def forward_logits(self, input_ids: Tensor, attention_mask: Tensor | None = None, position_ids: Tensor | None = None):
        res = self.__call__(input_ids, attention_mask, position_ids)
        return res.logits, res.state

    def measure(self, input_ids: Tensor, attention_mask: Tensor | None = None, position_ids: Tensor | None = None):
        import time
        t0 = time.perf_counter()
        res = self.__call__(input_ids, attention_mask, position_ids)
        t1 = time.perf_counter()
        state = res.state
        metrics = {
            "logits": res.logits,
            "final_hidden": res.final_hidden,
            "compute_used": float(state.compute_used),
            "executions": int(state.executions),
            "unit_loop_counts": {int(k): int(v) for k, v in state.unit_loop_counts.items()},
            "elapsed_s": t1 - t0,
            "tokens": int(input_ids.numel()),
        }
        return metrics
