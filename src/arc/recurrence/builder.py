from __future__ import annotations

from arc.models.base import ARCAdapter
from arc.recurrence.controller_factory import make_controller


def build_model(
    scale: str,
    adapter: ARCAdapter,
    *,
    recurrence: int = 1,
    adaptive: bool = False,
    max_loops: int = 4,
    compute_budget: float | None = None,
    controller_kwargs: dict | None = None,
):
    """Unified builder for all 7 variants.

    scale: 'base' | 'model' | 'block' | 'layer'
    adaptive: False -> fixed recurrence with integer R
              True  -> adaptive HALT/CONTINUE with controller
    """
    if scale == "base":
        from arc.recurrence.base import BaseLM
        return BaseLM(adapter)

    if adaptive:
        controller = make_controller(scale, adapter, max_loops=max_loops, compute_budget=compute_budget)

        if scale == "model":
            from arc.recurrence.adaptive import ModelAdaptiveRecurrenceLM
            return ModelAdaptiveRecurrenceLM(adapter, controller)
        if scale == "block":
            from arc.recurrence.adaptive import BlockAdaptiveRecurrenceLM
            return BlockAdaptiveRecurrenceLM(adapter, controller)
        if scale == "layer":
            from arc.recurrence.adaptive import LayerAdaptiveRecurrenceLM
            return LayerAdaptiveRecurrenceLM(adapter, controller)
        raise ValueError(f"unknown scale for adaptive: {scale}")

    # fixed
    if recurrence < 1:
        raise ValueError("recurrence must be >= 1 for fixed models")
    if scale == "model":
        from arc.recurrence.base import ModelRecurrenceLM
        return ModelRecurrenceLM(adapter, recurrence)
    if scale == "block":
        from arc.recurrence.base import BlockRecurrenceLM
        return BlockRecurrenceLM(adapter, recurrence)
    if scale == "layer":
        from arc.recurrence.base import LayerRecurrenceLM
        return LayerRecurrenceLM(adapter, recurrence)
    raise ValueError(f"unknown scale: {scale}")
