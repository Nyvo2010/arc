from __future__ import annotations

from arc.models.base import ARCAdapter
from arc.recurrence.controller_factory import make_controller


def build_model(
    scale: str,
    adapter: ARCAdapter,
    *,
    max_loops: int = 4,
    compute_budget: float | None = None,
    controller_kwargs: dict | None = None,
):
    """Builder for the adaptive-recurrence focus of this branch.

    scale: 'base' | 'model' | 'block' | 'layer'
    'base' -> native one-pass control; anything else -> adaptive HALT/CONTINUE
    with the threshold controller (Policy-T) or a learned halt head (Policy-NN).
    """
    if scale == "base":
        from arc.recurrence.base import BaseLM
        return BaseLM(adapter)

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
    raise ValueError(f"unknown scale: {scale}")
