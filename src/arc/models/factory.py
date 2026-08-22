from __future__ import annotations

from arc.models.registry import create_adapter
from arc.recurrence.builder import build_model
from arc.models.base import ARCAdapter


def build_arc_model(
    source: str,
    scale: str,
    block_size: int = 4,
    device_map: str | None = "auto",
    architecture: str = "jetmoe",
    *,
    recurrence: int = 1,
    adaptive: bool = False,
    max_loops: int = 4,
    compute_budget: float | None = None,
    controller_kwargs: dict | None = None,
):
    """High-level factory returning a model compatible with the same inference engine.

    All variants expose the same forward signature:
        model(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult
    """
    adapter: ARCAdapter = create_adapter(
        source=source,
        block_size=block_size,
        device_map=device_map,
        architecture=architecture,
    )
    model = build_model(
        scale=scale,
        adapter=adapter,
        recurrence=recurrence,
        adaptive=adaptive,
        max_loops=max_loops,
        compute_budget=compute_budget,
        controller_kwargs=controller_kwargs,
    )
    return model, adapter
