from __future__ import annotations

from dataclasses import dataclass, field

from arc.models.base import TraceEvent


@dataclass
class RecurrenceState:
    """Runtime state of one recurrent forward pass (BUILD_PLAN section 7)."""

    scale: str
    max_executions: int
    compute_budget: float | None = None
    compute_used: float = 0.0
    executions: int = 0
    unit_loop_counts: dict[int, int] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    truncated: bool = False

    def record_execution(self, unit_index: int) -> None:
        self.executions += 1
        self.unit_loop_counts[unit_index] = self.unit_loop_counts.get(unit_index, 0) + 1
