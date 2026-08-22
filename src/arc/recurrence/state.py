from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class RecurrenceState:
    """Runtime state of one recurrent forward pass."""

    scale: str
    compute_used: float = 0.0
    executions: int = 0
    unit_loop_counts: dict[int, int] = field(default_factory=dict)

    def record_execution(self, unit_index: int) -> None:
        self.executions += 1
        self.unit_loop_counts[unit_index] = self.unit_loop_counts.get(unit_index, 0) + 1

    @property
    def flops_per_token(self) -> float:
        # placeholder for token count; will be set by caller
        return self.compute_used

    def as_dict(self) -> dict:
        return {
            "scale": self.scale,
            "compute_used": self.compute_used,
            "executions": self.executions,
            "unit_loop_counts": dict(self.unit_loop_counts),
        }

