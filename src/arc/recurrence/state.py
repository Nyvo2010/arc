"""Recurrence state dataclass."""
from dataclasses import dataclass

@dataclass
class RecurrenceState:
    model_loops: int = 0
    block_loops: int = 0
    layer_loops: int = 0
    compute_used: float = 0.0
    compute_budget: float = 0.0
