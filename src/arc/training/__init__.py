"""Stage-A training: random-recurrence continued pre-training (PLAN Stage A)."""

from arc.training.random_recurrence import (
    DEPTH_DIST,
    RandomRecurrenceLM,
    causal_lm_loss,
    random_recurrence_forward,
    sample_depth,
)

__all__ = [
    "DEPTH_DIST",
    "RandomRecurrenceLM",
    "causal_lm_loss",
    "random_recurrence_forward",
    "sample_depth",
]
