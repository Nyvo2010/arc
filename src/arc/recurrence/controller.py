from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class ControllerFeatures:
    entropy: float
    entropy_delta: float
    js_divergence: float
    top1_stability: float
    hidden_cosine_change: float
    recurrence_count: int
    compute_used: float
    compute_budget: float | None = None


class RecurrenceController:
    """Deterministic HALT/CONTINUE controller.

    Returns True to continue recurrence, False to halt.
    """

    def __init__(self, max_loops: int = 8, compute_budget: float | None = None, halt_head: Any | None = None):
        self.max_loops = max_loops
        self.compute_budget = compute_budget
        self.halt_head = halt_head

    def build_features(
        self,
        logits_prev: Tensor | None,
        logits_cur: Tensor,
        hidden_prev: Tensor | None,
        hidden_cur: Tensor,
        recurrence_count: int,
        compute_used: float,
    ) -> ControllerFeatures:
        raise NotImplementedError

    def decide(self, features: ControllerFeatures, state: Any) -> bool:
        raise NotImplementedError


class ThresholdController(RecurrenceController):
    """Rule-based controller using distribution and hidden-state stability."""

    def __init__(
        self,
        max_loops: int = 8,
        compute_budget: float | None = None,
        halt_head: Any | None = None,
        js_threshold: float = 0.01,
        hidden_change_threshold: float = 0.01,
        top1_stability_threshold: float = 0.95,
        entropy_delta_threshold: float = -0.001,
    ):
        super().__init__(max_loops, compute_budget, halt_head)
        self.js_threshold = js_threshold
        self.hidden_change_threshold = hidden_change_threshold
        self.top1_stability_threshold = top1_stability_threshold
        self.entropy_delta_threshold = entropy_delta_threshold

    @staticmethod
    def _softmax(logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _entropy(p: Tensor) -> float:
        eps = 1e-12
        return float(-(p * torch.log(p + eps)).sum(dim=-1).mean().item())

    @staticmethod
    def _js_divergence(p: Tensor, q: Tensor) -> float:
        eps = 1e-12
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
        kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
        return float(((kl_pm + kl_qm) * 0.5).mean().item())

    @staticmethod
    def _cosine_change(h_prev: Tensor, h_cur: Tensor) -> float:
        if h_prev is None:
            return 1.0
        eps = 1e-12
        h_prev_n = h_prev / (h_prev.norm(dim=-1, keepdim=True) + eps)
        h_cur_n = h_cur / (h_cur.norm(dim=-1, keepdim=True) + eps)
        cos = (h_prev_n * h_cur_n).sum(dim=-1).mean().item()
        return float(1.0 - max(min(cos, 1.0), -1.0))

    def build_features(
        self,
        logits_prev: Tensor | None,
        logits_cur: Tensor,
        hidden_prev: Tensor | None,
        hidden_cur: Tensor,
        recurrence_count: int,
        compute_used: float,
    ) -> ControllerFeatures:
        p_cur = self._softmax(logits_cur)
        entropy = self._entropy(p_cur)

        if logits_prev is not None:
            p_prev = self._softmax(logits_prev)
            entropy_prev = self._entropy(p_prev)
            entropy_delta = entropy - entropy_prev
            js = self._js_divergence(p_cur, p_prev)
            top1_stability = float((p_cur.argmax(dim=-1) == p_prev.argmax(dim=-1)).float().mean().item())
        else:
            entropy_delta = 0.0
            js = 0.0
            top1_stability = 0.0

        hidden_change = self._cosine_change(hidden_prev, hidden_cur)

        return ControllerFeatures(
            entropy=entropy,
            entropy_delta=entropy_delta,
            js_divergence=js,
            top1_stability=top1_stability,
            hidden_cosine_change=hidden_change,
            recurrence_count=recurrence_count,
            compute_used=compute_used,
            compute_budget=self.compute_budget,
        )

    def decide(self, features: ControllerFeatures, state: Any) -> bool:
        if features.recurrence_count >= self.max_loops:
            return False
        if self.compute_budget is not None and features.compute_used >= self.compute_budget:
            return False

        # Optional learned halt head probability
        if self.halt_head is not None:
            # Head would be evaluated with current features; placeholder for now
            # Keep deterministic rule as fallback
            pass

        # Continue if distribution still moving or hidden state changing
        continue_signal = (
            features.js_divergence > self.js_threshold
            or features.hidden_cosine_change > self.hidden_change_threshold
            or features.top1_stability < self.top1_stability_threshold
        )
        # Allow one more step if entropy is still decreasing significantly
        if features.entropy_delta < self.entropy_delta_threshold:
            continue_signal = True

        return bool(continue_signal)
