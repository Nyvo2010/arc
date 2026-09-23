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
    nan: int = 0


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
    """Formula-based HALT controller (Policy-T, no learned parameters).

    Computes one ``converged_score`` from all stability signals, maps it to a
    halt probability via a sigmoid, and stops when ``p_halt`` clears a
    threshold. Default is STOP: a loop only runs if the sequence is clearly
    still moving (see the diminishing-returns gate). ``max_loops`` remains a
    hard safety cap, not the operative bound.

    Numerics: all feature math runs in float32 regardless of the model dtype,
    so fp16 logits / hidden states cannot overflow the entropy or JS sums.
    Non-finite features (NaN/inf in logits or hidden) are treated as a corrupted
    pass: the controller defaults to HALT (fail-safe) and counts them so sweeps
    can detect that the model diverged.

    References:
      - js / hidden change are normalized against ``ref_js`` / ``ref_hidden``
      - entropy is normalized by ``log(vocab)``
      - confidence trend by ``ref_entropy_delta`` (nats per loop)
    """

    def __init__(
        self,
        max_loops: int = 4,
        compute_budget: float | None = None,
        halt_head: Any | None = None,
        ref_js: float = 0.05,
        ref_hidden: float = 0.1,
        ref_entropy_delta: float = 0.1,
        k: float = 12.0,
        bias: float = 0.6,
        halt_threshold: float = 0.45,
        min_gain: float = 0.02,
        w_js: float = 0.25,
        w_hidden: float = 0.25,
        w_top1: float = 0.20,
        w_entropy: float = 0.15,
        w_conf: float = 0.15,
    ):
        super().__init__(max_loops, compute_budget, halt_head)
        self.ref_js = ref_js
        self.ref_hidden = ref_hidden
        self.ref_entropy_delta = ref_entropy_delta
        self.k = k
        self.bias = bias
        self.halt_threshold = halt_threshold
        self.min_gain = min_gain
        self.w_js = w_js
        self.w_hidden = w_hidden
        self.w_top1 = w_top1
        self.w_entropy = w_entropy
        self.w_conf = w_conf
        self._vocab_size: int | None = None

    def _log_vocab(self, logits: Tensor) -> float:
        if self._vocab_size is None:
            self._vocab_size = int(logits.shape[-1])
        return float(math.log(max(2, self._vocab_size)))

    @staticmethod
    def _softmax(logits: Tensor) -> Tensor:
        return torch.softmax(logits.float(), dim=-1)

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
        h_prev_f = h_prev.float()
        h_cur_f = h_cur.float()
        h_prev_n = h_prev_f / (h_prev_f.norm(dim=-1, keepdim=True) + eps)
        h_cur_n = h_cur_f / (h_cur_f.norm(dim=-1, keepdim=True) + eps)
        cos = (h_prev_n * h_cur_n).sum(dim=-1).mean().item()
        return float(1.0 - max(min(cos, 1.0), -1.0))

    @staticmethod
    def _finite(x: float, fallback: float = 0.0) -> float:
        return fallback if not math.isfinite(x) else float(x)

    def build_features(
        self,
        logits_prev: Tensor | None,
        logits_cur: Tensor,
        hidden_prev: Tensor | None,
        hidden_cur: Tensor,
        recurrence_count: int,
        compute_used: float,
    ) -> ControllerFeatures:
        self._vocab_size = int(logits_cur.shape[-1])
        p_cur = self._softmax(logits_cur)
        entropy = self._entropy(p_cur)

        nan_cnt = 0
        for probe in (logits_cur, hidden_cur):
            if probe is not None:
                t = probe.float()
                nan_cnt += int((~torch.isfinite(t)).any().item())

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
            nan=nan_cnt,
        )

    def converged_score(self, features: ControllerFeatures, logits: Tensor | None = None) -> float:
        """Instability vs convergence in [0,1]; 1 == fully settled, 0 == moving.

        Non-finite features are clamped to the "still moving" extreme (0 score)
        so a NaN/Inf logits or hidden pass can never pin the controller into an
        unconditional loop; callers may count them via ``features``.
        """
        log_v = self._log_vocab(logits) if logits is not None else 1.0

        def _bounded(x: float) -> float:
            return 0.0 if not math.isfinite(x) else max(0.0, min(float(x), 1.0))

        js_n = _bounded(features.js_divergence / max(self.ref_js, 1e-9))
        hidden_n = _bounded(features.hidden_cosine_change / max(self.ref_hidden, 1e-9))
        entropy_n = _bounded(features.entropy / max(log_v, 1e-9))
        conf_n = _bounded(-features.entropy_delta / max(self.ref_entropy_delta, 1e-9))

        score = (
            self.w_js * (1.0 - js_n)
            + self.w_hidden * (1.0 - hidden_n)
            + self.w_top1 * _bounded(features.top1_stability)
            + self.w_entropy * (1.0 - entropy_n)
            + self.w_conf * conf_n
        )
        return float(score)

    @staticmethod
    def _p_halt(score: float, k: float, bias: float) -> float:
        if not math.isfinite(score):
            return 1.0
        return 1.0 / (1.0 + math.exp(-k * (score - bias)))

    def decide(self, features: ControllerFeatures, state: Any) -> bool:
        # hard safety caps
        if features.recurrence_count >= self.max_loops:
            return False
        if self.compute_budget is not None and features.compute_used >= self.compute_budget:
            return False
        non_finite = getattr(features, "nan", 0)
        if non_finite:
            # A NaN/inf logits or hidden pass means the recurrence diverged;
            # halting here (fail-safe) beats looping a corrupted state forward.
            return False

        score = self.converged_score(features)

        # Diminishing-returns gate: only keep looping while the sequence keeps
        # adding NEW movement each pass. Otherwise bias hard toward stopping.
        key = getattr(state, "current_unit", 0)
        prev: dict[int, float] = getattr(state, "decide_scores", {})
        pscore = prev.get(key)
        prev[key] = score
        if features.recurrence_count >= 2 and pscore is not None:
            if score - pscore < self.min_gain:
                return False
        if self.max_loops is not None and features.recurrence_count >= self.max_loops:
            return False

        p_halt = self._p_halt(score, self.k, self.bias)
        if getattr(state, "decide_probs", None) is not None:
            state.decide_probs[key] = state.decide_probs.get(key, 0.0) + p_halt
        halt = p_halt >= self.halt_threshold
        return not halt
