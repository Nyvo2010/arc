"""G1 gate: ThresholdController regression tests (PLAN §6 G1).

Pins the known zero-shot pathology (iter-1 forces CONTINUE) and the
halting contract (max_loops / budget / convergence halt).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.recurrence.controller import ThresholdController


def make_ctrl(**kw):
    kw.setdefault("max_loops", 4)
    return ThresholdController(**kw)


def test_iter1_forces_continue_documented():
    """Iter 1 has no previous logits -> top1_stability=0 -> CONTINUE.

    This structural pathology (PLAN §2) is why Stage A removes the
    controller from training; the test pins the behavior so any
    controller change must confront it explicitly.
    """
    c = make_ctrl()
    cur = torch.randn(1, 8, 32)
    hid = torch.randn(1, 8, 16)
    f = c.build_features(None, cur, None, hid, 1, 0.0)
    assert f.top1_stability == 0.0
    assert c.decide(f, None) is True


def test_halts_at_max_loops():
    c = make_ctrl(max_loops=2)
    cur = torch.randn(1, 8, 32)
    hid = torch.randn(1, 8, 16)
    f = c.build_features(None, cur, None, hid, 2, 0.0)
    assert c.decide(f, None) is False


def test_halts_on_compute_budget():
    c = make_ctrl(compute_budget=10.0)
    cur = torch.randn(1, 8, 32)
    hid = torch.randn(1, 8, 16)
    f = c.build_features(None, cur, None, hid, 1, 10.0)
    assert c.decide(f, None) is False


def test_halts_at_convergence():
    """Identical consecutive passes: all stability signals say HALT."""
    c = make_ctrl()
    cur = torch.randn(1, 8, 32)
    hid = torch.randn(1, 8, 16)
    f = c.build_features(cur.clone(), cur.clone(), hid.clone(), hid.clone(), 2, 0.0)
    assert f.js_divergence == 0.0
    assert f.top1_stability == 1.0
    assert c.decide(f, None) is False


def test_features_finite():
    c = make_ctrl()
    prev = torch.randn(2, 16, 64)
    cur = torch.randn(2, 16, 64)
    f = c.build_features(prev, cur, torch.randn(2, 16, 32), torch.randn(2, 16, 32), 3, 123.0)
    for v in (f.entropy, f.entropy_delta, f.js_divergence, f.top1_stability, f.hidden_cosine_change):
        assert abs(v) != float("inf") and v == v  # finite, not NaN
    assert 0.0 <= f.top1_stability <= 1.0
    assert f.js_divergence >= 0.0
