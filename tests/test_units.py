from __future__ import annotations

import pytest
import torch

from arc.compute.flops import jetmoe_layer_flops_per_token, jetmoe_lm_head_flops_per_token
from arc.evaluation.benchmarks import generate_arithmetic
from arc.evaluation.metrics import trajectory_metrics
from arc.recurrence.scheduler import FixedSchedule


def test_scheduler_uniform_and_hetero():
    s = FixedSchedule.uniform(6, 2)
    assert s.count_for(3) == 2 and s.total_executions == 12
    h = FixedSchedule.heterogeneous([1, 2, 4])
    assert [h.count_for(i) for i in range(3)] == [1, 2, 4]
    assert h.count_for(99) == 1


def test_flops_sane(layer_adapter):
    cfg = layer_adapter.cfg
    f64 = jetmoe_layer_flops_per_token(cfg, 64)
    f128 = jetmoe_layer_flops_per_token(cfg, 128)
    assert f64 > 0 and f128 > f64
    assert layer_adapter.unit_flops("model", 0, 64) == pytest.approx(2 * f64)
    assert layer_adapter.unit_flops("block", 0, 64) == pytest.approx(f64)  # block_size=1 fixture
    assert jetmoe_lm_head_flops_per_token(cfg) > 0


def test_arithmetic_generation_deterministic_and_valid():
    p1 = generate_arithmetic(200, seed=7)
    p2 = generate_arithmetic(200, seed=7)
    assert p1 == p2

    ops_seen = {p.op for p in p1}
    assert ops_seen == {"+", "-", "*", "/"}
    for p in p1:
        if p.op == "/":
            assert p.a % p.b == 0 and int(p.answer) == p.a // p.b
        if p.op == "-":
            assert int(p.answer) >= 0
        assert str(eval(p.prompt.replace("=", "").replace("/", "//"))) == p.answer


def test_trajectory_metrics_shape():
    hist = [torch.randn(1, 50) for _ in range(4)]
    metrics = trajectory_metrics(hist)
    assert len(metrics) == 3
    assert all(m["entropy"] >= 0 and m["kl_to_prev"] >= 0 for m in metrics)
