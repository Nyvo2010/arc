from __future__ import annotations

import pytest

from arc.compute.flops import jetmoe_layer_flops_per_token, jetmoe_lm_head_flops_per_token


def test_flops_sane(layer_adapter):
    cfg = layer_adapter.cfg
    f64 = jetmoe_layer_flops_per_token(cfg, 64)
    f128 = jetmoe_layer_flops_per_token(cfg, 128)
    assert f64 > 0 and f128 > f64
    assert layer_adapter.unit_flops("model", 0, 64) == pytest.approx(2 * f64 * 64)
    assert layer_adapter.unit_flops("block", 0, 64) == pytest.approx(f64 * 64)  # block_size=1 fixture
    assert jetmoe_lm_head_flops_per_token(cfg) > 0


def test_forward_compute_includes_sequence_and_lm_head(layer_adapter):
    from arc.recurrence import BaseLM

    short = BaseLM(layer_adapter)(__import__("torch").randint(0, 128, (1, 8)))
    long = BaseLM(layer_adapter)(__import__("torch").randint(0, 128, (1, 16)))
    expected = (
        layer_adapter.unit_flops("model", 0, 8)
        + jetmoe_lm_head_flops_per_token(layer_adapter.cfg) * 8
    )
    assert short.state.compute_used == pytest.approx(expected)
    assert long.state.compute_used > short.state.compute_used * 2


def test_forward_compute_scales_with_batch(layer_adapter):
    import torch
    from arc.recurrence import BaseLM

    one = BaseLM(layer_adapter)(torch.randint(0, 128, (1, 8)))
    two = BaseLM(layer_adapter)(torch.randint(0, 128, (2, 8)))
    assert two.state.compute_used == pytest.approx(one.state.compute_used * 2)
