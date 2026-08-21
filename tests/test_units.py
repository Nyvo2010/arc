from __future__ import annotations

import pytest
import torch

from arc.compute.flops import jetmoe_layer_flops_per_token, jetmoe_lm_head_flops_per_token
from arc.models.jetmoe import build_tiny_jetmoe
from arc.recurrence import BaseLM, build_recurrent_model


def test_flops_sane(layer_adapter):
    cfg = layer_adapter.cfg
    f64 = jetmoe_layer_flops_per_token(cfg, 64)
    f128 = jetmoe_layer_flops_per_token(cfg, 128)
    assert f64 > 0 and f128 > f64
    assert layer_adapter.unit_flops("model", 0, 64) == pytest.approx(2 * f64 * 64)
    assert layer_adapter.unit_flops("block", 0, 64) == pytest.approx(f64 * 64)  # block_size=1 fixture
    assert jetmoe_lm_head_flops_per_token(cfg) > 0


def test_forward_compute_includes_sequence_and_lm_head(layer_adapter):
    short = BaseLM(layer_adapter)(torch.randint(0, 128, (1, 8)))
    long = BaseLM(layer_adapter)(torch.randint(0, 128, (1, 16)))
    expected = (
        layer_adapter.unit_flops("model", 0, 8)
        + jetmoe_lm_head_flops_per_token(layer_adapter.cfg) * 8
    )
    assert short.state.compute_used == pytest.approx(expected)
    assert long.state.compute_used > short.state.compute_used * 2


def test_forward_compute_scales_with_batch(layer_adapter):
    one = BaseLM(layer_adapter)(torch.randint(0, 128, (1, 8)))
    two = BaseLM(layer_adapter)(torch.randint(0, 128, (2, 8)))
    assert two.state.compute_used == pytest.approx(one.state.compute_used * 2)


def test_flops_match_measured_ground_truth():
    """Instrument every linear/matmul of an eager native forward and compare
    with the analytic formula. Catches formula drift such as a wrong top-k
    factor (attention heads already include the top-k expansion)."""
    model = build_tiny_jetmoe(seed=0, attn_implementation="eager")
    cfg = model.config
    seq_len = 32

    measured = 0.0
    orig_linear, orig_matmul = torch.nn.functional.linear, torch.matmul

    def counting_linear(x, weight, bias=None):
        nonlocal measured
        measured += 2.0 * x.numel() * weight.shape[0]
        return orig_linear(x, weight, bias)

    def counting_matmul(a, b, *args, **kwargs):
        nonlocal measured
        result = orig_matmul(a, b, *args, **kwargs)
        measured += 2.0 * result.numel() * a.shape[-1]
        return result

    torch.nn.functional.linear, torch.matmul = counting_linear, counting_matmul
    try:
        with torch.no_grad():
            model(input_ids=torch.randint(0, cfg.vocab_size, (1, seq_len)))
    finally:
        torch.nn.functional.linear, torch.matmul = orig_linear, orig_matmul

    analytic = (
        jetmoe_layer_flops_per_token(cfg, seq_len) * cfg.num_hidden_layers
        + jetmoe_lm_head_flops_per_token(cfg)
    ) * seq_len
    assert analytic == pytest.approx(measured, rel=0.01)


def test_matched_compute_across_scales(block_adapter):
    """Core experimental invariant: at equal recurrence all scales consume
    identical compute, and recurrent x1 equals the base control."""
    ids = torch.randint(0, 128, (1, 8))
    used = {
        scale: build_recurrent_model(scale, block_adapter, 3)(ids).state.compute_used
        for scale in ("layer", "block", "model")
    }
    assert len(set(used.values())) == 1

    base = BaseLM(block_adapter)(ids).state.compute_used
    r1 = build_recurrent_model("model", block_adapter, 1)(ids).state.compute_used
    assert r1 == pytest.approx(base)
