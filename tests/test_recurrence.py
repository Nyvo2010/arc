from __future__ import annotations

import torch

from arc.recurrence.base import build_recurrent_model


def test_model_recurrence_runs_and_chains(layer_adapter):
    model = build_recurrent_model("model", layer_adapter, recurrence=3)
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 3
    assert result.logits.shape == (1, 8, 128)
    assert torch.isfinite(result.logits).all()


def test_base_and_recurrence_one_match_native(layer_adapter):
    from arc.recurrence import BaseLM

    ids = torch.randint(0, 128, (1, 8))
    base = BaseLM(layer_adapter)(ids)
    for scale in ("layer", "block", "model"):
        recurrent = build_recurrent_model(scale, layer_adapter, recurrence=1)(ids)
        assert torch.allclose(recurrent.logits, base.logits, atol=1e-5)


def test_layer_recurrence_counts(layer_adapter):
    model = build_recurrent_model("layer", layer_adapter, recurrence=2)  # tiny model has 2 layers
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 4
    assert result.state.unit_loop_counts == {0: 2, 1: 2}


def test_block_recurrence_counts(block_adapter, monkeypatch):
    monkeypatch.setattr(block_adapter, "block_size", 1)  # 2 layers -> 2 blocks
    model = build_recurrent_model("block", block_adapter, recurrence=2)
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 4
    assert result.state.unit_loop_counts == {0: 2, 1: 2}


def test_recurrence_changes_output(layer_adapter):
    ids = torch.randint(0, 128, (1, 8))
    r1 = build_recurrent_model("model", layer_adapter, recurrence=1)(ids)
    r3 = build_recurrent_model("model", layer_adapter, recurrence=3)(ids)
    assert not torch.allclose(r1.logits, r3.logits)


def test_recurrence_rejects_invalid_values(layer_adapter):
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        build_recurrent_model("layer", layer_adapter, recurrence=0)
    with pytest.raises(TypeError, match="integer"):
        build_recurrent_model("layer", layer_adapter, recurrence=2.0)
