from __future__ import annotations

import torch

from arc.models.jetmoe import JetMoeAdapter
from arc.recurrence.base import build_recurrent_model


def test_model_recurrence_runs_and_chains(layer_adapter):
    model = build_recurrent_model("model", layer_adapter, loops=3)
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 3
    assert len(result.trace) == 3
    assert len(result.last_logits_history) == 4  # h0 baseline + 3 executions
    assert result.logits.shape == (1, 8, 128)
    assert torch.isfinite(result.logits).all()
    assert all(e.scale == "model" and e.iteration >= 1 for e in result.trace)


def test_layer_recurrence_counts(layer_adapter):
    model = build_recurrent_model("layer", layer_adapter, loops=[2, 1])  # tiny model has 2 layers
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 3
    assert result.state.unit_loop_counts == {0: 2, 1: 1}
    assert [e.unit_index for e in result.trace] == [0, 0, 1]


def test_block_recurrence_counts(block_adapter):
    block_adapter.block_size = 1
    model = build_recurrent_model("block", block_adapter, loops=[2, 1])
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.state.executions == 3
    assert result.state.unit_loop_counts == {0: 2, 1: 1}
    block_adapter.block_size = 2


def test_compute_budget_truncates(layer_adapter):
    model = build_recurrent_model("model", layer_adapter, loops=4)
    model.compute_budget_flops = layer_adapter.unit_flops("model", 0, 8) * 2.5
    ids = torch.randint(0, 128, (1, 8))
    result = model(ids)

    assert result.truncated
    assert result.state.executions == 2
    assert result.state.compute_used <= model.compute_budget_flops


def test_recurrence_changes_output(layer_adapter):
    ids = torch.randint(0, 128, (1, 8))
    r1 = build_recurrent_model("model", layer_adapter, loops=1)(ids)
    r3 = build_recurrent_model("model", layer_adapter, loops=3)(ids)
    assert not torch.allclose(r1.logits, r3.logits)


def test_router_records_populated(layer_adapter):
    adapter = JetMoeAdapter(layer_adapter.hf_model, block_size=1)
    ids = torch.randint(0, 128, (1, 8))
    result = build_recurrent_model("model", adapter, loops=2)(ids)
    summary = result.router_summary

    assert summary["num_calls"] > 0  # MoA + MoE routers per layer per loop
    assert abs(sum(summary["utilization"]) - 1.0) < 1e-4
    assert summary["num_experts"] == 4
