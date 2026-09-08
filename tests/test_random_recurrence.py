"""Random-recurrence forward tests (Stage-A train path)."""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.models.registry import create_adapter
from arc.training import causal_lm_loss, random_recurrence_forward, sample_depth


def test_depth_distribution_biased_shallow():
    rng = random.Random(0)
    ds = [sample_depth(rng) for _ in range(2000)]
    assert all(1 <= d <= 4 for d in ds)
    assert 1.7 < sum(ds) / len(ds) < 2.3  # dist mean is 2.0
    assert ds.count(1) > ds.count(4)


def test_all_scales_shapes_and_grad():
    adapter = create_adapter("tiny", device_map=None)
    ids = torch.randint(2, 128, (1, 16))
    for scale, depth in (("base", 1), ("model", 2), ("block", 2), ("layer", 1)):
        out = random_recurrence_forward(adapter, scale, ids, depth=depth)
        assert out["logits"].shape == (1, 16, 128)
        assert out["logits"].isfinite().all()
        assert out["depth"] == depth
        loss = causal_lm_loss(out["logits"], ids)
        assert loss.isfinite()
        adapter.hf_model.zero_grad()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.hf_model.parameters())


def test_execution_counts():
    adapter = create_adapter("tiny", device_map=None)
    ids = torch.randint(2, 128, (1, 8))
    n_blocks = adapter.num_blocks()
    n_layers = adapter.num_layers()
    assert random_recurrence_forward(adapter, "model", ids, depth=3)["executions"] == 3
    assert random_recurrence_forward(adapter, "block", ids, depth=2)["executions"] == n_blocks * 2
    assert random_recurrence_forward(adapter, "layer", ids, depth=2)["executions"] == n_layers * 2
    assert random_recurrence_forward(adapter, "base", ids)["executions"] == 1


def test_deterministic_in_eval_mode():
    adapter = create_adapter("tiny", device_map=None)
    adapter.hf_model.eval()
    ids = torch.randint(2, 128, (1, 12))
    a = random_recurrence_forward(adapter, "model", ids, depth=2)["logits"]
    b = random_recurrence_forward(adapter, "model", ids, depth=2)["logits"]
    assert torch.equal(a, b)


def test_per_loop_logits_model_only():
    adapter = create_adapter("tiny", device_map=None)
    ids = torch.randint(2, 128, (1, 8))
    out = random_recurrence_forward(adapter, "model", ids, depth=3, track_per_loop=True)
    assert out["per_loop_logits"] is not None and len(out["per_loop_logits"]) == 3
    out_b = random_recurrence_forward(adapter, "block", ids, depth=2, track_per_loop=True)
    assert out_b["per_loop_logits"] is None
