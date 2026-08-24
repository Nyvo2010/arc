from __future__ import annotations

import sys

sys.path.insert(0, "src")

import torch

from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe
from scripts import benchmark_matrix


def test_input_device_uses_dispatched_embedding():
    model = build_tiny_jetmoe(seed=0)
    model.hf_device_map = {"": "cpu"}
    adapter = JetMoeAdapter(model)
    assert benchmark_matrix._input_device(model, adapter, "cuda") == torch.device("cpu")


def test_config_grid_has_expected_variants():
    grid = benchmark_matrix.config_grid([2, 3, 4], max_loops=4)
    assert len(grid) == 9
    assert {row["variant"] for row in grid} == {
        "base", "block_fixed", "layer_fixed",
        "block_adaptive", "layer_adaptive",
    }
