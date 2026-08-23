from __future__ import annotations

import sys
sys.path.insert(0, "src")

from abc import ABC
from arc.models.base import ARCAdapter
from arc.models.jetmoe import JetMoeAdapter


def test_adapter_contract():
    from arc.models.jetmoe import build_tiny_jetmoe
    adapter = JetMoeAdapter(build_tiny_jetmoe())
    methods = [
        "embed","forward_native","prepare","forward_layer","forward_block","forward_model",
        "normalize","project_logits","final_logits","num_layers","num_blocks",
        "lm_head_flops_per_token","unit_flops"
    ]
    for m in methods:
        assert hasattr(adapter, m), f"missing {m}"
    assert isinstance(adapter, ARCAdapter)
    print("contract ok")


def test_kaggle_config_has_block_size():
    from arc.common.config import load_config
    config = load_config("configs/kaggle.yaml")
    assert config["model"]["block_size"] == 4


if __name__ == "__main__":
    test_adapter_contract()
