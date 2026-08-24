from __future__ import annotations

import sys
sys.path.insert(0, "src")

import torch
from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe
from arc.models.registry import create_adapter
from arc.models.factory import build_arc_model
from arc.models.jetmoe import verify_parity


def test_tiny_parity():
    adapter = JetMoeAdapter(build_tiny_jetmoe(seed=0))
    res = verify_parity(adapter, seq_len=16)
    assert res["ok"], f"Parity failed: {res}"
    print("parity ok", res)


def test_all_variants_smoke():
    source = "tiny"
    for scale in ["base", "block", "layer"]:
        for adaptive in [False, True]:
            if scale == "base" and adaptive:
                continue
            model, adapter = build_arc_model(
                source=source,
                scale=scale,
                adaptive=adaptive,
                max_loops=2,
                recurrence=2,
                architecture="jetmoe",
                block_size=1,
            )
            torch.manual_seed(0)
            ids = torch.randint(2, 128, (1, 16))
            out = model(ids)
            assert out.logits.shape[0] == 1
            assert out.state.executions > 0
    print("smoke ok")


if __name__ == "__main__":
    test_tiny_parity()
    test_all_variants_smoke()
