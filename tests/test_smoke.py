from __future__ import annotations

import sys
sys.path.insert(0, "src")

import torch
from arc.models.registry import MODEL_VARIANTS
from arc.models.factory import build_arc_model
from arc.inference import InferenceEngine


def test_variants_interface():
    for name, cfg in MODEL_VARIANTS.items():
        scale = cfg["scale"]
        adaptive = cfg["adaptive"]
        model, adapter = build_arc_model(
            source="tiny",
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
        assert hasattr(out, "logits")
        assert hasattr(out, "state")
        assert out.logits.shape[0] == 1
    print("interface ok")


def test_inference_engine():
    engine = InferenceEngine(source="tiny", variant="base", seed=0)
    ids = torch.randint(2, 128, (1, 16))
    metrics = engine.measure(ids)
    for k in ["logits","final_hidden","compute_used","executions","elapsed_s","tokens"]:
        assert k in metrics
    print("engine ok")


if __name__ == "__main__":
    test_variants_interface()
    test_inference_engine()
