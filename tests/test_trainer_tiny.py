"""End-to-end Stage-A trainer smoke test on the tiny model (CPU, synthetic)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.training.trainer import train_stage_a


def _cfg(seq_len=16):
    return {
        "model": {"path": "tiny", "block_size": 2},
        "arc": {"seed": 0},
        "cpt": {
            "method": "none", "lr": 1e-4, "warmup_ratio": 0.1, "seq_len": seq_len,
            "effective_batch": 2, "micro_batch": 1, "gradient_checkpointing": False,
            "tier_a_tokens": 10**9, "log_every": 1, "eval_every": 2,
            "checkpoint_every": 100, "per_loop_loss_every": 1, "max_grad_norm": 1.0,
            "output_dir": "unused",
        },
        "data": {"synthetic": True, "val_batches": 2},
    }


def test_trainer_runs_checkpoints_and_resumes(tmp_path):
    out = tmp_path / "ckpt"
    s1 = train_stage_a(_cfg(), "model_adaptive", out, tier="a", max_steps=2, resume=False)
    assert s1["steps"] == 2 and s1["tokens_processed"] == 2 * 16
    assert (out / "model_adaptive" / "train_metrics.csv").exists()
    assert (out / "model_adaptive" / "eval_metrics.csv").exists()
    assert (out / "model_adaptive" / "trainable_state.pt").exists()
    assert (out / "model_adaptive" / "provenance.json").exists()
    summary = json.loads((out / "model_adaptive" / "summary.json").read_text())
    assert set(map(int, summary["depth_hist"])) <= {1, 2, 3, 4}

    # Resume continues from step 2.
    s2 = train_stage_a(_cfg(), "model_adaptive", out, tier="a", max_steps=3, resume=True)
    assert s2["steps"] == 3


def test_trainer_base_control(tmp_path):
    s = train_stage_a(_cfg(), "base", tmp_path / "b", tier="a", max_steps=2, resume=False)
    assert s["depth_hist"] == {1: 2}
