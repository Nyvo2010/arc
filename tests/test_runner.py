from __future__ import annotations

import json

from arc.evaluation.runner import run_scale, trajectory_probe


def make_cfg(tmp_path):
    return {
        "run": {"seed": 0, "device": "cpu", "output_dir": str(tmp_path / "raw")},
        "model": {"path": "models/jetmoe-8b", "dtype": "float32", "block_size": 2},
        "benchmark": {"n_problems": 6, "seed": 9},
    }


def test_run_scale_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    summary = run_scale(cfg, scale="model", name="x2", loops=2, out_dir=cfg["run"]["output_dir"], tiny=True)

    assert summary["executions"] == 2
    assert summary["est_flops_forward"] > 0
    assert "accuracy" in summary and "by_op" in summary
    files = list((tmp_path / "raw").glob("*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in open(files[0])]
    kinds = {r["type"] for r in records}
    assert {"meta", "summary", "problem"} <= kinds
    meta = next(r for r in records if r["type"] == "meta")
    assert meta["hardware"]["device"] == "cpu"
    assert len(next(r for r in records if r["type"] == "problem")["prompt"]) > 0


def test_trajectory_probe_structure(tmp_path):
    cfg = make_cfg(tmp_path)
    probe = trajectory_probe(cfg, scale="layer", loops=[2, 1], problem_prompt="12 + 7 =", tiny=True)
    assert probe["state"]["executions"] == 3
    assert len(probe["trace"]) == 3
    assert len(probe["trajectory"]) == 3  # one progress signal per new execution vs h0
    assert probe["router"]["num_calls"] > 0
