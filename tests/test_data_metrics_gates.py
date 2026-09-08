"""Data packing + metrics logger + G2 gate tests."""

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.training.data import cycling, pack_token_ids, synthetic_batch
from arc.training.gates import evaluate_g2
from arc.training.metrics import CSVLogger, config_hash


def test_pack_token_ids():
    packs = pack_token_ids(list(range(10)), 4)
    assert packs == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert pack_token_ids([1, 2], 4) == []


def test_synthetic_batch():
    import random

    b = synthetic_batch(128, 16, 2, random.Random(0))
    assert isinstance(b, torch.Tensor) and b.shape == (2, 16) and b.dtype == torch.long
    assert int(b.min()) >= 2 and int(b.max()) < 128


def test_cycling_rebuilds():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return iter([calls["n"]])

    gen = cycling(factory)
    assert [next(gen), next(gen), next(gen)] == [1, 2, 3]
    assert calls["n"] == 3


def test_csv_logger_roundtrip(tmp_path):
    log = CSVLogger(tmp_path / "m.csv", ["step", "loss"])
    log.log({"step": 1, "loss": 2.5})
    log.log({"step": 2, "loss": 2.0})
    rows = log.rows()
    assert [r["step"] for r in rows] == ["1", "2"]
    with open(tmp_path / "m.csv") as f:
        assert f.readline().strip() == "step,loss"


def test_config_hash_stable():
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})


def _write_eval_csv(path, losses):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "tokens_processed", "val_loss", "val_ppl", "depth", "elapsed_s"])
        w.writeheader()
        for i, l in enumerate(losses):
            w.writerow({"step": i, "tokens_processed": i, "val_loss": l, "val_ppl": 1.0, "depth": 1, "elapsed_s": 0})


def test_g2_pass_fail_partial(tmp_path):
    t, c = tmp_path / "t.csv", tmp_path / "c.csv"
    _write_eval_csv(t, [5.0, 4.0, 4.5])
    _write_eval_csv(c, [5.0, 4.2, 4.1])
    r = evaluate_g2(t, c, arc_easy_treatment=40.0, arc_easy_control=41.0)
    assert r["pass"] is True
    r2 = evaluate_g2(t, c, arc_easy_treatment=30.0, arc_easy_control=41.0)
    assert r2["pass"] is False  # >2 pts worse
    _write_eval_csv(t, [5.0, 5.5])
    r3 = evaluate_g2(t, c)
    assert r3["pass"] is False  # no improvement + no probe
