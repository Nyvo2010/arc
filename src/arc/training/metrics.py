"""Instrumentation for Stage A (PLAN §5: log everything, flush incrementally).

Every training run appends one row per ``log_every`` steps to a CSV so partial
results survive Kaggle session timeouts, plus a ``provenance.json`` frozen at
start (config hash, seed, checkpoint hashes, data mix, versions).
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path


class CSVLogger:
    """Append-only CSV with header-on-create and flush-per-row."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow({k: row.get(k, "") for k in self.fieldnames})
            f.flush()

    def rows(self) -> list[dict]:
        with self.path.open() as f:
            return list(csv.DictReader(f))


TRAIN_FIELDS = [
    "step", "tokens_processed", "loss", "ppl", "depth", "lr",
    "tokens_per_s", "flops_est", "total_flops", "gpu_mem_gb", "elapsed_s",
]
EVAL_FIELDS = ["step", "tokens_processed", "val_loss", "val_ppl", "depth", "elapsed_s"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def gpu_mem_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return 0.0


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def write_provenance(output_dir: str | Path, cfg: dict, extra: dict | None = None) -> Path:
    """Freeze run provenance: config, versions, hardware, data mix."""
    import torch

    try:
        import transformers

        tf_version = transformers.__version__
    except Exception:
        tf_version = "unknown"
    try:
        import peft

        peft_version = peft.__version__
    except Exception:
        peft_version = "not-installed"

    prov = {
        "config_hash": config_hash(cfg),
        "config": cfg,
        "torch_version": torch.__version__,
        "transformers_version": tf_version,
        "peft_version": peft_version,
        "gpu": gpu_name(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    out = Path(output_dir) / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prov, indent=2))
    return out
