from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch
import transformers


def detect_hardware() -> dict:
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, name = "mps", "apple-mps"
    else:
        device, name = "cpu", "cpu"
    return {"device": device, "device_name": name}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def experiment_meta(config: dict, extra: dict | None = None) -> dict:
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hardware": detect_hardware(),
        "config": config,
    }
    if extra:
        meta.update(extra)
    return meta


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
