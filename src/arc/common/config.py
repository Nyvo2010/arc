from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path) -> dict:
    cfg: dict = {}
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    return cfg
