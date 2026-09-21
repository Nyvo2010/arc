#!/usr/bin/env python3
"""Assemble + push Tier B trained ARC models to Hugging Face Hub.

Each trained variant is pushed as its own private repo so that
`config.json` lives at the repo root and `AutoModelForCausalLM.from_pretrained(
repo_id, trust_remote_code=True)` resolves the ARC architecture via auto_map:

- config.json          (auto_map -> configuration_arc.ARCAutoConfig /
                         modeling_arc.ARCJetMoeForCausalLM)
- configuration_arc.py (self-contained AutoConfig)
- modeling_arc.py      (self-contained remote-code model)
- adapter/             (LoRA adapter only; base path rewritten to jetmoe/jetmoe-8b)
- best.json, summary.json, eval_metrics.csv, provenance.json,
  train_state.json     (training provenance/metadata)

The base MoE is NOT re-uploaded; modelling downloads it at load time.

Usage:
  HF_TOKEN=... python scripts/push_tierb_to_hub.py \
    --output-dir /tmp/opencode/blk3 \
    --tier-out-dir checkpoints/phase1-tier-b \
    --variant block_adaptive \
    --repo-id Nyvo/arc-jetmoe-block-adaptive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

BASE_MODEL_ID = "jetmoe/jetmoe-8b"
MAX_LOOPS = 4
BLOCK_SIZE = 4

CONTROLLER_KWARGS = {
    "ref_js": 0.05,
    "ref_hidden": 0.1,
    "ref_entropy_delta": 0.1,
    "k": 12.0,
    "bias": 0.6,
    "halt_threshold": 0.45,
    "min_gain": 0.02,
    "w_js": 0.25,
    "w_hidden": 0.25,
    "w_top1": 0.20,
    "w_entropy": 0.15,
    "w_conf": 0.15,
}

SCALE = {
    "model_adaptive": "model",
    "block_adaptive": "block",
    "layer_adaptive": "layer",
}

METADATA = ["best.json", "summary.json", "eval_metrics.csv", "provenance.json", "train_state.json"]

SRC_ARC = Path(__file__).resolve().parent.parent / "src/arc/hf"


def parse_args():
    p = argparse.ArgumentParser(description="Assemble + push a trained Tier B ARC model to HF Hub")
    p.add_argument("--output-dir", required=True, help="Kaggle kernel output dir (contains tier-out-dir)")
    p.add_argument("--tier-out-dir", default="checkpoints/phase1-tier-b",
                   help="path of the run dir inside output-dir, e.g. checkpoints/phase1-tier-b/r1")
    p.add_argument("--variant", required=True, choices=sorted(SCALE))
    p.add_argument("--repo-id", required=True, help="private Hub repo, e.g. Nyvo/arc-jetmoe-block-adaptive")
    p.add_argument("--stage-dir", default=None, help="assemble here instead of a temp dir (debug)")
    p.add_argument("--dry-run", action="store_true", help="assemble + validate locally, skip upload")
    return p.parse_args()


def build_config(scale: str) -> dict:
    cfg = {
        "architectures": ["ARCJetMoeForCausalLM"],
        "model_type": "arc-jetmoe",
        "auto_map": {
            "AutoConfig": "configuration_arc.ARCAutoConfig",
            "AutoModelForCausalLM": "modeling_arc.ARCJetMoeForCausalLM",
        },
        "base_model_id": BASE_MODEL_ID,
        "scale": scale,
        "block_size": BLOCK_SIZE,
        "max_loops": MAX_LOOPS,
        "compute_budget": None,
        "controller_kwargs": CONTROLLER_KWARGS,
        "quantize": "4bit",
        "torch_dtype": "float16",
        "max_length": 1024,
        "use_cache": False,
        "trust_remote_code": True,
    }
    return cfg


def assemble(stage: Path, run_dir: Path, scale: str) -> None:
    stage.mkdir(parents=True, exist_ok=True)

    cfg = build_config(scale)
    (stage / "config.json").write_text(json.dumps(cfg, indent=2))

    for fname in ("modeling_arc.py", "configuration_arc.py"):
        shutil.copy(SRC_ARC / fname, stage / fname)

    adapter_src = run_dir / "adapters-best"
    if not (adapter_src / "adapter_config.json").exists():
        raise SystemExit(f"no adapters-best in {run_dir}")
    adapter_dst = stage / "adapter"
    shutil.copytree(adapter_src, adapter_dst, dirs_exist_ok=True)

    acfg = json.loads((adapter_dst / "adapter_config.json").read_text())
    acfg["base_model_name_or_path"] = BASE_MODEL_ID
    (adapter_dst / "adapter_config.json").write_text(json.dumps(acfg, indent=2))

    for fname in METADATA:
        f = run_dir / fname
        if f.exists():
            shutil.copy(f, stage / fname)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.tier_out_dir / args.variant
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir} (check --output-dir/--tier-out-dir)")
    scale = SCALE[args.variant]

    stage = Path(args.stage_dir) if args.stage_dir else Path(tempfile.mkdtemp(prefix=f"arc-{args.variant}-"))
    assemble(stage, run_dir, scale)
    print(f"[hub] staged {args.variant} (scale={scale}) at {stage}")

    if not (stage / "adapter" / "adapter_model.safetensors").exists():
        raise SystemExit(f"adapter weights missing in {stage}")
    if args.dry_run:
        print(f"[hub] dry-run OK (no upload): {list(stage.iterdir())}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("huggingface_hub required (pip install huggingface_hub)") from e

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required")
    api = HfApi(token=token)

    repo_id = args.repo_id
    try:
        api.create_repo(repo_id=repo_id, private=True, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[hub] create_repo note: {e}")
    api.upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        commit_message=f"ARC {args.variant} (tier-B best, trust_remote_code) + LoRA adapter",
    )
    print(f"[hub] pushed {args.variant} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()