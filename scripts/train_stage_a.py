#!/usr/bin/env python3
"""Stage-A launcher: random-recurrence CPT for one/all variants (PLAN Stage A).

Examples:
  python scripts/train_stage_a.py --config configs/kaggle.yaml --variant base --tier a
  python scripts/train_stage_a.py --config configs/kaggle.yaml --variant all --tier a
  python scripts/train_stage_a.py --config configs/kaggle.yaml --variant model_adaptive --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.common.config import load_config, validate_config  # noqa: E402
from arc.training.trainer import train_stage_a  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Stage-A random-recurrence CPT")
    p.add_argument("--config", default="configs/kaggle.yaml")
    p.add_argument("--variant", default="base",
                   choices=["base", "model_adaptive", "block_adaptive", "layer_adaptive", "all"])
    p.add_argument("--tier", default="a", choices=["a", "b"])
    p.add_argument("--output-dir", default=None, help="override cpt.output_dir")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--run-id", default=None, help="version runs as <output_dir>/<run_id>/<variant>")
    p.add_argument("--dry-run", action="store_true",
                   help="build everything and run 2 steps with a synthetic batch, then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = validate_config(load_config(args.config))
    cpt = cfg.setdefault("cpt", {})
    if args.output_dir:
        cpt["output_dir"] = args.output_dir
    if args.dry_run:
        cpt["tier_a_tokens"] = 2 * int(cpt.get("seq_len", 32)) * int(cpt.get("effective_batch", 2))
        cfg.setdefault("data", {})["synthetic"] = True
        if not args.output_dir:
            cpt["output_dir"] = "checkpoints/stage-a-dryrun"
        cfg["model"]["path"] = "tiny"
        cpt["method"] = "none"
        cpt["seq_len"] = min(int(cpt.get("seq_len", 32)), 32)
        args.max_steps = 2

    variants = (
        ["model_adaptive", "block_adaptive", "layer_adaptive"]
        if args.variant == "all" else [args.variant]
    )
    summaries = []
    for v in variants:
        summaries.append(
            train_stage_a(cfg, v, cpt.get("output_dir", "checkpoints/stage-a"),
                          tier=args.tier, max_steps=args.max_steps, resume=not args.no_resume,
                          run_id=args.run_id)
        )
    for s in summaries:
        print(f"[stage-a] DONE {s['variant']}: {s['steps']} steps, "
              f"{s['tokens_processed']:,} tokens, depths {s['depth_hist']} -> {s['run_dir']}")


if __name__ == "__main__":
    main()
