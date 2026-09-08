#!/usr/bin/env python3
"""Push versioned BEST adapters to Hugging Face Hub (Phase 1).

Uploads per variant: adapters-best/, best_state.pt, best.json,
provenance.json, eval_metrics.csv — under a unique tag so Hub keeps
every version and nothing is overwritten.

Usage:
  HF_TOKEN=... python scripts/push_best_to_hub.py \
    --repo-id Nyvo/arc-jetmoe-recurrence-phase1 \
    --run-dir /kaggle/working/checkpoints/phase1-tier-a/model_adaptive \
    --tag phase1-tierA-model_adaptive-best
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Push best ARC adapters to Hub")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "best_state.pt").exists():
        raise SystemExit(f"no best checkpoint in {run_dir} (train first)")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("huggingface_hub required (pip install huggingface_hub)") from e

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required (Kaggle Secret 'HF_TOKEN')")
    api = HfApi(token=token)

    base = f"{args.tag}"
    api.upload_folder(
        folder_path=str(run_dir / "adapters-best"),
        repo_id=args.repo_id,
        path_in_repo=f"{base}/adapters-best",
        commit_message=f"phase1 best adapters {args.tag}",
    )
    for fname in ["best_state.pt", "best.json", "provenance.json", "eval_metrics.csv",
                  "train_metrics.csv", "summary.json"]:
        f = run_dir / fname
        if f.exists():
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"{base}/{fname}",
                repo_id=args.repo_id,
                commit_message=f"phase1 best {fname} {args.tag}",
            )
    print(f"[hub] pushed {args.tag} -> {args.repo_id}")


if __name__ == "__main__":
    main()
