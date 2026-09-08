#!/usr/bin/env python3
"""G2-lite gate for Phase 1 Tier A (no base control by user decision).

Pass condition per adaptive variant: best val_loss < step-0 val_loss.
ARC-Easy probe is not implemented yet, so a full G2 pass is impossible;
this script reports 'partial' and blocks Tier B unless the loss condition
holds for ALL three adaptive variants.

Usage:
  python scripts/check_g2_lite.py --run-dir /kaggle/working/checkpoints/phase1-tier-a/phase1-tier-a-r1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]


def read_eval_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="G2-lite gate (no base control)")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)

    ok_all = True
    for v in VARIANTS:
        d = run_dir / v
        eval_csv = d / "eval_metrics.csv"
        best_json = d / "best.json"
        if not eval_csv.exists():
            print(f"[g2-lite] {v}: MISSING eval_metrics.csv -> FAIL")
            ok_all = False
            continue
        rows = read_eval_csv(eval_csv)
        if not rows:
            print(f"[g2-lite] {v}: empty eval log -> FAIL")
            ok_all = False
            continue
        step0 = float(rows[0]["val_loss"])
        best = float(json.loads(best_json.read_text())["best_val_loss"]) if best_json.exists() else min(
            float(r["val_loss"]) for r in rows
        )
        passed = best < step0
        ok_all &= passed
        print(f"[g2-lite] {v}: best {best:.4f} {'<' if passed else '>='} step-0 {step0:.4f} -> {'PASS' if passed else 'FAIL'}")

    print("[g2-lite] ARC-Easy probe: not implemented -> full G2 cannot pass (partial at best)")
    if ok_all:
        print("[g2-lite] LOSS GATE PASSED for all variants -> Tier B allowed (pending probe)")
        return 0
    print("[g2-lite] LOSS GATE FAILED -> stop, rethink objective (PLAN G2)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
