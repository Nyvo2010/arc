from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.common.config import load_config
from arc.evaluation.runner import run_scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-recurrence sweeps over the arithmetic benchmark")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/model_fixed.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tiny", action="store_true", help="random-init tiny JetMoE for smoke testing (no weights)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of problems")
    args = parser.parse_args()

    cfg = load_config(args.base, args.config)
    rec = cfg["recurrence"]
    out_dir = args.out or cfg.get("run", {}).get("output_dir", "results/raw")

    for sweep in rec.get("sweeps", []):
        summary = run_scale(
            cfg,
            scale=rec["scale"],
            name=sweep["name"],
            loops=sweep["loops"],
            out_dir=out_dir,
            tiny=args.tiny,
            limit=args.limit,
        )
        print(f"[{rec['scale']}/{sweep['name']}] acc={summary['accuracy']:.4f} "
              f"flops={summary['est_flops_forward']:.3e} execs={summary['executions']}")


if __name__ == "__main__":
    main()
