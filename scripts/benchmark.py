from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.common.config import load_config
from arc.evaluation.runner import run_scale, trajectory_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-run arithmetic benchmark with one-token answers")
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/model_fixed.yaml")
    parser.add_argument("--scale", default=None, choices=["model", "block", "layer"])
    parser.add_argument("--loops", default="1", help="int or comma-separated per-unit list")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.base, args.config)
    rec_cfg = cfg.get("recurrence", {})
    scale = args.scale or rec_cfg.get("scale", "model")
    loops = [int(x) for x in args.loops.split(",")]
    if len(loops) == 1:
        loops = loops[0]

    name = f"bench-{scale}-{'x'.join(map(str, loops if isinstance(loops, list) else [loops]))}"
    summary = run_scale(cfg, scale=scale, name=name, loops=loops,
                        out_dir=args.out or cfg.get("run", {}).get("output_dir", "results/raw"),
                        tiny=args.tiny, limit=args.limit)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
