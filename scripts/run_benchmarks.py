#!/usr/bin/env python3
"""Run the benchmark suite for ARC models (base + LoRA-adapted variants).

Usage:
    python scripts/run_benchmarks.py \
        --base /kaggle/working/jetmoe-8b \
        --model base,model_adaptive,block_adaptive,layer_adaptive \
        --adapters model_adaptive=/kaggle/working/ck/model_adaptive/adapters-best,... \
        --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq,wikitext \
        --limits arc_easy=300,hellaswag=500 \
        --depth 1 \
        --out /kaggle/working/benchmarks/suite1.csv

When --adapters is omitted, every model runs with no LoRA (base baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc.eval.harness import evaluate_model, write_results_csv  # noqa: E402


def _parse_kv(s: str) -> dict[str, str]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base JetMoE weights")
    ap.add_argument("--model", default="base",
                    help="comma-separated model keys (from MODEL_VARIANTS)")
    ap.add_argument("--adapters", default="",
                    help="comma-separated key=adapter_dir (optional; missing keys = no LoRA)")
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq,wikitext")
    ap.add_argument("--limits", default="",
                    help="comma-separated task=limit overrides")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max_loops", type=int, default=4,
                    help="controller max_loops cap for the budgeted path")
    ap.add_argument("--budgeted", action="store_true",
                    help="use the REAL adaptive halt-head path (Policy-T decide()) "
                         "instead of fixed-depth random recurrence")
    ap.add_argument("--controller", default="",
                    help="JSON dict of ThresholdController kwargs (calibration sweep); "
                         "e.g. '{\"bias\": 0.55, \"halt_threshold\": 0.5}'. Applied to adaptive models.")
    ap.add_argument("--out", default="benchmarks/results.csv")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    limits = {k: int(v) if v.lower() not in ("none", "all") else None
              for k, v in _parse_kv(args.limits).items()}
    adapters = _parse_kv(args.adapters)
    models = [m for m in args.model.split(",") if m]
    controller_kwargs = json.loads(args.controller) if args.controller else {}
    if controller_kwargs:
        print(f"[bench] controller_kwargs: {controller_kwargs}")

    results = {}
    meta = {}
    for key in models:
        adapter_dir = adapters.get(key)
        print(f"[bench] evaluating {key} depth={args.depth} "
              f"budgeted={args.budgeted} max_loops={args.max_loops} adapter={adapter_dir or 'none'}")
        res = evaluate_model(
            key=key,
            base_path=args.base,
            adapter_dir=adapter_dir,
            tasks=args.tasks.split(","),
            limits=limits,
            max_len=args.max_len,
            depth=args.depth,
            device_map=args.device_map,
            budgeted=args.budgeted,
            max_loops=args.max_loops,
            controller_kwargs=controller_kwargs,
        )
        results[key] = res
        meta[key] = {
            "variant": key, "depth": args.depth,
            "budgeted": str(args.budgeted), "max_loops": str(args.max_loops),
            "controller": json.dumps(controller_kwargs, sort_keys=True),
            "adapter": adapter_dir or "none",
            "source": args.base,
        }

    out = Path(args.out)
    write_results_csv(results, out, meta)
    print("[bench] done")


if __name__ == "__main__":
    main()