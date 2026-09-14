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
    ap.add_argument("--out", default="benchmarks/results.csv")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    limits = {k: int(v) if v.lower() not in ("none", "all") else None
              for k, v in _parse_kv(args.limits).items()}
    adapters = _parse_kv(args.adapters)
    models = [m for m in args.model.split(",") if m]

    results = {}
    meta = {}
    for key in models:
        adapter_dir = adapters.get(key)
        print(f"[bench] evaluating {key} depth={args.depth} adapter={adapter_dir or 'none'}")
        res = evaluate_model(
            key=key,
            base_path=args.base,
            adapter_dir=adapter_dir,
            tasks=args.tasks.split(","),
            limits=limits,
            max_len=args.max_len,
            depth=args.depth,
            device_map=args.device_map,
        )
        results[key] = res
        meta[key] = {
            "variant": key, "depth": args.depth,
            "adapter": adapter_dir or "none",
            "source": args.base,
        }

    out = Path(args.out)
    write_results_csv(results, out, meta)
    print("[bench] done")


if __name__ == "__main__":
    main()