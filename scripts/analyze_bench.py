#!/usr/bin/env python3
"""Analyze eval-harness CSV(s): per-model avg acc, loops, GFLOP/item, acc-per-GFLOP.

The harness already records FLOPs analytically per item (avg_flops_per_item /
total_flops via adapter.unit_flops), so every benchmark run is auditable for
compute-efficiency. This script is the report layer: given one or more result
CSVs it prints a comparison table averaged over the MCQ tasks present.

Usage:
    python scripts/analyze_bench.py out1.csv out2.csv ...
    python scripts/analyze_bench.py --group config out1.csv out2.csv   # name by 'config' field
"""
import argparse
import csv
from pathlib import Path

MCQ = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]


def _num(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def analyze(path: Path, group_field: str | None = None) -> dict[str, dict]:
    rows = list(csv.DictReader(open(path)))
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("task") not in MCQ:
            continue
        key = r.get(group_field) if group_field else r.get("model")
        grouped.setdefault(key, []).append(r)
    out = {}
    for key, grp in sorted(grouped.items()):
        n = len(grp)
        acc = 100 * sum(_num(r["acc"]) for r in grp) / n
        accn = 100 * sum(_num(r["acc_norm"]) for r in grp) / n
        loops = sum(_num(r.get("avg_loops_per_item")) for r in grp) / n
        gflop = sum(_num(r.get("avg_flops_per_item")) for r in grp) / n / 1e9
        nan = sum(int(_num(r.get("nan_halts")) or 0) for r in grp)
        hist = next((r.get("halt_hist") for r in grp if r.get("halt_hist")), "")
        out[key] = {
            "acc": acc, "acc_norm": accn, "loops": loops,
            "gflop": gflop, "acc_per_gflop": acc / gflop if gflop else 0.0,
            "nan": nan, "hist": hist, "tasks": n,
        }
    return out


def render(tables: dict[str, dict]) -> None:
    hdr = f"{'model':24s} {'acc':>6s} {'accN':>6s} {'loops':>6s} {'GFLOP':>7s} {'acc/G':>6s} {'nan':>3s}  hist"
    print(hdr)
    print("-" * len(hdr))
    for label, stats in tables.items():
        s = stats
        print(f"{label:24s} {s['acc']:6.1f} {s['acc_norm']:6.1f} {s['loops']:6.1f} "
              f"{s['gflop']:7.1f} {s['acc_per_gflop']:6.3f} {s['nan']:3d}  {s['hist']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--group", default="",
                    help="group by this row field instead of 'model' (e.g. 'config')")
    args = ap.parse_args()
    for p in args.csvs:
        print(f"\n=== {p} ===")
        tables = analyze(p, group_field=args.group or None)
        render(tables)


if __name__ == "__main__":
    main()