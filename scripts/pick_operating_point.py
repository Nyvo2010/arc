#!/usr/bin/env python3
"""Pick the halt-head operating point from calibration sweeps.

Input: the per-variant sweep CSVs produced by ``calibrate_halt.py sweep`` for
the held-out calibration split (calib) and the benchmark split (bench).

Method (deliberately conservative - the halt head must earn its keep):
  1. Candidates are ranked on the CALIB split only. The bench split is never
     used for selection, so the reported number is not the selected number.
  2. A candidate must have genuine spread (``distinct_loop_levels >= 2``): a
     controller that collapses to a fixed depth is a fixed-depth model wearing
     a controller costume, and we do not ship that.
  3. Among survivors, take the acc_norm-per-GFLOP Pareto front and pick the
     knee: the point whose compute is within ``--slack`` of the cheapest
     frontier point that is still within ``--acc-floor`` of the best frontier
     accuracy. This avoids both failure modes - a policy that always stops at
     depth 1 (cheap, no adaptivity) and one that always runs to the cap.
  4. Report the chosen point's numbers on BOTH splits so the generalization
     gap from calibration is visible.

Usage:
    python scripts/pick_operating_point.py --sweep-dir sweep_out --out picks.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: (float(v) if k not in ("halt_hist",) and v not in ("",) else v)
                 for k, v in row.items()} for row in csv.DictReader(f)]


def pareto(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        dominated = any(
            c is not r
            and c["acc_norm_pct"] >= r["acc_norm_pct"]
            and c["gflop_per_item"] <= r["gflop_per_item"]
            and (c["acc_norm_pct"] > r["acc_norm_pct"] or c["gflop_per_item"] < r["gflop_per_item"])
            for c in rows
        )
        if not dominated:
            out.append(r)
    out.sort(key=lambda x: x["gflop_per_item"])
    return out


def knee(front: list[dict], slack: float, acc_floor: float) -> dict | None:
    """Cheapest frontier point that is within ``acc_floor`` of the frontier max.

    ``slack`` is the relative compute premium allowed over that cheapest
    qualifying point; the chosen config is the best acc_norm/GFLOP inside it.
    """
    if not front:
        return None
    best = max(r["acc_norm_pct"] for r in front)
    qual = [r for r in front if r["acc_norm_pct"] >= best - acc_floor]
    if not qual:
        return None
    floor_cost = min(r["gflop_per_item"] for r in qual)
    budget = floor_cost * (1.0 + slack)
    pool = [r for r in qual if r["gflop_per_item"] <= budget] or qual
    return max(pool, key=lambda r: r["acc_norm_per_gflop"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="sweep_out")
    ap.add_argument("--out", default="halt-operating-points.csv")
    ap.add_argument("--slack", type=float, default=0.35,
                    help="allowed compute premium over the cheapest qualifying point")
    ap.add_argument("--acc-floor", type=float, default=1.0,
                    help="accuracy (acc_norm pts) allowed below the frontier max")
    ap.add_argument("--min-spread", type=int, default=2)
    args = ap.parse_args()

    d = Path(args.sweep_dir)
    picked = []
    for calib_path in sorted(d.glob("*-calib-sweep.csv")):
        variant = calib_path.name[: -len("-calib-sweep.csv")]
        bench_path = d / f"{variant}-bench-sweep.csv"
        calib = [r for r in load(calib_path) if r["distinct_loop_levels"] >= args.min_spread]
        if not calib:
            print(f"[{variant}] no spread-qualified candidates, skipped")
            continue
        front = pareto(calib)
        chosen = knee(front, args.slack, args.acc_floor)
        if chosen is None:
            print(f"[{variant}] no qualifying operating point")
            continue

        key = (chosen["cap"], chosen["bias"], chosen["k"],
               chosen["halt_threshold"], chosen["min_gain"])
        bench_row = None
        if bench_path.exists():
            for r in load(bench_path):
                if (r["cap"], r["bias"], r["k"],
                        r["halt_threshold"], r["min_gain"]) == key:
                    bench_row = r
                    break

        row = {
            "variant": variant,
            "cap": chosen["cap"], "bias": chosen["bias"], "k": chosen["k"],
            "halt_threshold": chosen["halt_threshold"], "min_gain": chosen["min_gain"],
            "calib_acc_norm_pct": chosen["acc_norm_pct"],
            "calib_avg_loops": chosen["avg_loops"],
            "calib_gflop_per_item": chosen["gflop_per_item"],
            "calib_acc_norm_per_gflop": chosen["acc_norm_per_gflop"],
            "calib_halt_hist": chosen["halt_hist"],
            "calib_frontier_size": len(front),
        }
        if bench_row:
            row.update({
                "bench_acc_norm_pct": bench_row["acc_norm_pct"],
                "bench_acc_pct": bench_row["acc_pct"],
                "bench_avg_loops": bench_row["avg_loops"],
                "bench_gflop_per_item": bench_row["gflop_per_item"],
                "bench_acc_norm_per_gflop": bench_row["acc_norm_per_gflop"],
                "bench_halt_hist": bench_row["halt_hist"],
                "generalization_gap_pts": round(
                    bench_row["acc_norm_pct"] - chosen["acc_norm_pct"], 2),
            })
        picked.append(row)
        print(f"[{variant}] cap={key[0]} bias={key[1]} k={key[2]} "
              f"thr={key[3]} mgain={key[4]}")
        print(f"    calib accN={chosen['acc_norm_pct']:.2f} "
              f"loops={chosen['avg_loops']:.2f} "
              f"G={chosen['gflop_per_item']:.1f} {chosen['halt_hist']}")
        if bench_row:
            print(f"    bench accN={bench_row['acc_norm_pct']:.2f} "
                  f"loops={bench_row['avg_loops']:.2f} "
                  f"G={bench_row['gflop_per_item']:.1f} "
                  f"{bench_row['halt_hist']} "
                  f"(gap {row['generalization_gap_pts']:+.2f} pts)")

    if not picked:
        print("no operating points selected")
        return
    cols = sorted({k for r in picked for k in r})
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(picked)
    print(f"wrote {len(picked)} operating points -> {out}")


if __name__ == "__main__":
    main()
