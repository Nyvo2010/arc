#!/usr/bin/env python3
"""Combine all ARC tier-b benchmark CSVs into one summary table at repo root.

Aggregates per (model, source, config): accuracy, FLOPs, avg recursion,
recursion-spread (distinct loop levels + entropy), halt hist, nan halts.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path("/Users/niekvogelaar/Downloads/Files/CS/arc")
CSV_OUT = ROOT / "benchmarks-all-configs.csv"

MCQ = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def parse_hist(h):
    out = {}
    if not h:
        return out
    for part in str(h).split(";"):
        if "@" in part:
            k, v = part.split("@")
            try:
                out[int(k)] = int(v)
            except ValueError:
                pass
    return out


def hist_stats(h):
    d = parse_hist(h)
    if not d:
        return (0, 0, 0, 0.0)
    tot = sum(d.values())
    mn, mx = min(d), max(d)
    mean = sum(k * v for k, v in d.items()) / tot
    return (len(d), mn, mx, mean)


def entropy(h):
    d = parse_hist(h)
    if not d:
        return 0.0
    tot = sum(d.values())
    H = 0.0
    for v in d.values():
        p = v / tot
        H -= p * math.log2(p) if p > 0 else 0
    return H


def norm_entropy(h, max_loops):
    if max_loops <= 1:
        return 0.0
    return entropy(h) / math.log2(max_loops)


def add(path, source, task_label, config, model_filter=None, incl_mcq=True):
    rows = list(csv.DictReader(open(path)))
    if incl_mcq:
        rows = [r for r in rows if r["task"] in MCQ]
    if model_filter:
        rows = [r for r in rows if r["model"] == model_filter]
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    for m, mr in sorted(by.items(), key=lambda x: (x[0] != "base", x[0])):
        n_items = len(mr)
        n = int(sum(num(r["n_tot"]) for r in mr))
        acc = 100 * sum(num(r["acc"]) for r in mr) / n_items
        accn = 100 * sum(num(r["acc_norm"]) for r in mr) / n_items
        fl = sum(num(r.get("avg_flops_per_item")) for r in mr) / n_items
        lp = sum(num(r.get("avg_loops_per_item")) for r in mr) / n_items
        nan = sum(int(num(r.get("nan_halts")) or 0) for r in mr)
        hist = next((r.get("halt_hist", "") for r in mr if r.get("halt_hist")), "n/a")
        ctrl = mr[0].get("controller") or ""
        try:
            ctrl = json.dumps(json.loads(ctrl), sort_keys=True)
        except (ValueError, TypeError):
            pass
        ml = int(num(mr[0].get("max_loops")) or 0)
        out.append({
            "model": m, "source": source, "tasks": task_label, "config": config,
            "n_tasks": n_items, "n_items": n,
            "acc_pct": round(acc, 2), "acc_norm_pct": round(accn, 2),
            "avg_loops_per_item": round(lp, 3),
            "distinct_loop_levels": hist_stats(hist)[0],
            "min_loop_level": hist_stats(hist)[1],
            "max_loop_level": hist_stats(hist)[2],
            "loop_entropy_bits": round(entropy(hist), 3),
            "norm_loop_entropy": round(norm_entropy(hist, ml), 3),
            "halt_hist": hist,
            "gflop_per_item": round(fl / 1e9, 2),
            "acc_per_gflop": round(acc / (fl / 1e9), 4) if fl else 0.0,
            "nan_halts": nan,
            "max_loops": ml,
            "controller": ctrl,
        })


out = []
E = "/var/folders/7n/hcs67gyj0q9gy3xmdb_5zq6m0000gn/T/opencode/recalc"

# base model (single native pass) from the budgeted recap CSV
add(f"{E}/eval/benchmarks-tierb-budgeted.csv", "recap", "7-task", "base-1x", model_filter="base")
# fixed cache references
add(f"{E}/eval/benchmarks-tierb-loops2.csv", "recap", "7-task", "fixed-max_loops=2")
add(f"{E}/eval/benchmarks-tierb-loops3.csv", "recap", "7-task", "fixed-max_loops=3")
for _m in ["model_adaptive", "block_adaptive", "layer_adaptive"]:
        add(f"{E}/eval/benchmarks-tierb-budgeted.csv", "recap", "7-task", "budgeted-v1(never-halts)", model_filter=_m)
# calib sweep 2 (arc_easy + piqa, n=30)
for c in ["early4", "early4k8", "early45", "mixed", "soft3", "hard2b", "settle"]:
    for v in ["model", "block", "layer"]:
        add(f"{E}/calib/calib/calib-{v}_adaptive-{c}.csv", "calib-sweep", "easy+piqa", c)
# hard2b full 7-task
for v in ["model", "block", "layer"]:
    add(f"{E}/calibfinal/calib_final/calibfinal-{v}_adaptive.csv", "calib-final", "7-task", "hard2b")

cols = ["model", "source", "tasks", "config", "n_tasks", "n_items",
        "acc_pct", "acc_norm_pct",
        "avg_loops_per_item", "distinct_loop_levels", "min_loop_level", "max_loop_level",
        "loop_entropy_bits", "norm_loop_entropy", "halt_hist",
        "gflop_per_item", "acc_per_gflop", "nan_halts", "max_loops", "controller"]

with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(out, key=lambda x: (x["model"], x["config"])):
        w.writerow(r)

print(f"wrote {len(out)} rows -> {CSV_OUT}")
import collections
print(collections.Counter((r["model"], r["config"]) for r in out))