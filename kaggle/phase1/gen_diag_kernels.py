"""Generate a tiny diagnostic kernel (Tier B).

Purpose: verify the post-entropy-fix pipeline end-to-end (forced-depth refs,
budgeted adaptive path, resumable run()/manifest pattern) on minimal GPU
before pushing the full 30-config spread sweep.

Scope: model_adaptive only, arc_easy+piqa x10 items, 4 runs:
  fixed1, fixed2 (forced depth), hard2b, spreadE (budgeted, max_loops=4).

Usage: python kaggle/phase1/gen_diag_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT = "model_adaptive"
TASKS = "arc_easy,piqa"
LIMITS = "arc_easy=10,piqa=10"
ADAPTIVE = [
    ("hard2b", {"bias": 0.38, "k": 12, "halt_threshold": 0.33, "min_gain": 0.07}),
    ("spreadE", {"bias": 0.52, "k": 8, "halt_threshold": 0.40, "min_gain": 0.03}),
]


def src_cell(src: str) -> dict:
    lines = [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def build_notebook() -> dict:
    cells = [
        src_cell("""import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN attached:", True)
except Exception as e:
    print("HF_TOKEN missing (benchmarks still run):", e)"""),
        src_cell("""!rm -rf /kaggle/working/arc && git clone --branch stage-a-cpt https://github.com/Nyvo2010/arc.git /kaggle/working/arc
!pip install -q -r /kaggle/working/arc/requirements-kaggle.txt"""),
        src_cell("""from huggingface_hub import snapshot_download
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell(f"""import glob, json, os
from pathlib import Path
v = "{VARIANT}".split("_")[0]
cands = sorted(glob.glob(f"/kaggle/input/**/{{v}}/adapter", recursive=True))
cands += sorted(glob.glob("/kaggle/input/arc-tier-b-adapters/{{v}}/adapter"))
hits = [c for c in cands if (Path(c) / "adapter_config.json").exists()]
assert hits, "No adapter found - is dataset niyuvo/arc-tier-b-adapters attached?"
adapter_dir = hits[0]
print(f"adapter [{VARIANT}] ->", adapter_dir)
open("/kaggle/working/adapter.json", "w").write(json.dumps(adapter_dir))"""),
        src_cell(f"""import json, os, shutil, subprocess, sys, traceback
adapter_dir = json.load(open("/kaggle/working/adapter.json"))
variant = "{VARIANT}"
tasks = "{TASKS}"
limits = "{LIMITS}"
ADAPTIVE = {json.dumps(ADAPTIVE)}
os.makedirs("/kaggle/working/diag", exist_ok=True)
os.makedirs("/kaggle/output/diag", exist_ok=True)
manifest_path = "/kaggle/output/diag/manifest.json"
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {{}}
def save_manifest():
    json.dump(manifest, open(manifest_path, "w"), indent=1)
def run(tag, cmd):
    final = f"/kaggle/output/diag/diag-{{tag}}.csv"
    if manifest.get(tag) == "ok" and os.path.exists(final):
        print("skip (done):", tag, flush=True)
        return True
    print(">>>", tag, flush=True)
    try:
        subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)
        shutil.copy(f"/kaggle/working/diag/diag-{{tag}}.csv", final)
        manifest[tag] = "ok"
        save_manifest()
        return True
    except Exception as e:
        manifest[tag] = f"FAIL: {{type(e).__name__}}: {{e}}"
        save_manifest()
        print("FAILED:", tag, flush=True)
        traceback.print_exc()
        return False
base = [sys.executable, "scripts/run_benchmarks.py",
        "--base", "/kaggle/working/jetmoe-8b",
        "--model", variant,
        "--adapters", f"{{variant}}={{adapter_dir}}",
        "--tasks", tasks, "--limits", limits]
for ml in (1, 2):
    run(f"{{variant}}-fixed{{ml}}",
        base + ["--depth", str(ml),
                "--out", f"/kaggle/working/diag/diag-{{variant}}-fixed{{ml}}.csv"])
for cname, ckw in ADAPTIVE:
    run(f"{{variant}}-{{cname}}",
        base + ["--budgeted", "--max_loops", "4",
                "--controller", json.dumps(ckw),
                "--out", f"/kaggle/working/diag/diag-{{variant}}-{{cname}}.csv"])
print("manifest:", json.dumps(manifest, indent=1))"""),
        src_cell("""import csv, glob, json, os
from collections import Counter
def merge_hist(rows):
    c = Counter()
    for r in rows:
        for part in (r.get("halt_hist", "") or "").split(";"):
            if "@" in part:
                k, v = part.split("@")
                try:
                    c[int(k)] += int(v)
                except ValueError:
                    pass
    return ";".join(f"{k}@{c[k]}" for k in sorted(c)) if c else "n/a"
mcq = ["arc_easy", "piqa"]
print(f"{'config':24s} {'acc':>5s} {'loops':>5s} {'GFLOP':>6s} {'nan':>3s}  hist")
for fp in sorted(glob.glob("/kaggle/working/diag/diag-*.csv")):
    rows = [r for r in csv.DictReader(open(fp)) if r["task"] in mcq]
    if not rows:
        print("  (empty: " + os.path.basename(fp) + ")")
        continue
    acc = sum(float(r["acc"]) * 100 for r in rows) / len(rows)
    loops = sum(float(r.get("avg_loops_per_item", 0)) for r in rows) / len(rows)
    gflop = sum(float(r.get("avg_flops_per_item", 0)) for r in rows) / len(rows) / 1e9
    nan = sum(int(float(r.get("nan_halts", 0) or 0)) for r in rows)
    print(f"{os.path.basename(fp)[5:-4]:24s} {acc:5.1f} {loops:5.1f} {gflop:6.1f} {nan:>3}  {merge_hist(rows)}")"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata() -> dict:
    return {
        "id": "niyuvo/arc-diag-tier-b",
        "title": "ARC Diag Tier B",
        "code_file": "arc-diag-tierb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": ["niyuvo/arc-tier-b-adapters"],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    d = HERE / "diag-tierb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arc-diag-tierb.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()
