"""Generate the halt-head calibration sweep kernel (Tier B, post-eval).

The budgeted v1 run showed the formula controller NEVER halts: every adaptive
row ran to exactly ``max_loops`` and ``avg_decide_mean_p=nan`` (fp16 logits
overflow the feature math; NaN >= threshold is always False -> CONTINUE). This
kernel sweeps ThresholdController calibration per variant over a small
benchmark subset (arc_easy + piqa), using the trained adapters from the private
Kaggle dataset ``niyuvo/arc-tier-b-adapters``.

Usage: python kaggle/phase1/gen_calib_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
TASKS = "arc_easy,piqa"
LIMITS = "arc_easy=30,piqa=30"

# (name, controller_kwargs) — tuned halting thumb: with defaults halting needs
# score >= ~0.58 which is nearly unreachable, so bias/halt_threshold/min_gain
# are lowered to make HALT fire on weakly-converged passes.
CONFIGS = [
    ("default", {"bias": 0.6, "halt_threshold": 0.45, "min_gain": 0.02}),
    ("early1", {"bias": 0.5, "halt_threshold": 0.40, "min_gain": 0.03}),
    ("early2", {"bias": 0.45, "halt_threshold": 0.40, "min_gain": 0.04}),
    ("early3", {"bias": 0.45, "k": 10, "halt_threshold": 0.35, "min_gain": 0.05}),
    ("early4", {"bias": 0.4, "k": 10, "halt_threshold": 0.35, "min_gain": 0.06}),
    ("gain8", {"bias": 0.5, "k": 12, "halt_threshold": 0.45, "min_gain": 0.08}),
    ("settle", {"bias": 0.55, "k": 12, "halt_threshold": 0.50, "min_gain": 0.02}),
]


def src_cell(src: str) -> dict:
    lines = [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def build_notebook() -> dict:
    cfgs = json.dumps(CONFIGS)
    cells = [
        src_cell("""import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN attached:", True)
except Exception as e:
    print("HF_TOKEN missing (calibration still runs):", e)"""),
        src_cell("""!rm -rf /kaggle/working/arc && git clone --branch stage-a-cpt https://github.com/Nyvo2010/arc.git /kaggle/working/arc
!pip install -q -r /kaggle/working/arc/requirements-kaggle.txt"""),
        src_cell("""from huggingface_hub import snapshot_download
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell(f"""import glob, json, os
from pathlib import Path
adapter_dirs = {{}}
for variant in {VARIANTS!r}:
    v = variant.split("_")[0]
    cands = sorted(glob.glob(f"/kaggle/input/**/{{v}}/adapter", recursive=True))
    cands += sorted(glob.glob("/kaggle/input/arc-tier-b-adapters/{{v}}/adapter"))
    hits = [c for c in cands if (Path(c) / "adapter_config.json").exists()]
    if hits:
        adapter_dirs[variant] = hits[0]
        print(f"adapter [{{variant}}] ->{{hits[0]}}")
    else:
        print(f"!! NO adapter output for [{{variant}}]")
if not adapter_dirs:
    print("INPUT DIRS:", os.listdir("/kaggle/input"))
    for root in (Path("/kaggle/input")).rglob("adapter_config.json"):
        print("found adapter_config:", root)
    print("DATASETS DIR:", os.listdir("/kaggle/input/datasets") if os.path.isdir("/kaggle/input/datasets") else "n/a")
    raise SystemExit("No adapters found - is dataset niyuvo/arc-tier-b-adapters attached?")
open("/kaggle/working/adapters.json", "w").write(json.dumps(adapter_dirs, indent=2))"""),
        src_cell(f"""import json, os, subprocess, sys
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
CONFIGS = {cfgs}
tasks = "{TASKS}"
limits = "{LIMITS}"
os.makedirs("/kaggle/working/calib", exist_ok=True)
for variant in {VARIANTS!r}:
    for cname, ckw in CONFIGS:
        out = f"/kaggle/working/calib/calib-{{variant}}-{{cname}}.csv"
        cmd = [sys.executable, "scripts/run_benchmarks.py",
               "--base", "/kaggle/working/jetmoe-8b",
               "--model", variant,
               "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
               "--budgeted", "--max_loops", "4",
               "--tasks", tasks, "--limits", limits,
               "--controller", json.dumps(ckw),
               "--out", out]
        print(">>>", variant, cname, json.dumps(ckw))
        subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)"""),
        src_cell("""import csv, glob, math, os, shutil, json
os.makedirs("/kaggle/output/calib", exist_ok=True)
csvs = sorted(glob.glob("/kaggle/working/calib/calib-*.csv"))
for c in csvs:
    shutil.copy(c, f"/kaggle/output/calib/{{os.path.basename(c)}}")
print(f"{{len(csvs)}} calibration CSVs saved")
print()
mcq = ["arc_easy", "piqa"]
for key in ["model_adaptive", "block_adaptive", "layer_adaptive"]:
    print("\\n=== ", key, " ===")
    print(f"{'config':10s} {'task':9s} acc   loops  decide  nan   halts_at")
    for c in sorted(glob.glob(f"/kaggle/working/calib/calib-{key}-*.csv")):
        cname = os.path.basename(c).replace(f"calib-{key}-", "").replace(".csv", "")
        rows = list(csv.DictReader(open(c)))
        for r in rows:
            acc = float(r["acc"]) * 100
            print(f"{cname:10s} {r['task']:9s} {acc:5.1f}%  "
                  f"{r['avg_loops_per_item']:>5} {r['avg_decide_mean_p']:>6} "
                  f"{r['nan_halts']:>4} {r['halt_hist']}")"""),
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
        "id": "niyuvo/arc-calib-suite-tier-b",
        "title": "ARC Calib Suite Tier B",
        "code_file": "arc-calib-suite-tierb.ipynb",
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
    d = HERE / "calib-tierb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arc-calib-suite-tierb.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()