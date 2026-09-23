"""Generate the spread sweep kernel (Tier B, post-hard2b finding).

hard2b (bias=.38 k=12 thr=.33 mg=.07) reproduced fixed-2-loop accuracy but
collapsed every item to exactly 2 recursions (halt_hist all "2@N") - i.e. it is
a fixed-depth model, proving nothing about adaptivity. The user requirement is
that the (untrained formula) halt head produces a genuine SPREAD of recursion
amounts - some items 1, some 2, some 3, some 4 - chosen per item so that
accuracy at max_loops=4 is at least as good as every fixed-depth point at
matched compute.

This kernel: for each variant, run (a) fixed-depth budgeted caps max_loops=1,2,3
as the reference curve, (b) adaptive configs chosen for spread, (c) one
"headroom" config at max_loops=8 to show the head stops far below the cap.

Usage: python kaggle/phase1/gen_spread_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
SWEEP_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq"
SWEEP_LIMITS = ("arc_easy=40,arc_challenge=40,hellaswag=40,piqa=40,"
                "winogrande=40,boolq=40,sciq=40")

# (name, controller_kwargs) - all targeting spread across recursion counts.
# min_gain is the diminishing-returns gate: small enough that items still
# "gaining" genuinely pass loop 2 -> 3 -> 4; bias/threshold positioned so
# immediately-converged items can halt at 1 and weakly-converged at 3-4.
CONFIGS = [
    ("spreadA", {"bias": 0.50, "k": 10, "halt_threshold": 0.42, "min_gain": 0.035}),
    ("spreadB", {"bias": 0.48, "k": 10, "halt_threshold": 0.40, "min_gain": 0.040}),
    ("spreadC", {"bias": 0.45, "k": 12, "halt_threshold": 0.38, "min_gain": 0.050}),
    ("spreadE", {"bias": 0.52, "k": 8, "halt_threshold": 0.40, "min_gain": 0.030}),
    ("hard2b", {"bias": 0.38, "k": 12, "halt_threshold": 0.33, "min_gain": 0.070}),
    ("default", {"bias": 0.60, "k": 12, "halt_threshold": 0.45, "min_gain": 0.020}),
]

# headroom candidate: reuse spreadE params but let max_loops=8; the calibrated
# head must still stop around 1-4 (not march to the cap).
HEADROOM_CONFIG = ("headroomE8", {"bias": 0.52, "k": 8, "halt_threshold": 0.40, "min_gain": 0.030})


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
    raise SystemExit("No adapters found - is dataset niyuvo/arc-tier-b-adapters attached?")
open("/kaggle/working/adapters.json", "w").write(json.dumps(adapter_dirs, indent=2))"""),
        src_cell(f"""import json, os, subprocess, sys
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
CONFIGS = {json.dumps([(c[0], c[1]) for c in CONFIGS])}
HEADROOM = {json.dumps(list(HEADROOM_CONFIG))}
tasks = "{SWEEP_TASKS}"
limits = "{SWEEP_LIMITS}"
os.makedirs("/kaggle/working/spread", exist_ok=True)
# fixed-depth reference curve (budgeted caps): max_loops = 1, 2, 3
for variant in {VARIANTS!r}:
    for ml in (1, 2, 3):
        out = f"/kaggle/working/spread/spread-{{variant}}-fixed{{ml}}.csv"
        cmd = [sys.executable, "scripts/run_benchmarks.py",
               "--base", "/kaggle/working/jetmoe-8b",
               "--model", variant,
               "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
               "--budgeted", "--max_loops", str(ml),
               "--tasks", tasks, "--limits", limits,
               "--out", out]
        print(">>>", variant, "fixed", ml)
        subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)
# adaptive configs at max_loops=4
for variant in {VARIANTS!r}:
    for cname, ckw in CONFIGS:
        out = f"/kaggle/working/spread/spread-{{variant}}-{{cname}}.csv"
        cmd = [sys.executable, "scripts/run_benchmarks.py",
               "--base", "/kaggle/working/jetmoe-8b",
               "--model", variant,
               "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
               "--budgeted", "--max_loops", "4",
               "--tasks", tasks, "--limits", limits,
               "--controller", json.dumps(ckw),
               "--out", out]
        print(">>>", variant, cname, json.dumps(ckw))
        subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)
# headroom: same as spreadE but max_loops=8
hname, hkw = HEADROOM
for variant in {VARIANTS!r}:
    out = f"/kaggle/working/spread/spread-{{variant}}-{{hname}}.csv"
    cmd = [sys.executable, "scripts/run_benchmarks.py",
           "--base", "/kaggle/working/jetmoe-8b",
           "--model", variant,
           "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
           "--budgeted", "--max_loops", "8",
           "--tasks", tasks, "--limits", limits,
           "--controller", json.dumps(hkw),
           "--out", out]
    print(">>>", variant, hname, json.dumps(hkw))
    subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)"""),
        src_cell("""import csv, glob, math, os, shutil, json
os.makedirs("/kaggle/output/spread", exist_ok=True)
csvs = sorted(glob.glob("/kaggle/working/spread/spread-*.csv"))
for c in csvs:
    shutil.copy(c, f"/kaggle/output/spread/{{os.path.basename(c)}}")
print(len(csvs), "spread CSVs saved")
mcq = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]
def tag_of(fn):
    return os.path.basename(fn)[len("spread-"):].replace(".csv", "")
for key in ["model_adaptive", "block_adaptive", "layer_adaptive"]:
    files = sorted(glob.glob(f"/kaggle/working/spread/spread-{{key}}-*.csv"))
    print("\\n===", key, "===")
    print(f"{{'config':11s}} {{'acc':>5s}} {{'loops':>5s}} {{'nan':>3s}}  hist")
    rowsacc = {{}}
    for c in files:
        rows = list(csv.DictReader(open(c)))
        accs = [float(r["acc"]) * 100 for r in rows if r["task"] in mcq]
        avg = sum(accs) / len(accs)
        r0 = rows[0]
        hist = r0.get("halt_hist", "")
        print(f"{{tag_of(c):11s}} {{avg:5.1f}} {{r0['avg_loops_per_item']:>5}} {{r0['nan_halts']:>3}}  {{hist}}")
        rowsacc[tag_of(c)] = avg
    best = max(rowsacc, key=rowsacc.get)
    print("-> BEST:", best, f"{{rowsacc[best]:.1f}}%")"""),
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
        "id": "niyuvo/arc-spread-suite-tier-b",
        "title": "ARC Spread Suite Tier B",
        "code_file": "arc-spread-suite-tierb.ipynb",
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
    d = HERE / "spread-tierb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arc-spread-suite-tierb.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()