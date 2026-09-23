"""Generate the calibrated-halt-head confirmation kernel (Tier B, post-calib).

Sweeps 1+2 of gen_calib_kernels.py found ``hard2b`` (bias=.38, k=12,
halt_threshold=.33, min_gain=.07) makes the "untrained" formula halt head stop
at exactly 2 recursions for every item, exactly reproducing the hard-capped
loops2 benchmark on arc_easy+piqa (model 56.7/80, block 76.7/86.7, layer
73.3/90). This kernel runs the SAME calibrated controller over the full 7-task
sweep to confirm the gain generalizes, and writes results to /kaggle/output/.

Usage: python kaggle/phase1/gen_calib_final_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
SWEEP_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq"
SWEEP_LIMITS = ("arc_easy=40,arc_challenge=40,hellaswag=40,piqa=40,"
                "winogrande=40,boolq=40,sciq=40")
MAIN_LIMITS = SWEEP_LIMITS + ",wikitext=200"
HARD2B = {"bias": 0.38, "k": 12, "halt_threshold": 0.33, "min_gain": 0.07}


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
adap = ",".join(f"{{k}}={{v}}" for k, v in sorted(adapter_dirs.items()))
hard2b = {json.dumps(HARD2B)}
os.makedirs("/kaggle/working/calib_final", exist_ok=True)
for variant in {VARIANTS!r}:
    out = f"/kaggle/working/calib_final/calibfinal-{{variant}}.csv"
    cmd = [sys.executable, "scripts/run_benchmarks.py",
           "--base", "/kaggle/working/jetmoe-8b",
           "--model", variant,
           "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
           "--budgeted", "--max_loops", "4",
           "--tasks", "{MAIN_LIMITS.split(',wikitext')[0].split('arc_easy=')[0] + 'arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq'}",
           "--limits", "{SWEEP_LIMITS}",
           "--controller", json.dumps(hard2b),
           "--out", out]
    print(">>>", variant, json.dumps(hard2b))
    subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)
# wikitext rows appended per-variant via the same controller
for variant in {VARIANTS!r}:
    out = f"/kaggle/working/calib_final/calibfinal-wt-{{variant}}.csv"
    cmd = [sys.executable, "scripts/run_benchmarks.py",
           "--base", "/kaggle/working/jetmoe-8b",
           "--model", variant,
           "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
           "--budgeted", "--max_loops", "4",
           "--tasks", "wikitext",
           "--limits", "wikitext=200",
           "--controller", json.dumps(hard2b),
           "--out", out]
    print(">>> wikitext", variant)
    subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)"""),
        src_cell("""import csv, glob, math, os, shutil, json
os.makedirs("/kaggle/output/calibfinal", exist_ok=True)
csvs = sorted(glob.glob("/kaggle/working/calib_final/calibfinal-*.csv"))
for c in csvs:
    shutil.copy(c, f"/kaggle/output/calibfinal/{{os.path.basename(c)}}")
print(f"{{len(csvs)}} final CSVs saved")
mcq = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]
for key in ["model_adaptive", "block_adaptive", "layer_adaptive"]:
    rows = []
    for c in glob.glob(f"/kaggle/working/calib_final/calibfinal-{{key}}.csv"):
        rows.extend(csv.DictReader(open(c)))
    accs = [float(r["acc"]) * 100 for r in rows if r["task"] in mcq]
    print("\\n===", key, "=== avg acc", f"{{sum(accs)/len(accs):.1f}}% over", len(accs), "tasks")
    for r in rows:
        if r["task"] in mcq:
            print(f"  {{r['task']:14s}} acc={{float(r['acc'])*100:5.1f}}  loops={{r['avg_loops_per_item']:>6}}  "
                  f"decide_p={{r.get('avg_decide_mean_p','-'):>6}}  nan={{r['nan_halts']:>3}}  hist={{r.get('halt_hist','')}}")
        else:
            print(f"  wikitext      ppl={{float(r['ppl']):6.1f}}  loops={{r['avg_loops_per_item']:>6}}  tok/s={{r.get('tokens_per_s','-')}}")"""),
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
        "id": "niyuvo/arc-calibfinal-suite-tier-b",
        "title": "ARC CalibFinal Suite Tier B",
        "code_file": "arc-calibfinal-suite-tierb.ipynb",
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
    d = HERE / "calibfinal-tierb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arc-calibfinal-suite-tierb.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()