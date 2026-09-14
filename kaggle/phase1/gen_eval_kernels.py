"""Generate the Kaggle eval-suite kernel (post-Tier-B benchmarks).

The eval kernel consumes the finished Tier B training kernels' outputs as
inputs, downloads the stock 4-bit JetMoE-8B base from the Hub, and runs the
full benchmark suite at depth=1 (base + 3 adaptive variants) plus a reduced
depth sweep (depth 2..4) for the adaptive variants. Results are written to
/kaggle/output/ and (optional) pushed to the private Hub repo.

Usage: python kaggle/phase1/gen_eval_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_ID = "phase1-tier-b-r1"
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]

ADAPTER_DIR_GLOB = "/kaggle/input/*/checkpoints/phase1-tier-b/*/{variant}/adapters-best"
MAIN_LIMITS = "wikitext=1000"
SWEEP_LIMITS = ("arc_easy=150,arc_challenge=150,hellaswag=150,piqa=150,"
                "winogrande=150,boolq=150,sciq=150")
SWEEP_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,sciq"


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
    print("HF_TOKEN missing (benchmarks still run; Hub push of results will skip):", e)"""),
        src_cell("""!rm -rf /kaggle/working/arc && git clone --branch stage-a-cpt https://github.com/Nyvo2010/arc.git /kaggle/working/arc
!pip install -q -r /kaggle/working/arc/requirements-kaggle.txt huggingface_hub"""),
        src_cell("""from huggingface_hub import snapshot_download
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell(f"""import glob, json, os
from pathlib import Path
adapter_dirs = {{}}
for variant in {VARIANTS!r}:
    hits = sorted(glob.glob("/kaggle/input/*/checkpoints/phase1-tier-b/*/{{variant}}/adapters-best"))
    if hits:
        adapter_dirs[variant] = hits[0]
        print(f"adapter [{{variant}}] ->{{hits[0]}}")
    else:
        print(f"!! NO adapter output for [{{variant}}]")
if not adapter_dirs:
    print("INPUT DIRS:", os.listdir("/kaggle/input"))
    raise SystemExit("No Tier B checkpoint outputs found - were the training kernels completed and attached?")
open("/kaggle/working/adapters.json", "w").write(json.dumps(adapter_dirs, indent=2))
for variant, d in sorted(adapter_dirs.items()):
    rd = Path(d).resolve().parent
    best = json.load(open(rd / "best.json")) if (rd / "best.json").exists() else {{}}
    st = json.load(open(rd / "train_state.json")) if (rd / "train_state.json").exists() else {{}}
    print(variant, "| best.best_val_loss:", best.get("best_val_loss"),
          "| tokens:", st.get("tokens_processed"), "| step:", st.get("step"))"""),
        src_cell(f"""import json
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
adap = ",".join(f"{{k}}={{v}}" for k, v in sorted(adapter_dirs.items()))
print(adap)
!cd /kaggle/working/arc && python scripts/run_benchmarks.py \\
  --base /kaggle/working/jetmoe-8b \\
  --model base,model_adaptive,block_adaptive,layer_adaptive \\
  --adapters {{adap}} \\
  --depth 1 \\
  --limits {MAIN_LIMITS} \\
  --out /kaggle/working/benchmarks-tierb-r1.csv"""),
        src_cell(f"""import json, subprocess, sys
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
adap = ",".join(f"{{k}}={{v}}" for k, v in sorted(adapter_dirs.items()))
for depth in (2, 3, 4):
    out = f"/kaggle/working/benchmarks-tierb-depth{{depth}}.csv"
    cmd = [sys.executable, "scripts/run_benchmarks.py",
           "--base", "/kaggle/working/jetmoe-8b",
           "--model", "model_adaptive,block_adaptive,layer_adaptive",
           "--adapters", adap,
           "--depth", str(depth),
           "--tasks", "{SWEEP_TASKS}",
           "--limits", "{SWEEP_LIMITS}",
           "--out", out]
    print(">>> depth", depth)
    subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)"""),
        src_cell("""import csv, glob, math, os, shutil
from pathlib import Path
os.makedirs("/kaggle/output/benchmarks", exist_ok=True)
csvs = sorted(glob.glob("/kaggle/working/benchmarks-*.csv"))
for c in csvs:
    shutil.copy(c, f"/kaggle/output/benchmarks/{{Path(c).name}}")
    print("saved", c)
rows = []
for c in csvs:
    for r in csv.DictReader(open(c)):
        if r.get("task") and r.get("key"):
            rows.append(r)
import json
def g(key, task, depth):
    for r in rows:
        if r["key"] == key and r["task"] == task and int(r.get("depth", 1) or 1) == int(depth or 1):
            return r
    return None
keys = ["base", "model_adaptive", "block_adaptive", "layer_adaptive"]
for key in keys:
    print("\\n===", key, "===")
    for task in ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]:
        r = g(key, task, 1)
        if r: print(f"  {{task:14s}} acc={{float(r['acc'])*100:5.1f}}  acc_norm={{float(r['acc_norm'])*100:5.1f}}")
    r = g(key, "wikitext_ppl", 1) or g(key, "wikitext", 1)
    if r: print(f"  wikitext      ppl={{float(r['ppl']):6.1f}}")"""),
        src_cell("""import json, os
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi
    api = HfApi()
    repo = "Nyvo/arc-jetmoe-recurrence-phase1"
    for f in sorted(glob.glob("/kaggle/output/benchmarks/*.csv")):
        api.upload_file(path_or_fileobj=f, path_in_repo=f"benches/{{os.path.basename(f)}}",
                        repo_id=repo, token=os.environ["HF_TOKEN"])
        print("pushed", f)
else:
    print("HF_TOKEN absent: results kept in Kaggle output; push manually")"""),
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
        "id": "niyuvo/arc-eval-suite-tierb",
        "title": "ARC Eval Suite Tier B",
        "code_file": "arc-eval-suite-tierb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": [f"niyuvo/arc-cpt-tier-b-{v}" for v in VARIANTS],
        "model_sources": [],
    }


def main() -> None:
    d = HERE / "eval-tierb"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arc-eval-suite-tierb.ipynb").write_text(json.dumps(build_notebook(), indent=1))
    (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()