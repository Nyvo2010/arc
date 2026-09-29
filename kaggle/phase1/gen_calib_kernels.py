"""Generate the halt-head calibration kernel (trajectory replay).

Runs ONE GPU pass per variant that records the full-depth recurrent trajectory
per calibration item (controller features + per-choice loglikelihoods at every
loop). Because the hidden state at loop t does not depend on the halt decision,
that single recording is sufficient to replay ANY controller policy offline -
so we calibrate over hundreds of configs instead of the 4-8 that full-benchmark
sweeps allowed.

Calibration uses CALIB_TASKS (TRAIN splits), which are disjoint from every
split used for the reported benchmarks, so the chosen operating point is not
fit to the test numbers. The sweep is then re-scored on the benchmark splits
in a second (also single-pass) collection.

Usage: python kaggle/phase1/gen_calib_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
MAX_LOOPS = 8
LIMIT = 60
CALIB_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq"
# Wider, denser grid than before: offline replay makes this nearly free.
SWEEP_ARGS = (
    f"--caps 2,3,4,6,8 "
    f"--biases 0.20,0.30,0.40,0.50,0.60,0.70 "
    f"--ks 4,8,14,24 "
    f"--thresholds 0.30,0.45,0.60,0.75 "
    f"--min_gains 0.0,0.02,0.04,0.07"
)


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
    print("HF_TOKEN missing (private adapter dataset may fail):", e)"""),
        src_cell("""!rm -rf /kaggle/working/arc && git clone --branch stage-a-cpt https://github.com/Nyvo2010/arc.git /kaggle/working/arc
!pip install -q -r /kaggle/working/arc/requirements-kaggle.txt"""),
        src_cell("""from huggingface_hub import snapshot_download
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell("""import glob, json, os
from pathlib import Path
adapter_dirs = {}
for variant in %r:
    v = variant.split("_")[0]
    cands = sorted(glob.glob(f"/kaggle/input/**/{v}/adapter", recursive=True))
    hits = [c for c in cands if (Path(c) / "adapter_config.json").exists()]
    if hits:
        adapter_dirs[variant] = hits[0]
        print(f"adapter [{variant}] -> {hits[0]}")
    else:
        print(f"!! NO adapter for [{variant}]")
if not adapter_dirs:
    print("INPUT DIRS:", os.listdir("/kaggle/input"))
    raise SystemExit("no adapters found")
open("/kaggle/working/adapters.json","w").write(json.dumps(adapter_dirs, indent=2))"""
                 % (VARIANTS,)),
        src_cell("""import json, subprocess, sys
adapters = json.load(open("/kaggle/working/adapters.json"))
os.makedirs("/kaggle/output/traj", exist_ok=True)
os.makedirs("/kaggle/output/sweep", exist_ok=True)

# One collect pass per (variant, task_set). Each writes a trajectory JSONL.
for variant in %r:
    ad = f"{variant}={adapters[variant]}"
    for task_set, tasks in (("calib", "%s"), ("bench", "%s")):
        out = f"/kaggle/output/traj/{variant}-{task_set}.jsonl"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("skip (exists):", out, flush=True)
            continue
        print(f">>> collect {variant} {task_set}", flush=True)
        subprocess.run([sys.executable, "scripts/calibrate_halt.py", "collect",
                        "--base", "/kaggle/working/jetmoe-8b",
                        "--model", variant, "--adapters", ad,
                        "--task_set", task_set, "--tasks", tasks,
                        "--limit", "%d", "--max_loops", "%d",
                        "--out", out],
                       cwd="/kaggle/working/arc", check=True)

# Offline replay: the whole grid, no GPU needed.
for variant in %r:
    for task_set in ("calib", "bench"):
        traj = f"/kaggle/output/traj/{variant}-{task_set}.jsonl"
        if not os.path.exists(traj):
            print("MISSING", traj, flush=True); continue
        print(f">>> sweep {variant} {task_set}", flush=True)
        subprocess.run([sys.executable, "scripts/calibrate_halt.py", "sweep",
                        "--traj", traj,
                        "--out", f"/kaggle/output/sweep/{variant}-{task_set}-sweep.csv",
                        "%s"],
                       cwd="/kaggle/working/arc", check=True)
print("ALL DONE")"""
                 % (VARIANTS, CALIB_TASKS, CALIB_TASKS, LIMIT, MAX_LOOPS,
                    VARIANTS, SWEEP_ARGS)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_dir = HERE / "haltcal-tierb"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    path = out_dir / "arc-haltcal-tierb.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    # Own kernel slug: the sibling calib-tierb/ directory already owns
    # niyuvo/arc-calib-suite-tier-b, and `kaggle kernels push -p` reuses that
    # metadata when it is present, which would silently re-run the old sweep.
    # The id MUST match the slug the title resolves to, or SaveKernel 409s.
    meta = {
        "id": "niyuvo/arc-halt-head-calibration-tier-b",
        "title": "ARC Halt Head Calibration Tier B",
        "code_file": path.name,
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
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", path, "and kernel-metadata.json")
