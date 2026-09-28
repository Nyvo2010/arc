"""Generate the spread sweep kernel (Tier B, post-hard2b finding).

hard2b (bias=.38 k=12 thr=.33 mg=.07) reproduced fixed-2-loop accuracy but
collapsed every item to exactly 2 recursions (halt_hist all "2@N") - i.e. it is
a fixed-depth model, proving nothing about adaptivity. The user requirement is
that the (untrained formula) halt head produces a genuine SPREAD of recursion
amounts - some items 1, some 2, some 3, some 4 - chosen per item so that
accuracy at max_loops=4 is at least as good as every fixed-depth point at
matched compute.

This kernel: for each variant, run (a) fixed-depth forced caps depth=1,2,3
as the reference curve (non-budgeted path, immune to controller defaults),
(b) adaptive configs chosen for spread, (c) one "headroom" config at
max_loops=8 to show the head stops far below the cap.

Controller note: the entropy term is live (converged_score uses the vocab
cached by build_features). Fixed refs use forced depth so they cannot be
affected by controller retuning. Runs are resumable: per-config try/except,
immediate save to /kaggle/output, manifest.json, skip-if-exists.

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
        src_cell(f"""import csv, glob, json, os, shutil, subprocess, sys
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
CONFIGS = {json.dumps([(c[0], c[1]) for c in CONFIGS])}
HEADROOM = {json.dumps(list(HEADROOM_CONFIG))}
tasks = "{SWEEP_TASKS}"
limits = "{SWEEP_LIMITS}"
os.makedirs("/kaggle/working/spread", exist_ok=True)
os.makedirs("/kaggle/output/spread", exist_ok=True)
manifest_path = "/kaggle/output/spread/manifest.json"
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {{}}
def save_manifest():
    json.dump(manifest, open(manifest_path, "w"), indent=1)
def run(tag, cmd):
    # Resumable: a failed config must not abort the sweep, and a re-pushed
    # kernel skips configs already saved (attach prior output as input).
    final = f"/kaggle/output/spread/spread-{{tag}}.csv"
    if manifest.get(tag) == "ok" and os.path.exists(final):
        print("skip (done):", tag, flush=True)
        return True
    print(">>>", tag, flush=True)
    try:
        subprocess.run(cmd, cwd="/kaggle/working/arc", check=True)
        shutil.copy(f"/kaggle/working/spread/spread-{{tag}}.csv", final)
        manifest[tag] = "ok"
        save_manifest()
        return True
    except Exception as e:
        manifest[tag] = f"FAIL: {{type(e).__name__}}: {{e}}"
        save_manifest()
        print("FAILED:", tag, e, flush=True)
        return False
# fixed-depth reference curve: FORCED depth via the non-budgeted path.
# (Old loops2/loops3 CSVs used budgeted caps relying on never-halting
# controller defaults; forced depth is immune to controller retuning.)
for variant in {VARIANTS!r}:
    for ml in (1, 2, 3):
        run(f"{{variant}}-fixed{{ml}}",
            [sys.executable, "scripts/run_benchmarks.py",
             "--base", "/kaggle/working/jetmoe-8b",
             "--model", variant,
             "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
             "--depth", str(ml),
             "--tasks", tasks, "--limits", limits,
             "--out", f"/kaggle/working/spread/spread-{{variant}}-fixed{{ml}}.csv"])
# adaptive configs at max_loops=4 (entropy term now live via cached vocab)
for variant in {VARIANTS!r}:
    for cname, ckw in CONFIGS:
        run(f"{{variant}}-{{cname}}",
            [sys.executable, "scripts/run_benchmarks.py",
             "--base", "/kaggle/working/jetmoe-8b",
             "--model", variant,
             "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
             "--budgeted", "--max_loops", "4",
             "--tasks", tasks, "--limits", limits,
             "--controller", json.dumps(ckw),
             "--out", f"/kaggle/working/spread/spread-{{variant}}-{{cname}}.csv"])
# headroom: same as spreadE but max_loops=8
hname, hkw = HEADROOM
for variant in {VARIANTS!r}:
    run(f"{{variant}}-{{hname}}",
        [sys.executable, "scripts/run_benchmarks.py",
         "--base", "/kaggle/working/jetmoe-8b",
         "--model", variant,
         "--adapters", f"{{variant}}={{adapter_dirs[variant]}}",
         "--budgeted", "--max_loops", "8",
         "--tasks", tasks, "--limits", limits,
         "--controller", json.dumps(hkw),
         "--out", f"/kaggle/working/spread/spread-{{variant}}-{{hname}}.csv"])
print("manifest:", json.dumps(manifest, indent=1))"""),
        src_cell("""import csv, glob, json, os
from collections import Counter
os.makedirs("/kaggle/output/spread", exist_ok=True)
def merge_hist(rows):
    # Halt hists count unit-turns per task row; merge across ALL tasks.
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
mcq = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]
for key in ["model_adaptive", "block_adaptive", "layer_adaptive"]:
    files = sorted(glob.glob(f"/kaggle/working/spread/spread-{key}-*.csv"))
    print(f"\\n=== {key} ===")
    print(f"{'config':11s} {'acc':>5s} {'loops':>5s} {'GFLOP':>6s} {'acc/G':>5s} {'nan':>3s}  hist")
    best, best_acc = None, -1.0
    for fp in files:
        rows = [r for r in csv.DictReader(open(fp)) if r["task"] in mcq]
        if not rows:
            print(f"  (empty: {os.path.basename(fp)})")
            continue
        accs = [float(r["acc"]) * 100 for r in rows]
        avg = sum(accs) / len(accs)
        gflop = sum(float(r.get("avg_flops_per_item", 0)) for r in rows) / len(rows) / 1e9
        loops = sum(float(r.get("avg_loops_per_item", 0)) for r in rows) / len(rows)
        nan = sum(int(float(r.get("nan_halts", 0) or 0)) for r in rows)
        tag = os.path.basename(fp)[len(f"spread-{key}-"):-4]
        print(f"{tag:11s} {avg:5.1f} {loops:5.1f} {gflop:6.1f} {avg/gflop:5.2f} {nan:>3}  {merge_hist(rows)}")
        if avg > best_acc:
            best, best_acc = tag, avg
    print(f"-> BEST: {best} {best_acc:.1f}%")"""),
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