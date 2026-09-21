"""Generate Kaggle Tier B training kernels (one per variant) + metadata.

Each kernel trains ONE variant under configs/phase1/tier_b.yaml with run-id
phase1-tier-b-r1.

Session model (Kaggle wipes /kaggle/working between runs): the trainer saves
train_state.json + trainable_state.pt + optim_state.pt into the run dir on a
clean session stop (max_runtime_s). To resume, a follow-up run of the SAME
kernel self-mounts its own previous version's output as an input
(kernel_sources -> /kaggle/input/<self-slug>) and copies
/kaggle/input/<slug>/checkpoints back into /kaggle/working/checkpoints before
training. Variants with prior completed versions are regenerated with
expect_resume=True and self-mount turned on; a variant's very first launch
(expect_resume=False) uses no kernel_sources because it has no prior output.

Usage: python kaggle/phase1/gen_tierb_kernels.py [variant...]

Examples:
  python kaggle/phase1/gen_tierb_kernels.py                      # all variants
  python kaggle/phase1/gen_tierb_kernels.py model_adaptive block_adaptive  # session-2 (resume)
  python kaggle/phase1/gen_tierb_kernels.py layer_adaptive       # session-1 (fresh)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
RUN_ID = "phase1-tier-b-r1"

# Variants that already have a completed session-1 version to resume from.
EXPECT_RESUME = {
    "model_adaptive": True,
    "block_adaptive": True,
    "layer_adaptive": True,
}


def src_cell(src: str) -> dict:
    lines = [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def build_notebook(variant: str) -> dict:
    expect_resume = EXPECT_RESUME[variant]
    run_dir = f"/kaggle/working/checkpoints/phase1-tier-b/{RUN_ID}/{variant}"
    check = ""
    if expect_resume:
        check = f"""
st_path = "{run_dir}/train_state.json"
if os.path.exists(st_path):
    st = json.load(open(st_path))
    print("RESUME STATE: step", st.get("step"), "tokens", st.get("tokens_processed"))
else:
    raise SystemExit("RESUME FAILED: expected train_state.json missing at " + st_path)"""
    cells = [
        src_cell("""import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN attached:", True)
except Exception as e:
    print("HF_TOKEN missing (training continues, Hub push will skip):", e)"""),
        src_cell("""!rm -rf /kaggle/working/arc && git clone --branch stage-a-cpt https://github.com/Nyvo2010/arc.git /kaggle/working/arc
!pip install -q -r /kaggle/working/arc/requirements-kaggle.txt huggingface_hub"""),
        src_cell(f"""import glob, os, shutil
cands = sorted(glob.glob("/kaggle/input/*/jetmoe-8b/config.json"))
if cands:
    base_src = os.path.dirname(cands[0])
    print("reusing base weights from input:", base_src)
    shutil.copytree(base_src, "/kaggle/working/jetmoe-8b", dirs_exist_ok=True)
else:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell(f"""import glob, os, shutil, json
restored_any = False
for d in sorted(glob.glob("/kaggle/input/*/checkpoints")):
    print("restoring checkpoints from input:", d)
    shutil.copytree(d, "/kaggle/working/checkpoints", dirs_exist_ok=True)
    restored_any = True
if not restored_any:
    print("no prior checkpoints found as kernel input; starting fresh"){check}"""),
        src_cell("""!cd /kaggle/working/arc && bash scripts/preflight_check.sh /kaggle/working/jetmoe-8b configs/phase1/tier_b.yaml"""),
        src_cell(f"""!cd /kaggle/working/arc && python scripts/train_stage_a.py \\
  --config configs/phase1/tier_b.yaml --variant {variant} \\
  --tier b --run-id {RUN_ID}"""),
        src_cell(f"""import json, glob
for f in sorted(glob.glob("{run_dir}/best.json")):
    d = json.load(open(f))
    print(f, "->", {{k: d[k] for k in ("best_val_loss", "best_val_ppl", "step", "tokens_processed")}})
st = json.load(open("{run_dir}/train_state.json"))
print("train_state: step", st.get("step"), "tokens", st.get("tokens_processed"))"""),
        src_cell(f"""import os
if not os.environ.get("HF_TOKEN"):
    print("HF_TOKEN absent: skipping Hub push, keeping Kaggle outputs")
else:
    import subprocess
    rd = "{run_dir}"
    tag = "phase1-tierB-{variant}-best"
    cmd = ("cd /kaggle/working/arc && python scripts/push_best_to_hub.py "
           "--repo-id Nyvo/arc-jetmoe-recurrence-phase1 --run-dir " + rd + " --tag " + tag)
    print("pushing", rd, "->", tag)
    r = subprocess.run(cmd, shell=True)
    print("push exit:", r.returncode)"""),
        src_cell(f"!ls -R {run_dir} 2>/dev/null | head -40"),
        src_cell("""import shutil, os
# Keep captured output small: drop the base weights so self-mount staging for
# the next session is fast. Checkpoints (resume state) stay in the output.
shutil.rmtree("/kaggle/working/jetmoe-8b", ignore_errors=True)
print("removed /kaggle/working/jetmoe-8b from captured output")"""),
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


def build_metadata(variant: str) -> dict:
    kernel_sources = []
    if EXPECT_RESUME[variant]:
        # self-mount the previous completed version for state recovery
        kernel_sources = [f"niyuvo/arc-cpt-tier-b-{variant.replace('_', '-')}"]
    return {
        "id": f"niyuvo/arc-cpt-tier-b-{variant}",
        "title": f"ARC CPT Tier B {variant}",
        "code_file": f"arc-tierb-{variant}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }


def main(argv: list[str]) -> None:
    variants = argv or VARIANTS
    for variant in variants:
        if variant not in EXPECT_RESUME:
            raise SystemExit(f"unknown variant {variant!r}; expected one of {VARIANTS}")
        d = HERE / f"tierb-{variant}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"arc-tierb-{variant}.ipynb").write_text(json.dumps(build_notebook(variant), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(variant), indent=2))
        mode = "resume (session-2)" if EXPECT_RESUME[variant] else "fresh (session-1)"
        print(f"wrote {d} [{mode}]")


if __name__ == "__main__":
    main(sys.argv[1:])