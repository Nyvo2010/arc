"""Generate Kaggle Tier B training kernels (one per variant) + metadata.

Each kernel trains ONE variant under configs/phase1/tier_b.yaml with run-id
phase1-tier-b-r1. Rerunning the same kernel resumes the session from
train_state.json (trainer auto-resume + max_runtime_s clean stop).

Usage: python kaggle/phase1/gen_tierb_kernels.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["model_adaptive", "block_adaptive", "layer_adaptive"]
RUN_ID = "phase1-tier-b-r1"


def src_cell(src: str) -> dict:
    lines = [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def build_notebook(variant: str) -> dict:
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
        src_cell("""from huggingface_hub import snapshot_download
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir="/kaggle/working/jetmoe-8b")
print("weights ready")"""),
        src_cell("""!cd /kaggle/working/arc && bash scripts/preflight_check.sh /kaggle/working/jetmoe-8b configs/phase1/tier_b.yaml"""),
        src_cell(f"""!cd /kaggle/working/arc && python scripts/train_stage_a.py \\
  --config configs/phase1/tier_b.yaml --variant {variant} \\
  --tier b --run-id {RUN_ID}"""),
        src_cell(f"""import json, glob
for f in sorted(glob.glob("/kaggle/working/checkpoints/phase1-tier-b/{RUN_ID}/{variant}/best.json")):
    d = json.load(open(f))
    print(f, "->", {{k: d[k] for k in ("best_val_loss", "best_val_ppl", "step", "tokens_processed")}})
st = json.load(open("/kaggle/working/checkpoints/phase1-tier-b/{RUN_ID}/{variant}/train_state.json"))
print("train_state: step", st.get("step"), "tokens", st.get("tokens_processed"))"""),
        src_cell(f"""import os
if not os.environ.get("HF_TOKEN"):
    print("HF_TOKEN absent: skipping Hub push, keeping Kaggle outputs")
else:
    import subprocess
    rd = "/kaggle/working/checkpoints/phase1-tier-b/{RUN_ID}/{variant}"
    tag = "phase1-tierB-{variant}-best"
    cmd = ("cd /kaggle/working/arc && python scripts/push_best_to_hub.py "
           "--repo-id Nyvo/arc-jetmoe-recurrence-phase1 --run-dir " + rd + " --tag " + tag)
    print("pushing", rd, "->", tag)
    r = subprocess.run(cmd, shell=True)
    print("push exit:", r.returncode)"""),
        src_cell(f"!ls -R /kaggle/working/checkpoints/phase1-tier-b/{RUN_ID}/{variant} 2>/dev/null | head -40"),
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
    return {
        "id": f"niyuvo/arc-tierb-{variant}",
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
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    for variant in VARIANTS:
        d = HERE / f"tierb-{variant}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"arc-tierb-{variant}.ipynb").write_text(json.dumps(build_notebook(variant), indent=1))
        (d / "kernel-metadata.json").write_text(json.dumps(build_metadata(variant), indent=2))
        print("wrote", d)


if __name__ == "__main__":
    main()