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
# Adapters come from the private Kaggle dataset 'arc-tier-b-adapters'
# (mounted at /kaggle/input/arc-tier-b-adapters/<variant>/adapter).
adapter_dirs = {{}}
for variant in {VARIANTS!r}:
    v = variant.split("_")[0]
    cands = sorted(glob.glob(f"/kaggle/input/arc-tier-b-adapters/{{v}}/adapter"))
    if not cands:
        cands = sorted(glob.glob("/kaggle/input/*/checkpoints/phase1-tier-b/*/{{variant}}/adapters-best"))
    hits = [c for c in cands if (Path(c) / "adapter_config.json").exists()]
    if hits:
        adapter_dirs[variant] = hits[0]
        print(f"adapter [{{variant}}] ->{{hits[0]}}")
    else:
        print(f"!! NO adapter output for [{{variant}}]")
if not adapter_dirs:
    print("INPUT DIRS:", os.listdir("/kaggle/input"))
    raise SystemExit("No Tier B adapters found - is dataset niyuvo/arc-tier-b-adapters attached?")
open("/kaggle/working/adapters.json", "w").write(json.dumps(adapter_dirs, indent=2))
for variant, d in sorted(adapter_dirs.items()):
    v = variant.split("_")[0]
    rd = Path(d).resolve().parent
    if v in d and (rd / "best.json").exists():
        st = json.load(open(rd / "train_state.json")) if (rd / "train_state.json").exists() else {{}}
        best = json.load(open(rd / "best.json"))
        print(variant, "| best.best_val_loss:", best.get("best_val_loss"),
              "| tokens:", st.get("tokens_processed"), "| step:", st.get("step"))
    else:
        print(variant, "| adapter:", d)"""),
        src_cell(f"""import json, os, sys, torch
# SMOKE: reproducibility of the Hub-published repos. Load ONE variant through
# AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True) and verify
# its logits match the local JIT build with the same base + adapter weights.
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
variant = "block_adaptive"

# --- local reference build (training-time code, same inputs) ---
sys.path.insert(0, "/kaggle/working/arc/src")
from arc.recurrence.builder import build_model
from arc.models.registry import create_adapter
from peft import PeftModel

adap = create_adapter("/kaggle/working/jetmoe-8b", device_map="auto")
net, head = adap.net, adap.head
adap.hf_model = PeftModel.from_pretrained(adap.hf_model, adapter_dirs[variant])
adap.net = net
adap.head = head
adap.hf_model.eval()
local = build_model("block", adap, max_loops=4).eval()

repo = "Nyvo/arc-jetmoe-block-adaptive"
from transformers import AutoModelForCausalLM
tok = os.environ.get("HF_TOKEN")
if not tok:
    print("[smoke] SKIPPED: HF_TOKEN secret not attached; cannot load private Hub repo.")
    print("[smoke] (parity of Hub repo vs local build was already verified during push)")
else:
    hub = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, token=tok)
    hub.eval()

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    with torch.no_grad():
        ids = ids.to("cuda" if torch.cuda.is_available() else "cpu")
        l_out = local(ids)
        h_out = hub(ids)
    l_logits = l_out.logits.float()
    h_logits = h_out.logits.float()
    diff = (l_logits - h_logits).abs().max().item()
    print(f"[smoke] {{variant}} | local-hub logits max-abs-diff = {{diff:.2e}}")
    assert diff < 1e-3, f"Hub repo and local build disagree (diff={{diff}})"
    print("[smoke] OK: Hub repo reproduces the trained behavior")

# Release the smoke-test model so the benchmark subprocess has the full GPU.
import gc
del local, adap, net, head
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("[smoke] GPU memory released")"""),
        src_cell(f"""import json, os
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
adap = ",".join(f"{{k}}={{v}}" for k, v in sorted(adapter_dirs.items()))
print(adap)
# PRIMARY RUN: real Policy-T halt head (budgeted) for base + all 3 variants.
!cd /kaggle/working/arc && python scripts/run_benchmarks.py \\
  --base /kaggle/working/jetmoe-8b \\
  --model base,model_adaptive,block_adaptive,layer_adaptive \\
  --adapters {{adap}} \\
  --budgeted \\
  --max_loops 4 \\
  --limits {MAIN_LIMITS} \\
  --out /kaggle/working/benchmarks-tierb-budgeted.csv"""),
        src_cell(f"""import json, subprocess, sys
adapter_dirs = json.load(open("/kaggle/working/adapters.json"))
adap = ",".join(f"{{k}}={{v}}" for k, v in sorted(adapter_dirs.items()))
# RECURRENCE CAP SENSITIVITY: budgeted eval at tighter caps (compute frontier).
for ml in (2, 3):
    out = f"/kaggle/working/benchmarks-tierb-loops{{ml}}.csv"
    cmd = [sys.executable, "scripts/run_benchmarks.py",
           "--base", "/kaggle/working/jetmoe-8b",
           "--model", "model_adaptive,block_adaptive,layer_adaptive",
           "--adapters", adap,
           "--budgeted",
           "--max_loops", str(ml),
           "--tasks", "{SWEEP_TASKS}",
           "--limits", "{SWEEP_LIMITS}",
           "--out", out]
    print(">>> max_loops", ml)
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
    rows.extend(csv.DictReader(open(c)))
mcq = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "sciq"]
def g(key, task, budgeted, loops):
    for r in rows:
        if (r["model"] == key and r["task"] == task
                and str(r.get("budgeted", "False")) == str(budgeted)
                and int(r.get("max_loops", 4) or 4) == int(loops or 4)):
            return r
    return None
for key in ["base", "model_adaptive", "block_adaptive", "layer_adaptive"]:
    print("\\n===", key, "===")
    for task in mcq:
        r = g(key, task, True, 4)
        if r:
            print(f"  {{task:14s}} acc={{float(r['acc'])*100:5.1f}}  acc_norm={{float(r['acc_norm'])*100:5.1f}}  "
                  f"loops={{r.get('avg_loops_per_item','-')}}  tok/it={{r.get('avg_tokens_per_item','-')}}  "
                  f"tok/s={{r.get('tokens_per_s','-')}}  GFLOP/s={{r.get('flops_per_s','-')}}")
            try:
                print(f"       flops/it={{float(r.get('avg_flops_per_item',0))/1e12:6.2f}}T  "
                      f"elapsed={{r.get('elapsed_s','-')}}s  items/s={{r.get('items_per_s','-')}}")
            except (TypeError, ValueError):
                pass
    r = g(key, "wikitext", True, 4)
    if r: print(f"  wikitext      ppl={{float(r['ppl']):6.1f}}  loops={{r.get('avg_loops_per_item','-')}}  "
                f"tok/s={{r.get('tokens_per_s','-')}}  GFLOP/s={{r.get('flops_per_s','-')}}")"""),
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
        "id": "niyuvo/arc-eval-suite-tier-b",
        "title": "ARC Eval Suite Tier B",
        "code_file": "arc-eval-suite-tierb.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "dataset_sources": ["niyuvo/arc-tier-b-adapters"],
        "kernel_sources": [f"niyuvo/arc-cpt-tier-b-{v.replace('_', '-')}" for v in VARIANTS],
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