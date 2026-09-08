#!/usr/bin/env bash
# Stage-A preflight: model weights, Python/CUDA deps, config, LoRA targets.
# Usage: scripts/preflight_check.sh <MODEL_PATH> [CONFIG]
set -euo pipefail

MODEL_PATH=${1:-}
CONFIG=${2:-configs/kaggle.yaml}
if [[ -z "$MODEL_PATH" ]]; then
  echo "[preflight] ERROR: MODEL_PATH not provided" >&2
  exit 1
fi

echo "[preflight] Checking model path $MODEL_PATH"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[preflight] ERROR: Model directory not found" >&2
  exit 1
fi

for f in config.json model.safetensors.index.json tokenizer.json; do
  if [[ ! -f "$MODEL_PATH/$f" ]]; then
    echo "[preflight] ERROR: Missing $f in model dir" >&2
    exit 1
  fi
done

python3 - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
index = json.loads((model_path / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("[preflight] ERROR: model index has no weight_map entries")
missing = [name for name in shards if not (model_path / name).is_file()]
if missing:
    raise SystemExit("[preflight] ERROR: Missing model shards: " + ", ".join(missing))
print(f"[preflight] {len(shards)} model shards present")
PY

python3 - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
import torch

print(f"[preflight] python ok, torch {torch.__version__}, cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[preflight] gpu: {torch.cuda.get_device_name(0)} "
          f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Required imports for Stage A (fail fast on Kaggle, not mid-training).
import transformers
print(f"[preflight] transformers {transformers.__version__}")
try:
    import peft
    print(f"[preflight] peft {peft.__version__}")
except ImportError:
    print("[preflight] ERROR: peft missing (needed for QLoRA)") 
    raise
try:
    import datasets
    print(f"[preflight] datasets {datasets.__version__}")
except ImportError:
    print("[preflight] ERROR: datasets missing (needed for streaming)")
    raise
if torch.cuda.is_available():
    try:
        import bitsandbytes
        print(f"[preflight] bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        raise SystemExit("[preflight] ERROR: bitsandbytes missing (needed for 8-bit on CUDA)")

# Config loads and LoRA targets are sane for JetMoE (no q/k/v/o_proj).
from arc.common.config import load_config, validate_config
cfg = validate_config(load_config(sys.argv[1]))
print(f"[preflight] config {sys.argv[1]} ok (hash fields present)")
targets = (cfg.get("cpt", {}).get("lora", {}) or {}).get("targets", [])
bad = [t for t in targets if t in ("q_proj", "k_proj", "v_proj", "o_proj")]
if bad:
    raise SystemExit(f"[preflight] ERROR: LoRA targets {bad} do not exist on JetMoE "
                     "(use ['kv_proj']); training would learn nothing")
print(f"[preflight] LoRA targets {targets} ok")

import arc.models.registry as r
assert set(r.MODEL_VARIANTS) == {"base", "model_adaptive", "block_adaptive", "layer_adaptive"}
print("[preflight] arc imports + variants ok")
PY

echo "[preflight] All checks passed"
