#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-}
if [[ -z "$MODEL_PATH" ]]; then
  echo "[preflight] ERROR: MODEL_PATH not provided" >&2
  exit 1
fi

echo "[preflight] Checking model path $MODEL_PATH"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[preflight] ERROR: Model directory not found" >&2
  exit 1
fi

# Check Python version >=3.11
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("[preflight] ERROR: Python >=3.11 required")
    sys.exit(1)
print(f"[preflight] Python version {sys.version.split()[0]} ok")
PY

for f in config.json model.safetensors.index.json; do
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

if command -v python3 >/dev/null; then
  echo "[preflight] python3 found"
else
  echo "[preflight] ERROR: python3 not found" >&2
  exit 1
fi

# Quick import sanity
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
try:
    import arc.models.registry as r
    assert 'base' in r.MODEL_VARIANTS
    print("[preflight] arc imports ok")
except Exception as e:
    print(f"[preflight] ERROR import {e}")
    raise
PY

echo "[preflight] All checks passed"
