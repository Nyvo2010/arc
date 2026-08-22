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

for f in config.json model.safetensors.index.json; do
  if [[ ! -f "$MODEL_PATH/$f" ]]; then
    echo "[preflight] ERROR: Missing $f in model dir" >&2
    exit 1
  fi
done

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
try:
    import arc.models.registry as r
    assert 'base' in r.MODEL_VARIANTS
    print("[preflight] arc imports ok")
except Exception as e:
    print(f"[preflight] ERROR import {e}")
    raise
PY

echo "[preflight] All checks passed"
