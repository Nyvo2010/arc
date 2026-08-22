#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2; exit 1' ERR

# ARC auto-run for Kaggle with GPU preference
# Usage: ./kaggle_run_all.sh [MODEL_PATH] [OUTPUT_DIR] [DEVICE]
MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
OUTPUT_DIR=${2:-/kaggle/working/arc_results}
DEVICE=${3:-${KAGGLE_DEVICE:-cuda}}

# Kaggle compatibility preflight
echo "[INFO] Python: $(python3 --version)"
python3 - <<'PY'
import sys
print(f"[INFO] Python executable: {sys.executable}")
print(f"[INFO] Python version: {sys.version.split()[0]}")
PY

# Device validation - prefer GPU
if python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True; then
  DEVICE="cuda"
  echo "[INFO] CUDA available, using GPU"
else
  echo "[WARN] CUDA not available, falling back to cpu"
  DEVICE="cpu"
fi

mkdir -p "$OUTPUT_DIR/logs"
LOG="$OUTPUT_DIR/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ARC Kaggle Auto-Run ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Log: $LOG"

# Pre-flight checks
bash scripts/preflight_check.sh "$MODEL_PATH" || { echo "Preflight failed"; exit 1; }

pip install -q -e .
pip install -q -r requirements-kaggle.txt

echo "[1/4] Installing complete"

# Verify dependencies
python3 - <<'PY'
import torch, transformers, accelerate, safetensors, psutil
print(f"[INFO] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
print(f"[INFO] transformers {transformers.__version__}")
print(f"[INFO] accelerate {accelerate.__version__}")
PY

# Quick parity smoke on tiny stub to validate adapter contract
echo "[1.5/4] Parity smoke"
python3 - <<'PY'
import sys
sys.path.insert(0, 'src')
from arc.models.jetmoe import JetMoeAdapter, build_tiny_jetmoe, verify_parity
adapter = JetMoeAdapter(build_tiny_jetmoe(seed=0))
res = verify_parity(adapter, seq_len=16)
print("Parity:", res)
assert res["ok"], "Parity failed"
PY

# Synthetic benchmarks
echo "[2/4] Synthetic benchmarks"
python scripts/benchmark_all.py --source "$MODEL_PATH" --device "$DEVICE" --outdir "$OUTPUT_DIR/synthetic" || { echo "Benchmark_all failed"; exit 1; }

# Full metrics benchmark with RAM tracking
echo "[3/4] Metrics benchmark"
python scripts/benchmark_metrics.py --source "$MODEL_PATH" --device "$DEVICE" --outdir "$OUTPUT_DIR/metrics" || { echo "Benchmark_metrics failed"; exit 1; }

# Aggregate summary CSV
python3 - <<PY
import csv, pathlib
out = pathlib.Path("$OUTPUT_DIR")
rows = []
for p in [out/"synthetic"/"synthetic_summary.csv", out/"metrics"/"metrics_summary.csv"]:
    if p.exists():
        with p.open() as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
# simple summary
summary = out/"summary.csv"
with summary.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant","scale","adaptive","source"])
    # collect variants from synthetic
    syn = out/"synthetic"/"synthetic_summary.csv"
    if syn.exists():
        with syn.open() as f:
            r = csv.DictReader(f)
            for row in r:
                w.writerow([row.get("variant"), row.get("scale"), row.get("adaptive"), "synthetic"])
print("Wrote", summary)
PY

# LM Eval free tasks
echo "[4/4] LM Eval benchmarks"
TASKS="hellaswag,arc_easy,arc_challenge,gsm8k,mmlu"
mkdir -p "$OUTPUT_DIR/lm_eval"
for V in base model_fixed block_fixed layer_fixed model_adaptive block_adaptive layer_adaptive; do
  echo "  LM Eval $V"
  python -m arc.benchmarks.lm_eval_bridge --source "$MODEL_PATH" --variant "$V" --tasks "$TASKS" --device "$DEVICE" --max_loops 4 --seed 0 > "$OUTPUT_DIR/lm_eval/${V}.json" 2>&1 || echo "WARN: LM Eval $V failed, continuing"
done

echo "=== Done ==="
find "$OUTPUT_DIR" -type f | sort

