#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2; exit 1' ERR

# ARC Kaggle Auto-Run with fail-fast and no-progress guards
# Usage: ./kaggle_run_all.sh [MODEL_PATH] [OUTPUT_DIR] [DEVICE]
MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
OUTPUT_DIR=${2:-/kaggle/working/arc_results}
DEVICE=${3:-${KAGGLE_DEVICE:-cuda}}

# Kaggle limits: interactive idle ~20 min, batch max 12h CPU/GPU, 9h TPU
# We enforce early exit on errors and stalls to save compute
export PIPELINE_TIMEOUT=1800   # 30 min per stage max

# --- Preflight ---
echo "[INFO] Python: $(python3 --version)"
python3 - <<'PY'
import sys, json
from pathlib import Path
if sys.version_info < (3,11):
    print("[ERROR] Python >=3.11 required")
    sys.exit(1)
print(f"[INFO] Python {sys.version.split()[0]} ok, executable {sys.executable}")

# Log git commit for reproducibility
try:
    import subprocess
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    print(f"[INFO] git commit {commit}")
    Path('/kaggle/working').joinpath('git_commit.txt').write_text(commit)
except Exception as e:
    print(f"[WARN] git commit not found: {e}")
PY

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

bash scripts/preflight_check.sh "$MODEL_PATH" || { echo "Preflight failed"; exit 1; }

pip install -q -e .
pip install -q -r requirements-kaggle.txt
echo "[1/4] Installing complete"

python3 - <<'PY'
import torch, transformers, accelerate, safetensors, psutil
print(f"[INFO] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
print(f"[INFO] transformers {transformers.__version__}")
print(f"[INFO] accelerate {accelerate.__version__}")
PY

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

# Helper: run with timeout and check output produced
run_stage() {
  local name=$1
  local cmd=$2
  local out_file=$3
  echo "[$name] start"
  timeout "$PIPELINE_TIMEOUT" bash -c "$cmd" || { echo "[$name] timeout or error"; exit 1; }
  if [[ -n "$out_file" && ! -e "$out_file" ]]; then
    echo "[$name] no progress - output missing $out_file"; exit 1
  fi
  echo "[$name] done"
}

# Synthetic benchmarks
echo "[2/4] Synthetic benchmarks"
timeout "$PIPELINE_TIMEOUT" python scripts/benchmark_all.py --source "$MODEL_PATH" --device "$DEVICE" --outdir "$OUTPUT_DIR/synthetic" || { echo "Benchmark_all failed/timeout"; exit 1; }
[[ -f "$OUTPUT_DIR/synthetic/synthetic_summary.csv" ]] || { echo "No synthetic output"; exit 1; }

# Metrics benchmark
echo "[3/4] Metrics benchmark"
timeout "$PIPELINE_TIMEOUT" python scripts/benchmark_metrics.py --source "$MODEL_PATH" --device "$DEVICE" --outdir "$OUTPUT_DIR/metrics" || { echo "Benchmark_metrics failed/timeout"; exit 1; }
[[ -f "$OUTPUT_DIR/metrics/metrics_summary.csv" ]] || { echo "No metrics output"; exit 1; }

# Aggregate summary
python3 - <<PY
import csv, pathlib
out = pathlib.Path("$OUTPUT_DIR")
summary = out/"summary.csv"
with summary.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant","scale","adaptive","source"])
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
  timeout "$PIPELINE_TIMEOUT" python -m arc.benchmarks.lm_eval_bridge --source "$MODEL_PATH" --variant "$V" --tasks "$TASKS" --device "$DEVICE" --max_loops 4 --seed 0 > "$OUTPUT_DIR/lm_eval/${V}.json" 2>&1 || echo "WARN: LM Eval $V failed/timeout, continuing"
done

echo "=== Done ==="
find "$OUTPUT_DIR" -type f | sort
