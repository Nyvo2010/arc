#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2; exit 1' ERR

# ARC Kaggle Auto-Run with fail-fast and no-progress guards
# Usage: ./kaggle_run_all.sh [MODEL_PATH] [OUTPUT_DIR] [DEVICE]
# MODEL_PATH: usually a Kaggle working-directory download or read-only dataset mount
MODEL_PATH=${1:-$(ls -d /kaggle/input/*/ 2>/dev/null | head -1 || echo /kaggle/input/jetmoe-8b)}
OUTPUT_DIR=${2:-/kaggle/working/arc_results}
DEVICE=${3:-${KAGGLE_DEVICE:-cuda}}

# Kaggle limits: interactive idle ~20 min, batch max 12h CPU/GPU, 9h TPU
# We enforce early exit on errors and stalls to save compute
export PIPELINE_TIMEOUT=${PIPELINE_TIMEOUT:-1800}   # 30 min per stage max

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

# The notebook installs the package before downloading the model. Do not
# reinstall Kaggle's CUDA stack here; a second pip resolution can replace
# torch/transformers and invalidate an already-loaded runtime.
python3 - <<'PY'
import importlib
required = ("torch", "transformers", "accelerate", "safetensors", "yaml", "psutil")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing runtime dependencies: " + ", ".join(missing) + ". Re-run the notebook install cell.")
print("[INFO] Runtime dependencies already installed")
PY
echo "[1/4] Installation check complete"

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

# Unified experiment matrix: 7 variants, fixed ones at R in {2,3,4}, all metrics
echo "[2/5] Experiment matrix (7 variants, full metrics)"
timeout "$PIPELINE_TIMEOUT" python3 scripts/benchmark_matrix.py --source "$MODEL_PATH" --device "$DEVICE" --outdir "$OUTPUT_DIR/matrix" --block_size 4 || { echo "Matrix benchmark failed/timeout"; exit 1; }
[[ -f "$OUTPUT_DIR/matrix/matrix_results.csv" ]] || { echo "No matrix output"; exit 1; }

# Aggregate summary (numbers, not just names)
echo "[3/5] Aggregating summary"
python3 - <<PY
import csv, pathlib
out = pathlib.Path("$OUTPUT_DIR")
summary = out/"summary.csv"
with summary.open("w", newline="") as f:
    w = csv.writer(f)
    src = out/"matrix"/"matrix_results.csv"
    if src.exists():
        with src.open() as fh:
            r = csv.reader(fh)
            for i, row in enumerate(r):
                w.writerow(row)
print("Wrote", summary)
PY

# LM Eval: per-R accuracy on trimmed benchmarks (limit keeps it in one session).
# Matrix (stage 2) already covers tokens/FLOPs/RAM per R; this adds accuracy.
# Default tasks are loglikelihood-only. gsm8k is generative through the
# recurrence wrapper (no KV cache) and cannot finish inside PIPELINE_TIMEOUT;
# enable it via LM_EVAL_TASKS if you accept timeouts on that task only —
# scores now save incrementally per task, so completed tasks survive anyway.
echo "[4/5] LM Eval benchmarks (trimmed, limit=${LM_EVAL_LIMIT:-200}/task)"
TASKS="${LM_EVAL_TASKS:-arc_challenge,mmlu}"
mkdir -p "$OUTPUT_DIR/lm_eval"
EVAL_CONFIGS=(
  "base|"
  "model_fixed|--recurrence 2" "model_fixed|--recurrence 3" "model_fixed|--recurrence 4"
  "block_fixed|--recurrence 2" "block_fixed|--recurrence 3" "block_fixed|--recurrence 4"
  "layer_fixed|--recurrence 2" "layer_fixed|--recurrence 3" "layer_fixed|--recurrence 4"
  "model_adaptive|" "block_adaptive|" "layer_adaptive|"
)
for ENTRY in "${EVAL_CONFIGS[@]}"; do
  V="${ENTRY%%|*}"; EXTRA="${ENTRY#*|}"
  LABEL="$V${EXTRA//--recurrence /_R}"
  echo "  LM Eval $LABEL"
  timeout "$PIPELINE_TIMEOUT" python3 -m arc.benchmarks.lm_eval_bridge --source "$MODEL_PATH" --variant "$V" $EXTRA --tasks "$TASKS" --device "$DEVICE" --max_loops 4 --seed 0 --limit "${LM_EVAL_LIMIT:-200}" --output "$OUTPUT_DIR/lm_eval/${LABEL}.json" || echo "WARN: LM Eval $LABEL failed/timeout, continuing"
done

echo "[5/5] Done ==="
find "$OUTPUT_DIR" -type f | sort
