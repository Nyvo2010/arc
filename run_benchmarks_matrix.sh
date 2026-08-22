#!/usr/bin/env bash
set -euo pipefail

# Kaggle matrix run for ARC 7 variants
# Fixed variants tested with 3 iteration counts: 2, 4, 8
# Adaptive variants use max_loops 4

MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
OUTDIR=${2:-/kaggle/working/arc_results}
DEVICE=${KAGGLE_DEVICE:-cuda}

mkdir -p "$OUTDIR"

pip install -q -e .
pip install -q "lm-eval[hf]>=0.4.12"

FIXED_VARIANTS=(model_fixed block_fixed layer_fixed)
ADAPTIVE_VARIANTS=(model_adaptive block_adaptive layer_adaptive)
BASE_VARIANT=(base)

RECURRENCES=(2 4 8)

# Base
python benchmark.py \
  --source "$MODEL_PATH" \
  --device "$DEVICE" \
  --variants base \
  --seed 0 \
  --max_loops 4 \
  --recurrence 1 \
  --output "$OUTDIR/base.csv"

# Fixed matrix
for V in "${FIXED_VARIANTS[@]}"; do
  for R in "${RECURRENCES[@]}"; do
    echo "Running $V with recurrence $R ..."
    python benchmark.py \
      --source "$MODEL_PATH" \
      --device "$DEVICE" \
      --variants "$V" \
      --seed 0 \
      --max_loops 4 \
      --recurrence "$R" \
      --output "$OUTDIR/${V}_R${R}.csv"
  done
done

# Adaptive
for V in "${ADAPTIVE_VARIANTS[@]}"; do
  echo "Running $V adaptive max_loops 4 ..."
  python benchmark.py \
    --source "$MODEL_PATH" \
    --device "$DEVICE" \
    --variants "$V" \
    --seed 0 \
    --max_loops 4 \
    --recurrence 1 \
    --output "$OUTDIR/${V}.csv"
done

echo "Matrix complete."
ls -lh "$OUTDIR"
