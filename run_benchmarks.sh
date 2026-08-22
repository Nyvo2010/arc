#!/usr/bin/env bash
set -euo pipefail

# Kaggle run script for ARC 7 variants
# Usage: ./run_benchmarks.sh /kaggle/working/models/jetmoe-8b results.csv

MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
OUTPUT=${2:-results.csv}
DEVICE=${KAGGLE_DEVICE:-cuda}

echo "Installing ARC..."
pip install -q -e .

echo "Checking model path..."
if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path $MODEL_PATH not found. Exiting."
  exit 1
fi

echo "Running benchmarks on all 7 variants..."
python benchmark.py \
  --source "$MODEL_PATH" \
  --device "$DEVICE" \
  --variants base model_fixed block_fixed layer_fixed model_adaptive block_adaptive layer_adaptive \
  --seed 0 \
  --max_loops 4 \
  --recurrence 2 \
  --output "$OUTPUT"

echo "Done. Results written to $OUTPUT"
ls -lh "$OUTPUT"
