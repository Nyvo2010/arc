#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
PROMPTS=${2:-/kaggle/working/prompts.jsonl}
OUTPUT=${3:-results_full.csv}
DEVICE=${KAGGLE_DEVICE:-cuda}

pip install -q -e .

python benchmark_full.py \
  --source "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --device "$DEVICE" \
  --variants base model_fixed block_fixed layer_fixed model_adaptive block_adaptive layer_adaptive \
  --seed 0 \
  --max_loops 4 \
  --recurrence 2 \
  --output "$OUTPUT"

echo "Done. Summary: $OUTPUT"
