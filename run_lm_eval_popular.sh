#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-/kaggle/working/models/jetmoe-8b}
OUTPUT_DIR=${2:-/kaggle/working/arc_results}
DEVICE=${KAGGLE_DEVICE:-cuda}

mkdir -p "$OUTPUT_DIR"

pip install -q -e .
pip install -q "lm-eval[hf]>=0.4.12"

# Popular free loglikelihood tasks
TASKS="mmlu,hellaswag,piqa,winogrande,truthfulqa,arc_challenge"

for VARIANT in base model_fixed block_fixed layer_fixed model_adaptive block_adaptive layer_adaptive; do
  echo "Evaluating $VARIANT ..."
  python -m arc.benchmarks.lm_eval_bridge \
    --source "$MODEL_PATH" \
    --variant "$VARIANT" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --max_loops 4 \
    --seed 0 \
    > "$OUTPUT_DIR/${VARIANT}.json" 2>&1
done

echo "Done. Results in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
