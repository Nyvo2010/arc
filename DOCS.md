# ARC — Adaptive Recurrence Computing

## 7 Variants
All share `ARCAdapter` and return `RecurrenceResult(logits, final_hidden, state)`.

- base — JetMoE one-pass
- model_fixed, block_fixed, layer_fixed — fixed R recurrence
- model_adaptive, block_adaptive, layer_adaptive — adaptive HALT per granularity

Builder:
```python
from arc.models.factory import build_arc_model
model, adapter = build_arc_model(source="tiny", scale="model", adaptive=True, max_loops=4)
logits = model(input_ids).logits
```

## Adapter Contract
`ARCAdapter` must implement: embed, prepare, forward_native, forward_layer, forward_block, forward_model, normalize, project_logits, final_logits, num_layers, num_blocks, lm_head_flops_per_token, unit_flops.

## HALT Heads
Per Notion docs:
- model adaptive → HALT after each model turn
- block adaptive → HALT after each block turn
- layer adaptive → HALT after each layer turn

Initial controller is deterministic threshold rules with entropy, JS divergence, top-1 stability, hidden cosine change, recurrence count, compute budget. `HaltHead` modules are attached per scale and wired into `ThresholdController.decide`.

## Running locally / Kaggle
1. Install: `pip install -e .`
2. Load adapter: `from arc.models.registry import create_adapter`
3. Build model with `build_arc_model` or `InferenceEngine`
4. Weights are kept under `models/`; the repo does not ship weights. Place weights or use `source="hf"` with internet access.

## Repo layout
src/arc/models/ — adapters
src/arc/recurrence/ — fixed & adaptive runtimes, controllers, halt heads, state
src/arc/inference.py — unified inference engine

## Files
README.md — overview
DOCS.md — this file

