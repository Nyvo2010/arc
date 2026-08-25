# ARC — Adaptive Recurrence Computing

Adaptive (dynamic) recurrence for pretrained MoE Transformers, with a dual
halting-policy track: the calibrated non-neural `ThresholdController`
(Policy-T) and a learned neural halt head (Policy-NN, to be built).

Branch `stage-a-cpt` scope: **continued pre-training of adaptive models only**,
trained in five stages — backbone adaptation at random recurrence, calibrate
non-neural halt policy, train neural halt head, chase-the-leader distillation/
calibration loop, then exploit winner & deploy. Fixed-R variants and benchmark
harnesses live on `main`; this branch keeps the models, inference engine, and
ops docs needed for CPT.

## Variants

| Name | Scale | Recursion | Builder |
|------|-------|-----------|---------|
| base | base | one-pass control | `build_arc_model(source, scale='base')` |
| model_adaptive | model | HALT/CONTINUE per full traversal, max_loops=M | `scale='model'` |
| block_adaptive | block | HALT/CONTINUE per transformer block, max_loops=M | `scale='block'` |
| layer_adaptive | layer | HALT/CONTINUE per layer, max_loops=M | `scale='layer'` |

All share the `ARCAdapter` interface: `model(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult`

## Adapter contract

`ARCAdapter` provides:
- `embed`, `forward_native`, `prepare`
- `forward_layer`, `forward_block`, `forward_model`
- `normalize`, `project_logits`, `final_logits`
- `num_layers`, `num_blocks`
- `lm_head_flops_per_token`, `unit_flops(scale, unit_index, seq_len, batch_size)`

## Halting policies

- **Policy-T (existing):** per-unit halt heads feed hand-calibrated
  `ThresholdController` thresholds. Features: entropy, entropy_delta,
  JS divergence, top-1 stability, hidden cosine change, recurrence count.
  Calibration on the adapted checkpoint is Stage B of the curriculum.
- **Policy-NN (to build):** tiny learned halt head (~10⁴ params,
  PonderNet-style expected loss + compute penalty λ·Δcompute), trained jointly
  with an unfrozen backbone from the Stage-A checkpoint (Stage C).

## Inference

```python
from arc.inference import InferenceEngine

engine = InferenceEngine(source="models/jetmoe-8b", variant="block_adaptive", max_loops=4)
result = engine(input_ids)          # -> RecurrenceResult(logits, final_hidden, state)
metrics = engine.measure(input_ids) # compute_used, executions, unit_loop_counts, timing
```

## Quick start

```bash
pip install -e .
python -c "from arc.inference import InferenceEngine; print('ok')"
```

## Repo layout

```
src/arc/
  models/                 # JetMoe adapter + registry + factory
  recurrence/             # adaptive runtimes, controllers, halt heads, state
  common/config.py
  inference.py            # unified inference engine
scripts/preflight_check.sh
configs/kaggle.yaml       # model + CPT hyperparameters
```

Weights are not shipped. Place JetMoE weights under `models/jetmoe-8b/` or pass a HF path.

Docs: `PLAN.md` — the single source of truth: five-stage curriculum
(Stage A random-recurrence CPT → Stage B1 threshold calibration → Stage B2
neural halt head training → Stage C chase-the-leader distillation/calibration
→ Stage D exploit winner & deploy), publication-grade requirements, measurement
checklist, gates, and ops.
