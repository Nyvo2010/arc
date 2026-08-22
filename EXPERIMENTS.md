# ARC Research Constants

## Reproducibility defaults
- **seed**: 0
- **python**: >=3.11
- **torch**: 2.4.0
- **transformers**: 4.44.0
- **accelerate**: 0.34.0

## Model variants
Seven variants defined in `src/arc/models/registry.MODEL_VARIANTS`:
- base
- model_fixed
- block_fixed
- layer_fixed
- model_adaptive
- block_adaptive
- layer_adaptive

## Recurrence constants
- **fixed recurrence values**: 2, 3, 4
- **adaptive max_loops**: 4
- **parity gate**: `JetMoeAdapter.verify_parity` must pass before experiments

## Benchmarks
Free community benchmarks via LM-Eval loglikelihood-only:
- hellaswag
- arc_easy
- arc_challenge
- gsm8k
- mmlu zero-shot

## Metrics recorded
- avg_time_s, tokens_per_s
- total/avg FLOPs (upper bound MoE estimate)
- avg_executions, avg/max RAM GB
- total tokens in/out
- avg recurrence per unit

## Experiment notes
- FLOP estimator is upper bound, not sparsity-aware
- HALT heads use entropy, entropy_delta, JS divergence, top-1 stability, hidden cosine change, recurrence count
- Device: prefer CUDA, fallback CPU
- Kaggle limits: CPU/GPU batch max 12h, TPU max 9h, interactive idle ~20min

## Changelog
2026-08-22: Pinned requirements for Kaggle reproducibility, added fail-fast timeouts and no-progress guards
