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
- Kaggle limits (verified 2026-08): T4 x2 / P100 16GB VRAM, GPU sessions up to 9-12h,
  ~30 GPU hrs/week free quota, TPU v3-8 max 9h, interactive idle prompts after ~20min.
  Use Save & Run All (Commit) for unattended runs; weights attach as read-only datasets
  under /kaggle/input/<slug>; only /kaggle/working persists; internet toggle required
  for pip. Never reinstall torch on Kaggle (image ships CUDA-matched build).

## Unified matrix run
`scripts/benchmark_matrix.py` runs the full grid: base(R1) + fixed variants at
R in {2,3,4} + adaptive variants at max_loops=4 = 13 configs, measuring avg/p50/max
time, tokens/s, FLOPs, executions, recurrence-per-unit, RAM/GPU mem, tokens in/out.
Results flush incrementally to matrix_results.csv/.json so partial results survive
session timeouts.

## Changelog
2026-08-22: Pinned requirements for Kaggle reproducibility, added fail-fast timeouts and no-progress guards
2026-08-23: Added benchmark_matrix.py unified 13-config grid; kaggle_run_all.sh no longer reinstalls torch, defaults to /kaggle/input dataset path
