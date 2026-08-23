# What a Kaggle Run Actually Executes

This document describes **exactly** what `kaggle_run_all.sh` runs, in order,
what is measured, and what the limits and caveats are. Read this before
starting a full run.

## Model

- **JetMoE-8B** (`jetmoe/jetmoe-8b`), MoE with ~8B total / ~2B active params per token.
- Loaded once per process in **8-bit (bitsandbytes)** on GPU via `device_map="auto"`.
- If 8-bit GPU load fails, the run **fails fast** — there is no CPU float32 fallback
  (a 24GB float32 load would OOM or take hours; it was removed deliberately).
- The checkpoint is loaded **once** by `scripts/benchmark_matrix.py` and reused for
  all 13 matrix configs. The notebook must NOT run its own parity cell anymore.

## Stage-by-stage

| # | Stage | What happens | Approx time |
|---|-------|--------------|-------------|
| 0 | Preflight | Python >=3.11 check, model dir + shard validation from `model.safetensors.index.json`, `arc` import check | seconds |
| 1 | Dep check | Verifies torch/transformers/accelerate/safetensors/yaml/psutil are importable. Does NOT reinstall anything (protects Kaggle's CUDA-matched torch). | seconds |
| 1.5 | Parity smoke | Tiny random JetMoE forward-parity test (decomposed vs native HF forward) | <1 min |
| 2 | Real-model parity + matrix | Loads JetMoE-8B once, runs `verify_parity` on real weights (must pass), then runs all 13 configs below | ~20–50 min |
| 3 | Summary | Copies matrix CSV to `summary.csv` | seconds |
| 4 | LM-Eval | Runs the bridge for all 13 configs (fixed variants per R) on trimmed `arc_challenge,mmlu,gsm8k` | see runtime section |

Each stage is capped at `PIPELINE_TIMEOUT` (default 1800s = 30 min). Matrix results
flush to CSV incrementally after every config, so partial results survive timeouts.

## The 13 matrix configs

Measured workload: 32 prompts × seq_len 128, batch size 1, seed 0.

| Config | Scale | Recurrence |
|--------|-------|------------|
| base | base (one pass) | R=1 |
| model_fixed R=2/3/4 | whole-model traversal repeated | 2, 3, 4 |
| block_fixed R=2/3/4 | block-of-4-layers repeated | 2, 3, 4 |
| layer_fixed R=2/3/4 | individual layer repeated | 2, 3, 4 |
| model_adaptive | HALT controller, max_loops=4 | up to 4 |
| block_adaptive | HALT controller, max_loops=4 | up to 4 |
| layer_adaptive | HALT controller, max_loops=4 | up to 4 |

### Metrics recorded per config (`matrix_results.csv/.json`)
- avg / p50 / max latency per prompt
- tokens/sec
- FLOPs: total and per-prompt (**upper-bound MoE estimate — sparsity not modeled**)
- executions, avg recurrence per unit
- avg/max RAM GB and GPU memory GB
- tokens in / out

## LM-Eval benchmarks (real benchmarks)

Run for **all 13 configs** — each fixed variant is evaluated separately per R value
so you can compare which loop counts perform better. Trimmed with `--limit`
(default 200 samples/task, override via `LM_EVAL_LIMIT` env var):

| Eval | arc_challenge (200) | mmlu (200) | gsm8k (200 gens) |
|------|---------------------|------------|-------------------|
| base | ✓ | ✓ | ✓ |
| model_fixed_R2 / _R3 / _R4 | ✓ | ✓ | ✓ |
| block_fixed_R2 / _R3 / _R4 | ✓ | ✓ | ✓ |
| layer_fixed_R2 / _R3 / _R4 | ✓ | ✓ | ✓ |
| model_adaptive / block_adaptive / layer_adaptive | ✓ | ✓ | ✓ |

13 evals × ~600 trimmed requests ≈ **1.5–3 GPU-hours total**. Fits one session.

**Cost side per R:** tokens/sec, FLOPs, RAM, latency for every R value already come
from the matrix stage (`matrix_results.csv`) — combine those rows with these eval
JSONs to get the accuracy-vs-compute tradeoff per loop count.

hellaswag (~40k requests) and arc_easy (~9.5k requests) were **removed for runtime**.
Add them back in `kaggle_run_all.sh` (`TASKS=`) if you accept multi-session runs.

## Runtime reality check (T4 x2 / P100, 8-bit)

- Full pipeline including trimmed LM-Eval: **~2.5–4 hours** in one session. Fits.
- LM-Eval untrimmed (no `--limit`, current 3 tasks × 13 configs): **~30–55 GPU-hours**.
  This does NOT fit one Kaggle session (max 9–12h GPU).
  The runner treats per-config LM-Eval failures/timeouts as warnings and continues,
  so you still get matrix results plus whatever eval output finished.

### Score caveats at limit=200
- A 200-sample slice has wide error bars (±5–10 pts). Good enough to RANK configs
  against each other on the same slice; NOT comparable to published leaderboard numbers.
- gsm8k is generative: 200 greedy generations per config, still the slowest task per sample.

## Kaggle constraints & disclaimers

- Only `/kaggle/working/` persists. All results go there.
- Internet toggle must be ON for pip install and (if used) HF download.
- Never let pip reinstall `torch` mid-session — the image ships a CUDA-matched build.
  The runner intentionally only *checks* dependencies.
- If an install cell changes torch/transformers, restart the session before running.
- GPU quota is ~30 hrs/week free. A single untrimmed LM-Eval pass can consume most of it.
- gsm8k scores from this bridge use greedy generation without few-shot prompt formatting
  guarantees; treat absolute numbers as indicative, not leaderboard-comparable.
- FLOPs numbers are analytic upper bounds (dense assumption), useful for relative
  comparison between configs, not absolute energy/compute claims.
- Adaptive variants' loop counts depend on threshold controllers with default settings;
  they have not been tuned, so adaptive configs may consistently hit max_loops=4.
