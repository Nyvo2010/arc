# What a Kaggle Run Actually Executes

This document describes **exactly** what `kaggle_run_all.sh` runs, in order,
what is measured, and what the limits and caveats are. Read this before
starting a full run.

> **Status: pilot-study pipeline.** This run validates the harness and produces
> directional signals only. See "Scope of this run" below for what the next
> full evaluation must add before any research conclusions are drawn.

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

| Eval | arc_challenge (200) | mmlu (200) |
|------|---------------------|------------|
| base | ✓ | ✓ |
| model_fixed_R2 / _R3 / _R4 | ✓ | ✓ |
| block_fixed_R2 / _R3 / _R4 | ✓ | ✓ |
| layer_fixed_R2 / _R3 / _R4 | ✓ | ✓ |
| model_adaptive / block_adaptive / layer_adaptive | ✓ | ✓ |

12 evals × ~400 trimmed requests ≈ **40–90 min total**. Fits one session easily.

**gsm8k is opt-in, off by default.** It is generative through the recurrence
wrapper (no KV cache), so 200 greedy generations cannot finish inside the
30-min stage cap — the first full run timed out on it and produced nothing.
Enable with `LM_EVAL_TASKS=arc_challenge,mmlu,gsm8k` if you accept that; scores
now save incrementally per task, so completed arc_challenge/mmlu results
survive a gsm8k timeout within an eval.

**Results are saved per task**: each eval writes its JSON after every finished
task (`--output`), so a timeout loses at most the in-progress task, not the
whole eval.

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
- **mmlu does not complete at limit=200**: `--limit` applies PER SUBTASK and mmlu is
  57 subtasks → ~45k requests, which always exceeds the 30-min stage cap. Each eval
  therefore yields arc_challenge only; mmlu needs a per-task limit (~25/subtask) or
  subtask selection to fit.

## Scope of this run: PILOT STUDY — not publication-grade

This pipeline validates that the harness works end-to-end (parity gates pass,
metrics are collected, benchmarks score) and gives **directional signals only**.
It is general experimentation, not evidence for research claims. Do not cite
these numbers as findings.

Known limitations that block academic claims:
1. **Single benchmark** (arc_challenge, 200 samples). One dataset cannot support
   a quality-per-FLOP claim.
2. **Underpowered statistics.** ±3–7pt stderr at n=200; only gaps >~7pts are
   distinguishable (e.g., base vs model_fixed R=2).
3. **FLOPs are a dense analytic upper bound**, not measurement. JetMoE routes
   top-2-of-8 experts per token, so active compute is roughly 1/4 of reported
   values and is input-dependent (routing not yet measured).

## Required for the NEXT full evaluation (research-grade)

The full publication-grade design lives in **RESEARCH_PLAN.md**. Summary of
what must change before drawing conclusions from all models/configs:

1. **Full datasets or large samples** — arc_challenge full test set (1172 docs)
   at minimum; report mean ± CI over >=3 seeds per config with paired
   significance tests between configs.
2. **Multiple benchmarks** — at least 3 diverse tasks (e.g., arc_challenge,
   hellaswag subset, mmlu subset with per-task limits, plus a commonsense task
   like PIQA/WinoGrande).
3. **MoE-aware compute accounting** — count actual expert activations per forward
   (router selections are observable) and report BOTH dense-bound and
   active-FLOPs; wall-clock time and memory stay as measured secondary metrics.
4. **Per-task sample limits** in the bridge (`--limit` currently applies per
   subtask, which silently explodes grouped tasks like mmlu) so every task fits
   its stage budget deterministically.
5. **Tuned or at least characterized adaptive controllers** — record actual
   halting behavior distribution; untuned thresholds may make adaptive variants
   degenerate to max_loops every time, which must be visible in the results.

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
