# Research Plan: Quality-per-Compute for Recurrence on MoE Transformers

Publication-grade experimental design. This supersedes the pilot-study scope in
TEST_RUN_DESCRIPTION.md once implemented.

## Research question

Does input-independent recurrence (model / block / layer granularity, fixed R and
HALT-controlled) improve task quality **per unit of compute** on a pretrained MoE
LLM, and how does the answer depend on the recurrence count?

## What we measure (exact quantities)

### A. Quality (dependent variable 1)

| Quantity | How |
|---|---|
| Task accuracy (acc, acc_norm where defined) | lm-eval-harness, loglikelihood tasks preferred (deterministic) |
| Per-sample predictions | `log_samples=True` — raw outputs stored for reanalysis |
| Uncertainty | mean ± 95% CI over >= 3 seeds; paired tests between configs on identical samples |

**Benchmarks (>= 4, diverse capabilities):**

| Benchmark | Capability | Samples | Why |
|---|---|---|---|
| ARC-Challenge (test) | grade-school science reasoning | 1172 (full) | standard, loglikelihood MC |
| HellaSwag (validation) | commonsense sentence completion | >=1000 | standard, loglikelihood MC |
| PIQA (validation) | physical commonsense | >=1000 | complements HellaSwag |
| WinoGrande (validation) | coreference / world knowledge | >=1000 | different failure mode |
| MMLU (selected subjects) | broad knowledge | 500 total (per-task limit) | breadth; per-task limits mandatory |

Rule: every task must fit its stage budget deterministically (per-task sample
limits), and every reported number carries n and CI.

### B. Compute (dependent variable 2)

Report ALL THREE; never a single cost metric:

1. **Active FLOPs (primary)** — measured, not modeled. Instrument forwards to
   count: executed layer-passes (from recurrence state), router top-k expert
   selections per token (observable from the gating module), attention score
   FLOPs from sequence lengths. Formula documented in-repo; validated against
   wall-clock sanity checks.
2. **Dense-bound FLOPs** (current analytic estimate) — kept for continuity,
   clearly labeled upper bound.
3. **Wall-clock** — ms/prompt, ms/output-token, tokens/s, on named hardware
   (GPU model, driver, torch/cuda versions). Memory peak alongside.

### C. Size accounting

For JetMoE-8B (8 experts, top-k=2): report total params (8B) AND active params
per token (~2B) separately. Define **effective size** = sum of active params
over all executed layer-passes per prompt — this is the honest "how big was the
model during this inference" number that recurrence multiplies.

## Experimental protocol

1. **Configs**: base; {model,block,layer}_fixed x R in {2,3,4}; {model,block,layer}
   adaptive (max_loops=4). Same 13 as now.
2. **Seeds**: >= 3 seeds per config per benchmark; identical prompt subsets across
   configs (paired design).
3. **Controls**: same checkpoint, same 8-bit quantization, greedy/loglikelihood
   decoding (no sampling), parity gate before measurement, git commit +
   dependency lockfile logged per run (already automated).
4. **Statistics**: paired bootstrap (10k resamples) or McNemar's test between
   each config and base per benchmark; Holm-Bonferroni correction across the
   12 comparisons; report effect size (accuracy delta) with CI, not p-values alone.
5. **Quality-per-compute reporting**:
   - Pareto plot: accuracy vs active-GFLOPs-per-prompt (all 13 configs, all benchmarks)
   - Delta-quality / delta-compute ratio vs base, with CIs propagated
   - Quality at matched compute budget (interpolate configs to equal-FLOP points)
6. **Adaptive analysis**: distribution of actual halting positions per benchmark
   (must not silently saturate at max_loops); controller thresholds ablated.

## Known gaps to implement before this run

- [ ] Per-task sample limits in bridge (`--limit` currently applies per subtask)
- [ ] Router-activation counting hook in JetMoeAdapter
- [ ] Multi-seed loop in runner (seeds as outer dimension, resume-safe JSONs)
- [ ] Per-sample logging persisted (`log_samples`) for offline statistics
- [ ] Paired-stats script producing tables + Pareto figures from result JSONs
- [ ] Optional: nvidia-smi power sampling for energy-per-prompt

## Runtime estimate (T4-class GPU, 8-bit)

Matrix cost table: unchanged (~10 min).
Eval load: 13 configs x 5 benchmarks x 3 seeds ~= 195 eval-units; with full
ARC (1172x4 reqs) and capped others (~4k requests/unit avg): ~60-90 GPU-hours.
Split across sessions via resume logic, or cut to 2 seeds / drop WinoGrande to
fit ~30h weekly quota in two weeks.
