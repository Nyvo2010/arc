# Research Plan: Quality-per-Compute for Recurrence on MoE Transformers

Publication-grade experimental design. v2 — revised after the completed zero-shot pilot
(commit `66adc00`, run `20260823_184315`). Supersedes all earlier scope notes.

## 1. Pilot results (completed — evidence base)

Zero-shot JetMoE-8B, ARC-Challenge n=200, greedy loglikelihood, parity gate max_abs_diff=0.0.

### 1.1 Quality vs compute

| Config | Compute | acc | acc_norm |
|---|---|---|---|
| base | 1× | 0.395 | 0.370 |
| layer_fixed R2 | 2× | **0.410** | 0.385 |
| block_fixed R2 | 2× | 0.385 | **0.395** |
| block_fixed R3 | 3× | 0.360 | 0.380 |
| block_fixed R4 | 4× | 0.350 | 0.335 |
| layer_fixed R3 | 3× | 0.295 | 0.350 |
| model_fixed R2 | 2× | 0.285 | 0.315 |
| model_fixed R3 | 3× | 0.250 | 0.320 |
| model_fixed R4 | 4× | 0.230 | 0.295 |
| block_adaptive | ≤4× (actual 3.90×) | 0.350 | 0.340 |
| model_adaptive | ≤4× (actual 4.00×) | 0.250 | 0.285 |
| layer_adaptive | ≤4× (actual 3.73×) | 0.200 | 0.235 |

n=200 → 95% CI ≈ ±3–7 pts; differences within one CI width are not yet significant.

### 1.2 Pilot findings (these drive the revised plan)

1. **Granularity dominates.** Whole-model recurrence collapses fast (−11 pts acc at R2).
   Block-level degrades gracefully (−1 pt at R2, inside noise). Layer is non-monotonic
   (R2 best-in-run on acc, R4 worst) — unstable, deprioritized.
2. **Shallow local recurrence is quality-neutral but not quality-positive.** block_fixed_R2
   matches base at 2× compute. Zero-shot, no configuration beats base on quality-per-compute.
3. **The adaptive controller saturates.** Measured avg recurrence/unit: model 4.00,
   block 3.90, layer 3.73 out of max 4. Root cause is structural, not tuning luck:
   - Iteration 1 has no previous logits → `top1_stability = 0` → the rule
     `top1_stability < 0.95` forces CONTINUE unconditionally (≥2 executions always).
   - The three conditions are OR-combined (`controller.py`); ANY violated condition
     forces another loop, so near-threshold signals never accumulate into a halt.
4. **Adaptive overhead is real.** layer_adaptive: 0.97 s/prompt vs layer_fixed_R4's
   0.72 s (+35% wall-clock) for ~6% fewer FLOPs. Zero-shot adaptive is strictly dominated.

### 1.3 Conclusion

Untuned recurrence damages quality; untuned halting degenerates to max loops. Both point
to the same next step: **adapt the backbone to a shallow block-recurrent architecture, then
make early exit earn its keep via calibrated thresholds** (learned halt head demoted —
see §3.0).

## 2. Research question (revised)

After equal-budget adaptation, does a block-recurrent JetMoE-8B (R=2) match base quality
while a calibrated-halting variant recovers part of the extra compute — i.e., can adapted
recurrence reach base-level quality at <2× compute, with early exit pushing toward ≤1.5×?

Secondary: does the same hold at matched wall-clock (halting overhead counted)?

## 3. Experimental phases

### Phase 0 — Threshold calibration on the zero-shot backbone (cheap, days)

Calibrate, don't train. No new parameters; reuse `ThresholdController`.

* Fix controller decision logic first (prerequisite):
  - [ ] Make iteration 1 an explicit structural rule (always ≥2 executions) instead of a
    threshold accident (`top1_stability=0`).
  - [ ] Support configurable combine mode: OR (current), AND, and per-signal-only modes.
  - [ ] Expose all four thresholds via config so a sweep needs no code changes.
* Calibration procedure:
  - Validation set: ARC-Easy validation, n≈500 (never touched during reporting).
  - Grid: `top1_stability_threshold ∈ {0.5,0.6,…,0.9}` × `js_threshold ∈
    {0.001,0.005,0.01,0.05}` × `hidden_change_threshold ∈ {0.001,0.005,0.01}` ×
    combine ∈ {OR, AND}; `entropy_delta` rule ablated on/off.
  - Output: Pareto curve of (acc_norm, avg recurrence/unit). Select operating point by
    pre-declared rule: highest quality within recurrence ≤ 2.0 average.
* Deliverable: evidence whether simple stability signals can halt at all on an unadapted
  backbone. Expected outcome (from saturation data): partial savings, quality capped.
  This is a valid negative result either way and validates the harness for Phase 2.

A learned halt head remains a contingency ONLY if Phase 2 re-calibration fails
(see §3 Phase 2 gate).

### Phase 1 — Backbone adaptation via QLoRA (the main bet)

Adapt the backbone WITH block-R=2 recurrence active, so weights learn to exploit re-reads
instead of being broken by them (looped-transformer recipe). Operational details
(checkpointing, resume, session design, data streaming) follow
`CONTINUED_PRETRAIN_PLAN.md`; this section overrides it where the pilot evidence demands.

* Architecture under adaptation: `block_fixed R=2` (primary), `base` (control).
  **Model-level recurrence is dropped** — pilot finding #1 (−11 pts acc at R2, worst
  granularity everywhere). This supersedes CONTINUED_PRETRAIN_PLAN.md's "start with
  model_fixed". Layer-level stays exploratory-only, last in queue.
* Method: QLoRA (4-bit nf4 base, bf16 LoRA adapters r=16 α=32 dropout 0.05) on attention
  projections (q,k,v,o); MoE expert projections only if the shared-adapter spike passes.
  Full fine-tuning is infeasible on T4/P100 16GB for an 8B — adapters are mandatory,
  not optional.
* Two-tier data budget (resolves CONTINUED_PRETRAIN_PLAN's 0.5–1M vs larger corpora):
  - Tier A — feasibility run: ~1M tokens CPT mix (their warm-up protocol). Cost ≈ 1–3
    GPU-h. Success gate: recurrent loss decreases and ARC-Easy-val n=200 does not
    degrade vs step 0. If Tier A shows nothing, stop and rethink objective — do not
    burn the weekly quota.
  - Tier B — real run: 20–50M tokens (wikitext-103 + C4 slice), 1 epoch minimum.
    Only after Tier A passes.
* Equal-budget control: identical recipe on `base` (same tokens, steps, LR, adapter
  placement, no recurrence), same tier gates. This control is NON-NEGOTIABLE — without
  it, "adaptation helped recurrence" is unfalsifiable.
* Hyperparameters: LR 1e-4 (LoRA-typical; supersedes the 5e-6 full-FT value in
  CONTINUED_PRETRAIN_PLAN.md), cosine decay, warmup 3%, seq_len 1024 (2048 only if VRAM
  allows with checkpointing), micro-batch 1–2, grad-accum to effective 16.
  CONTINUED_PRETRAIN_PLAN's "batch 64 × accum 256" effective batch is not reachable on
  one T4 with an 8B — treat as aspirational for multi-GPU/A100 venues.
* Validation during training: wikitext ppl + ARC-Easy-val n=200 every N steps, tracked
  separately for recurrent and control runs; CSV logging + auto-resume exactly as
  specified in CONTINUED_PRETRAIN_PLAN.md (adopt unchanged).
* Compute estimate (T4 16GB): Tier A ≈ 1–3 GPU-h/run ×2 runs; Tier B ≈ 15–40 GPU-h/run
  ×2 runs → fits Kaggle's ~30 h/week quota over 1–2 weeks with session resume.
  Colab Pro / rented A100 compresses Tier B to hours and unlocks seq_len 2048.

### Phase 2 — Re-calibrate halting on the ADAPTED backbone

* Repeat Phase 0's calibration grid against the Phase 1 treatment checkpoint.
  Hypothesis: an adapted backbone converges faster between loops → thresholds become
  separable → early exit saves real compute without quality loss.
* Gate for considering a learned halt head: if the calibrated-threshold Pareto front is
  dominated by fixed R=2 (i.e., halting never beats just running R=2), THEN a learned
  head (PonderNet-style expected loss + compute penalty) becomes justified as Phase 3.
  Otherwise it stays dead — simpler machinery wins ties.

### Phase 3 — Full evaluation (pre-registered)

* Configs evaluated (reduced from 13 — pilot already answered the zero-shot question):
  1. base (original)
  2. base-adapted (control)
  3. block_fixed R2-adapted (treatment)
  4. block_adaptive-adapted + calibrated thresholds (treatment)
  5. Optional: layer_fixed R2-adapted
* Benchmarks (all loglikelihood unless noted): ARC-Challenge (1172, full),
  HellaSwag (n=1000 capped), PIQA (1838 full), WinoGrande (n=1000 capped),
  BoolQ (n=2000 full), MMLU (100/subtask cap — requires §5 gap fix).
  GSM8K/MATH500 only if a generative decode path through the wrapper is validated;
  otherwise explicitly excluded and stated.
* Seeds: ≥3 evaluation-order/data-order seeds per config; identical prompt subsets
  across configs. Paired bootstrap 10k resamples + McNemar where applicable,
  Holm-Bonferroni over config pairs. Effect sizes with CIs.
* Reporting: Pareto plot quality vs measured active GFLOPs AND vs wall-clock (both);
  Δquality/Δcompute ratios; halting-position histograms; % hitting max_loops.

## 4. What we measure (unchanged in substance)

1. **Active FLOPs measured** — instrumented forwards + router top-2 counts
   (gap below). 2. **Dense-bound FLOPs** analytic, labeled. 3. **Wall-clock**
   p50/p95/p99 ms/prompt, tokens/s, named hardware. Plus peak RAM/VRAM, tokens in/out,
   effective size (active params × executed layer-passes), avg recurrence/unit,
   halting distributions.

Quality: acc + acc_norm, per-sample logs persisted (`log_samples=True`),
mean ± 95% CI over seeds.

## 5. Infrastructure gaps (ordered by blocking priority)

- [ ] **Per-task sample limits in `lm_eval_bridge.py`** — BLOCKING for MMLU. Current
  `--limit 200` applies per subtask → MMLU = 57 × 200 requests, guaranteed stage timeout
  (observed in both Kaggle runs; every mmlu result lost). Patch: pass per-task limit dict.
- [ ] Controller config surface + combine-mode + explicit iteration-1 rule (Phase 0 prereq).
- [ ] Threshold-calibration sweep script (reuses matrix harness; writes Pareto CSV).
- [ ] Active-FLOPs hook (router top-2 counting) — needed before Phase 3 claims.
- [ ] QLoRA training script + dataset prep (Phase 1 prereq; largest new component).
- [ ] Resume-safe multi-seed eval loop; per-sample JSON persistence.
- [ ] Paired-stats + Pareto figure script.
- [ ] Pre-registration file frozen before Phase 3 launch (configs, seeds, splits, metrics,
  selection rules — including the Phase-2 operating-point rule).

## 6. Open decisions (needed before Phase 1 launch)

1. **Dataset**: CPT mix (wikitext/C4) vs SFT mix — Tier A can run on either cheaply;
   decide before Tier B based on which moves ARC-Easy-val more.
2. **Compute venue**: Kaggle free (T4/P100, 30 h/wk, session resume per
   CONTINUED_PRETRAIN_PLAN.md) vs Colab Pro vs rented A100.
3. **Adapter scope**: attention-only vs attention+MoE-experts (feasibility spike needed).
4. **Exploratory layer arm**: include or cut.

## 7. Non-goals / honesty constraints

* No claims from the pilot beyond its n=200 CIs — it motivates, it does not confirm.
* Zero-shot adaptive results will be reported (they're interesting negative results),
  clearly separated from post-adaptation results.
* If adapted recurrence fails to match the base control, the paper's contribution flips
  to the negative/mechanistic result; that is still a publishable outcome and the protocol
  above supports it without modification.
