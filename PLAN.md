# ARC Plan — Training Adaptive Recurrence Models

Single source of truth for the `stage-a-cpt` branch. Supersedes and replaces
`RESEARCH_PLAN.md` + `CONTINUED_PRETRAIN_PLAN.md`.

## 1. What we are doing

We adapt JetMoE-8B into **adaptive (dynamic) recurrence models** and train them
in three stages, in this order:

1. **Stage A — adapt the backbone at random recurrence.** Train the recurrent
   backbone with depth-randomized loops (random depth ∈ {1..4} per forward,
   mass biased toward 1–2), NO halting machinery. The weights learn to work
   under varying recursion and to produce clean shallow-pass representations.
   This produces THE shared checkpoint for everything downstream, plus an
   equal-budget `base` control trained identically minus recurrence.
2. **Stage B — calibrate the non-neural halt policy.** Freeze the adapted
   backbone; calibrate the existing `ThresholdController` thresholds on it
   (Policy-T). This is both the fallback deployment point and the bar Stage C
   must beat.
3. **Stage C — joint training with a neural halt head.** Unfreeze the backbone;
   build and train a tiny learned halt head (~10⁴ params) jointly with it
   (Policy-NN): PonderNet-style expected next-token loss with a compute penalty
   `Δquality − λ·Δcompute`, λ ∈ {0.5, 1.0} — quality counts equally or more
   than compute savings. Backbone LR dropped ~10× vs Stage A (adapter refresh).
   All-at-once: head + transformer trained together, initialized from Stage-A
   weights, judged against Stage-B's operating point.

Every stage has a cheap kill gate; a Stage-C collapse still leaves Stage B as a
working deployment point. Deployment halting is recalibrated post-training
without exception — training sets capability, post-hoc sweeps set the operating
point (target average recursion ≈ 1.3–1.5× base).

## 2. Evidence so far (zero-shot pilot — motivates, does not confirm)

- Zero-shot recurrence is quality-neutral at best (2 of 12 configs beat base,
  inside noise); harmful at depth.
- Root cause of halting saturation is structural: iteration 1 has no previous
  logits → top-1 stability = 0 → threshold rules force CONTINUE unconditionally;
  OR-combined conditions mean any violated signal forces another loop. The
  curriculum removes exactly this pathology.
- Latency scales linearly with recursion; adaptive variants cost more wall-clock
  than equivalent fixed-R while executing fewer FLOPs. Recurrence costs compute,
  not memory (~2.48 GB flat).

## 3. Publication-grade requirements

**Every part of this research must be publication grade.** Concretely:

- Pre-declared configs, seeds, splits, selection rules, and kill gates — frozen
  before final evaluation; no test peeking, no tuning-until-it-looks-good.
- Claims require multiple benchmarks, ≥3 seeds, paired significance tests
  (paired bootstrap + McNemar, Holm-Bonferroni over pre-declared primary pairs),
  effect sizes with confidence intervals.
- All compute comparisons use **measured** active FLOPs and wall-clock, never
  analytic or nominal numbers.
- External open models of comparable active-parameter count evaluated through
  the identical harness for the quality-per-compute claim.
- Negative results are reported in full. The loser of Policy-T vs Policy-NN is
  an ablation, never silently dropped. If adaptation fails to match control,
  the contribution flips to the negative/mechanistic result — protocol unchanged.

## 4. Final evaluation — benchmarks MUST be run

At the end, run the full benchmark suite through one harness, identical subsets
across all configs:

ARC-Challenge (full, primary), ARC-Easy, HellaSwag, PIQA, WinoGrande, BoolQ,
SciQ, MMLU, wikitext-103 perplexity; GSM8K generative reported separately if
the decode path validates. External references: JetMoE-8B (substrate),
TinyLlama-1.1B, Qwen2.5-1.5B/Gemma-2B, Pythia-1.4B/2.8B.

## 5. Measure EVERYTHING

Instrumentation is not optional. Every training and eval run logs, at minimum:

- **Throughput/compute:** tokens/s, wall-clock latency per prompt, measured
  active FLOPs (incl. router top-k counting), total FLOPs consumed, GPU-h used,
  VRAM and system RAM profiles.
- **Halting behavior:** number of recursions chosen by the halt head per unit
  (full histogram, not just means), average and max executions, % hitting
  max_loops, halt-position distributions over training time.
- **Quality:** benchmark scores with CIs, validation loss/perplexity curves,
  per-loop-position losses every N steps (does pass-2 loss fall below pass-1? —
  the mechanistic figure), ARC-Easy-val probes at checkpoints.
- **Provenance:** config dumps, seed, checkpoint hashes, data mix, tokens
  processed, cost reports (GPU-h, tokens/hour).
- Collapse monitors: abort Stage C if p(halt@1) > 0.95 or < 0.05 without
  val-loss justification, or val ppl degrades >2% vs Stage-A init.

If a metric would be embarrassing, log it anyway. Everything goes to CSV/JSON,
incrementally flushed so partial results survive session timeouts.

## 6. Decision gates

| Gate | Proceed if | Else |
|---|---|---|
| G1 | Controller repairs pass regression tests | fix before any sweep |
| G2 | Tier A: recurrent val loss < its own step-0 AND ARC-Easy-val within 2 pts of control | stop, rethink objective |
| G3 | Tier B treatment ≥ base control on ARC-Easy-val | negative/mechanistic paper — still publishable |
| G4 | Stage C beats Stage-B Pareto front by ≥2 pts acc_norm at equal avg recursion, val ppl within 2% | deploy Policy-T |
| G5 | Pre-registration frozen before final eval | no final eval |

## 7. Compute budget & ops (Kaggle)

~30 free GPU-h/week, T4 x2 / P100 16 GB, 9–12 h sessions, only `/kaggle/working`
persists. Never reinstall torch. QLoRA (4-bit nf4 base, bf16 LoRA r=16 α=32 on
q/k/v/o projections) — full FT is infeasible on 16 GB for 8B.

Token budget: **Tier A ≈ 1M tokens** (feasibility gate, 1–3 GPU-h ×2 runs);
**Tier B = 20–50M tokens** (wikitext-103 + C4 slice, ≥1 epoch, 15–40 GPU-h ×2).
If Tier-B losses have not plateaued by end of budget, extend rather than
declare convergence early. Hyperparameters: LR 1e-4 cosine, warmup 3%,
seq_len 1024 (2048 if VRAM allows), effective batch 16 via grad accumulation.

Ops rules: preflight checks before each session; load weights once; checkpoint
every 500 steps to `/kaggle/working/checkpoints/`; auto-resume from latest
checkpoint; CSV metric logging; stream datasets in small batches and
pre-tokenize into reusable shards; monitor first 5 min then auto-cancel on stalls.

Rough schedule: W1 infra + Tier A → W2–3 Tier B ×2 → W4 Stages B+C → W5+
full evaluation (all configs × suite × 3 seeds + external refs).
