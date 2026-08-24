# Research Plan: Quality-per-Compute for Recurrence on MoE Transformers

Publication-grade experimental design. **v5 — revised 2026-08-24** after the completed
zero-shot pilot (commit `66adc00`, quality run `20260823_184315`), the throughput/
halting matrix (`RESULTS.CSV`, 32 prompts × seq 128, T4, 8-bit), and the v4 design
review. Supersedes v4 and all earlier scope notes.

> **Branch scope note (`stage-a-cpt`):** work on this branch narrows focus to
> continued pre-training of the **adaptive recurrence models at all scales**
> (model/block/layer) plus the base control, and the dual halting-policy track
> (calibrated ThresholdController + learned neural halt head). This partially
> supersedes locked decision #5's model-level drop; the fixed-R ablation arms and
> benchmark harnesses remain on `main` for the Phase 3 evaluation protocol.

## 0. Decisions locked by this revision

1. **Three-stage training curriculum — halting is never trained from step 0.**
   - **Stage A:** adapt the recurrent backbone alone, depth-randomized block
     recurrence (random loops ∈ {1..4}, mass biased toward 1–2), no halting
     machinery. Produces THE shared checkpoint for everything downstream.
   - **Stage B:** FREEZE the backbone; calibrate + fine-tune the NON-neural halt
     policy (`ThresholdController` thresholds from their current values).
   - **Stage C:** UNFREEZE — joint training of learned halt head + transformer
     backbone simultaneously, initialized from Stage-A weights, with Stage-B's
     operating point as the reference to beat.
   Rationale: every stage has a cheap kill gate; a Stage-C collapse still leaves a
   working deployment point (Stage B); one backbone checkpoint serves all configs.
   The wrapper exposes `recurrence_mode ∈ {fixed, adaptive}` × `fixed_R ∈ {1..4}`,
   so any checkpoint also evaluates at any fixed recurrence from identical weights.
2. **Dual halting policies, compared on the same Stage-A checkpoint:**
   Policy-T = the existing non-neural `ThresholdController`, thresholds fine-tuned
   from their current values (Stage B); Policy-NN = a tiny learned halt head
   (~10⁴ full-precision params, hidden → {1..4}, PonderNet-style expected loss),
   trained JOINTLY with the unfrozen backbone (Stage C). **Reward shape
   (Policy-NN): quality first** — objective = Δquality − λ·Δcompute with λ ≤ 1,
   swept {0.5, 1.0}: compute savings count equally or LESS than quality gains,
   never more. Head trains on a held-out CPT slice only; benchmark validation
   sets stay reserved for policy selection.
3. **max_loops = 4 permanently** (architectural ceiling, not tuned). Halting is
   expected well below it; max_loops exists as the hard safety bound only.
4. **1-pass exit allowed.** Iteration-1 halting must be decidable from step 1 —
   no structural ≥2 rule survives. The threshold-controller pathology
   (top1_stability=0 forcing CONTINUE) is exactly what the curriculum removes.
5. **Stage-A arms:** `block_rand` (treatment backbone), `base` control
   (equal-budget, no recurrence), and `block_fixed R2` ablation (gated on remaining
   quota). All recurrent quality claims are made against the control at equal
   token/step budget. Halting policies are trained in Stages B/C, not as separate
   backbone runs.
6. **Deployment halting is recalibrated after training without exception** —
   post-hoc sweeps on the trained checkpoint set the *deployment* operating point
   (target avg recurrence ≈ 1.3–1.5); training sets the *capability*.
7. **Claims require the full protocol:** multiple benchmarks, ≥3 seeds, paired
   significance tests, measured (not analytic) active FLOPs, and external reference
   models of comparable active-parameter count.

## 1. Research question

**Primary:** after equal-budget adaptation, does the block-recurrent JetMoE-8B
with a learned halting policy beat the equal-budget base control, and does its
halting hold most of that quality at **≤1.5× base FLOPs** (vs 2× for fixed R=2)?

**Secondary:**
- Does the advantage hold at matched *wall-clock* once controller overhead is counted?
- Does adaptation close the zero-shot degradation mechanistically (do loop-2 losses
  fall below loop-1 losses after training)?

**Paper headline format:** Δquality per Δ*measured* FLOPs vs (a) own base,
(b) equal-budget adapted control, (c) external open models at 1–3B active params.

## 2. Evidence base (pilot — completed, motivates only)

### 2.1 Zero-shot quality (ARC-Challenge n=200, greedy loglikelihood, seed 0)

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
| block_adaptive | 3.90× actual | 0.350 | 0.340 |
| model_adaptive | 4.00× actual | 0.250 | 0.285 |
| layer_adaptive | 3.73× actual | 0.200 | 0.235 |

Of 12 modified configs, exactly **2 beat base on any metric** (bold) — both inside
the ±3–7 pt CI. Honest framing: zero-shot recurrence is *quality-neutral at best*
at shallow depth, harmful at depth. No zero-shot configuration wins on
quality-per-compute.

### 2.2 Throughput & halting (`RESULTS.CSV`, 32 prompts × seq 128)

| Config | avg executions | recurrence/unit | tokens/s | s/prompt |
|---|---|---|---|---|
| base | 1.0 | — | 652 | 0.196 |
| model_fixed R2/R3/R4 | 2/3/4 | 2.0/3.0/4.0 | 326/222/169 | 0.39/0.58/0.76 |
| block_fixed R2/R3/R4 | 12/18/24 block-passes | 2.0/3.0/4.0 | 355/244/184 | 0.36/0.52/0.70 |
| layer_fixed R2/R3/R4 | 48/72/96 layer-passes | 2.0/3.0/4.0 | 359/238/178 | 0.36/0.54/0.72 |
| model_adaptive | 4.0 | 4.00 | 165 | 0.78 |
| block_adaptive | 23.4 | 3.90 | 165 | 0.78 |
| layer_adaptive | 89.5 | 3.73 | 132 | 0.97 |

Findings: latency scales linearly with R; adaptive variants cost *more* wall-clock
than the equivalent fixed-R config (+35% for layer_adaptive vs layer_fixed R4) while
executing fewer FLOPs; RAM is flat (~2.48 GB avg) — recurrence costs compute, not
memory.

### 2.3 Root cause of halting saturation (structural, not tuning luck)

- Iteration 1 has no previous logits → `top1_stability = 0` → the
  `top1_stability < 0.95` rule forces CONTINUE unconditionally (≥2 executions always).
- The three conditions are OR-combined (`controller.py`); ANY violated condition
  forces another loop, so near-threshold signals never accumulate into a halt.

### 2.4 What the pilot licenses

Nothing about post-adaptation quality. It licenses: (a) the granularity ranking
hypothesis block > layer > model; (b) the controller pathology above; (c) harness
validity (parity gate max_abs_diff = 0.0).

## 3. Experimental phases

### Phase 0 — Controller repair + threshold baseline on the zero-shot backbone (days, cheap)

Still calibrate, don't train — this phase now serves as the *hand-tuned baseline*
that Stage B must beat, plus harness validation. Reuses `ThresholdController`.

* Controller repairs (prerequisite, each covered by a pytest regression test):
  - [ ] Iteration-1 halting DECIDABLE (thresholds evaluated from step 1) — the
    structural ≥2 rule is removed; 1-pass exit must be reachable.
  - [ ] Configurable combine mode: OR (current), AND, per-signal-only.
  - [ ] All four thresholds exposed via config (no code edits during sweeps).
  - [ ] `max_loops` configurable per-run, hard-capped at 4 (locked decision #3).
* Calibration grid: `top1_stability ∈ {0.5,…,0.9}` × `js ∈ {0.001,0.005,0.01,0.05}`
  × `hidden_change ∈ {0.001,0.005,0.01}` × combine ∈ {OR, AND}; entropy_delta
  ablated on/off; max_loops ∈ {2,4}.
* Validation set: ARC-Easy *validation* n≈500 (never touched during reporting).
* Output: Pareto curve (acc_norm, avg recurrence/unit). Operating-point rule
  (pre-declared): highest acc_norm subject to avg recurrence ≤ 1.5.
* Deliverable: the hand-calibrated baseline + negative-result evidence of why fixed
  thresholds are insufficient on an unadapted backbone. This is what justifies the
  learned halt head in the paper's ablation story.

### Phase 1 = Stage A — Backbone adaptation via QLoRA (critical path)

Adapt the backbone with depth-randomized block recurrence (random loops ∈ {1..4},
mass biased toward 1–2), NO halting machinery — weights learn both to exploit
re-reads AND to produce clean 1-pass representations. Operational details
(checkpointing, resume, session design, streaming) follow
`CONTINUED_PRETRAIN_PLAN.md`; this section overrides it wherever the pilot demands.

* Stage A arms (frozen):
  - **Treatment (primary):** `block_rand` — one checkpoint serves ALL inference
    modes via the wrapper's `recurrence_mode`/`fixed_R` parameters: fixed R1/R2/R4,
    Policy-T, Policy-NN.
  - **Control:** `base`, identical recipe minus recurrence. NON-NEGOTIABLE.
  - **Ablation arm (conditional):** `block_fixed R=2`, only if Tier A passes AND
    ≥20 GPU-h remain in quota — isolates what depth-randomization costs at R2.
* Dropped: model-level recurrence (worst everywhere). Layer-level exploratory-only.
* Method: QLoRA (4-bit nf4 base, bf16 LoRA r=16 α=32 dropout 0.05) on attention
  projections (q,k,v,o); MoE expert adapters only if the shared-adapter spike passes.
  Full FT is infeasible on T4/P100 16GB for 8B — adapters are mandatory.
* Two-tier token budget:
  - **Tier A — feasibility:** ~1M tokens CPT mix, ≈1–3 GPU-h ×2 runs.
    Gate (pass/fail, pre-declared): recurrent val loss < its own step-0 AND
    ARC-Easy-val n=200 within 2 pts of control at matched steps. **Kill criterion:**
    recurrent loss diverges or ARC-Easy-val degrades >5 pts → stop, rethink
    objective, do not enter Tier B.
  - **Tier B — real run:** 20–50M tokens (wikitext-103 + C4 slice), ≥1 epoch,
    ≈15–40 GPU-h ×2 runs. Only after Tier A passes.
* Hyperparameters: LR 1e-4 cosine, warmup 3%, seq_len 1024 (2048 only if VRAM allows
  with checkpointing), micro-batch 1–2, grad-accum → effective 16.
* Training instrumentation (new): log **per-loop-position losses** every N steps —
  does pass 2's loss fall relative to pass 1 as training proceeds? Cheap, and it is
  the mechanistic figure for the paper.
* Validation cadence: wikitext ppl + ARC-Easy-val n=200 every N steps, tracked
  separately per arm; CSV logging + auto-resume per CONTINUED_PRETRAIN_PLAN.md.
* Checkpoint selection rule (pre-declared, no test peeking): select by wikitext-val
  ppl, then confirm on ARC-Easy-val.
* Budget: fits Kaggle ~30 h/week over 1–2 weeks with session resume; rented
  A100/Colab Pro compresses Tier B to hours and unlocks seq_len 2048.

### Phase 2 = Stages B + C — Halting policies on the ADAPTED checkpoint

**Stage B (frozen backbone, hours):** calibrate Policy-T, then **Stage C
(unfrozen, joint)**: train the learned halt head TOGETHER with the transformer.
Both start from the SAME Stage-A checkpoint, so quality gaps are attributable to
the policy + joint-training effect alone.

* **Stage B — Policy-T (non-neural baseline, backbone frozen):**
  - Re-run the Phase 0 grid on the adapted checkpoint; thresholds fine-tuned
    from their current values (not from scratch).
  - Operating point: highest acc_norm subject to avg recurrence ≤ 1.5.
  - Output doubles as Stage C's reference line and fallback deployment point.
* **Stage C — Policy-NN (learned halt head + transformer trained jointly):**
  - Init: Stage-A weights (backbone unfrozen) + tiny MLP head on block-boundary
    hidden states → halt distribution over {1..4} passes (~10⁴ full-precision
    params, negligible compute/memory). Early-exit-favoring prior at init so the
    untrained head defaults to low recurrence.
  - Objective: PonderNet-style expected next-token loss with a compute penalty —
    reward = Δquality − λ·Δcompute, **λ ∈ {0.5, 1.0}** (quality counts equally or
    more than compute savings, per locked decision #2).
  - Joint-training guardrails: LR for the backbone dropped ~10× vs Stage A
    (adapter refresh, not re-learning); head LR normal. Stage-B's operating point
    is the bar — if joint training ends below it, Stage C failed and we deploy B.
  - Training data: held-out slice of the CPT mix ONLY (benchmark validation sets
    stay untouched); a few GPU-h on T4.
  - Collapse monitoring: per-checkpoint halting histogram + fixed-R1/R2 accuracy
    probes; abort if p(halt@1) > 0.95 or < 0.05 without val-loss justification,
    or if val ppl degrades >2% vs Stage-A init → revert to Stage-A weights.
* Selection rule (pre-declared): pick the better policy by ARC-Easy-val acc_norm
  at avg recurrence ≤ 1.5; ties go to the simpler one (Policy-T). The loser is
  reported as an ablation, not hidden.
* Hypothesis chain: depth-randomized adaptation → pass-1 representations native →
  both policies separate cleanly → winner holds ≥ base-adapted−2 pts at
  avg recurrence ≤ 1.5; joint Stage-C ≥ frozen Stage-B (weights adapt to halting).

### Phase 3 — Full pre-registered evaluation

* Configs (7 internal, all from ≤2 trained checkpoints + original):
  1. base (original)
  2. base-adapted (control)
  3. block_rand-adapted @ fixed R1
  4. block_rand-adapted @ fixed R2
  5. block_rand-adapted @ Policy-T (tuned thresholds, avg recurrence ≤ 1.5)
  6. block_rand-adapted @ Policy-NN (learned halt head, winning λ)
  Optional 7th: block_fixed_R2-adapted (the conditional ablation arm).
* **External reference set** (new — required for the quality-per-compute claim;
  identical harness, tasks, seeds, prompt subsets):

| Model | Active params | Role |
|---|---|---|
| JetMoE-8B (our base) | ~2B | substrate |
| TinyLlama-1.1B | 1.1B | dense ref |
| Qwen2.5-1.5B (or Gemma-2B) | 1.5–2B | strong modern ref |
| Pythia-1.4B / 2.8B | 1.4–2.8B | scaling-family ref |

* Benchmark suite (loglikelihood unless noted):

| Task | n | Role |
|---|---|---|
| ARC-Challenge test | 1172 (full) | primary; pilot continuity |
| ARC-Easy test | 2376 | secondary (val split reserved for calibration) |
| HellaSwag val | 1002 | commonsense continuation |
| PIQA test | 1838 | physical reasoning |
| WinoGrande dev | 1000 (capped) | coreference |
| BoolQ val | 2000 (capped) | reading comprehension |
| SciQ test | 1000 (capped) | science QA |
| MMLU | 25/subtask × 57 = 1425 | knowledge breadth (**requires per-task-limit fix**) |
| wikitext-103 test ppl | — | canonical adaptation-quality metric |
| GSM8K n=200 | generative | only if decode path validated; reported separately, excluded from headline claims |

* Statistics: ≥3 seeds/config; identical subsets across configs; paired bootstrap
  10k resamples + McNemar; Holm-Bonferroni over the ≤6 pre-declared primary pairs;
  effect sizes with CIs. Power note: at n≈1172, paired differences <~2 pts are
  reported as "not distinguishable", never "equal".
* Reporting: Pareto plots of quality vs **measured** active GFLOPs AND wall-clock;
  Δquality/Δcompute ratios; halting-position histograms; % hitting max_loops;
  per-loop-position loss deltas; all negative results included.

## 4. Pre-declared decision gates

| Gate | Condition to proceed | Else |
|---|---|---|
| G0 | Controller repairs + regression tests pass | fix before any sweep |
| G1 | Phase 0 produces a Pareto point with avg recurrence ≤ 1.5 at ≥ base−3 pts | proceed anyway (negative result informs Phase 2) |
| G2 | Tier A passes both gates (loss ↓, ARC-Easy-val within 2 pts of control) | stop; rethink objective; no Tier B |
| G3 | Tier B treatment ≥ base control on ARC-Easy-val | contribution flips to negative/mechanistic result — still publishable, protocol unchanged |
| G4 | Policy-NN (Stage C) replaces Policy-T as primary ONLY if it beats the Stage-B tuned-threshold Pareto front by ≥2 pts acc_norm at equal avg recurrence, with val ppl not degraded >2% vs Stage-A | otherwise deploy Policy-T; NN + joint-training delta reported as ablation |
| G5 | Pre-registration file frozen (configs, seeds, splits, selection rules) before Phase 3 launch | no Phase 3 |

## 5. Infrastructure gaps (ordered by blocking priority)

- [ ] Per-task sample limits in `lm_eval_bridge.py` — BLOCKING for MMLU
  (`--limit` is per-subtask → 57 × 200 requests, guaranteed timeout; seen twice).
- [ ] Controller config surface: combine modes, decidable iteration-1 halting,
  configurable max_loops + thresholds (Phase 0 prereq) + pytest regressions.
- [ ] Depth-randomized training mode (`block_rand`: per-forward depth ∈ {1..4},
  mass biased toward 1–2) in the recurrent wrapper + training loop — Stage A prereq.
- [ ] `recurrence_mode`/`fixed_R` wrapper parameters so any checkpoint evaluates
  fixed or adaptive from identical weights.
- [ ] Threshold-calibration sweep script (reuses matrix harness; writes Pareto CSV).
- [ ] Tiny halt-head module (~10⁴ params, block-boundary inputs) + PonderNet-style
  expected-loss joint trainer (head + unfrozen backbone, backbone LR ×0.1) with
  λ sweep {0.5, 1.0} + halting-histogram logging — Stage C prereq.
- [ ] Active-FLOPs hook (router top-2 counting) — required before any Phase 3 claim.
- [ ] QLoRA training script + dataset prep + per-loop-position loss logging
  (Phase 1 prereq; largest new component).
- [ ] External-reference eval config (4 models through the same bridge).
- [ ] Resume-safe multi-seed eval loop; per-sample JSON persistence.
- [ ] Paired-stats + power-analysis + Pareto figure scripts.
- [ ] Pre-registration file frozen before Phase 3 (includes Phase-2 operating-point
  rule and gate table above).

## 6. Budget sketch (Kaggle free tier, ~30 GPU-h/week)

| Week | Work | GPU-h |
|---|---|---|
| W1 | Infra fixes + Phase 0 sweep + Tier A ×2 | ~6–10 |
| W2–3 | Tier B ×2 (block_rand + control), resumed sessions | ~30–60 |
| W4 | Phase 2: Stage-B calibration → Stage-C joint halt-head+backbone training (λ sweep) + freeze | ~8–15 |
| W5+ | Phase 3 eval (7 configs × ~8k requests × 3 seeds) + 4 external refs | ~25–45 |

A rented A100 collapses W2–3 to hours and unlocks seq_len 2048 — revisit if Tier A
passes cleanly. The conditional `block_fixed R2` ablation arm, if run, adds
~15–40 GPU-h and shifts Phase 3 by a week.

## 7. Non-goals / honesty constraints

* No quality claims from the zero-shot pilot beyond its CIs — it motivates, it does
  not confirm. The two cells that beat base are inside noise and will be described
  as such.
* Zero-shot adaptive results are reported as negative results, separated from
  post-adaptation results. Policy-T vs Policy-NN: the loser is reported as an
  ablation with its full Pareto curve — no silent drops.
* Halting distributions, % hitting max_loops=4, and per-position loss deltas are
  always reported alongside quality; a policy that quietly degenerates to
  all-halt-at-1 or never-halt is a finding, not a failure to hide.
* Collapse monitoring is honest: if Stage C collapses (p(halt@1) > 0.95 without
  val-loss justification, or val ppl degrading >2%), we report it and fall back to
  Stage-B Policy-T rather than re-tuning until it looks good.
* No leaderboard-comparable absolute scores from capped samples; cross-model
  comparisons only within our identical protocol.
* Adaptive-vs-fixed comparisons always at *measured* compute and wall-clock, never
  nominal.
* If adapted recurrence fails to match the control, the paper's contribution flips
  to the negative/mechanistic result; the protocol supports that without modification.
