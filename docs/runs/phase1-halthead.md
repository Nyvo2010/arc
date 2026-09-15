# Phase 1 — Halt-Head Formula (Policy-T): Probability Score + Threshold

Status: PLAN (approved direction). Formula-based only — no trained head.
Goal: with the fully-trained recursive variants, run the *minimal* number of
loops that still yields the best possible results (mean loops ~half of today,
accuracy statistically equal to the best depth).

## Core idea

Compute ONE score from all informative signals per loop-pass, convert it to a
halt probability, compare against a threshold:

```
halting score  →  p_halt  →  if p_halt >= threshold: STOP, else: CONTINUE
```

Only a formula + a threshold. The "probability" is a normalized score, not a
learned network.

## Inputs to the formula (per loop pass)

| Signal | Meaning | Direction that favors HALT |
|---|---|---|
| `js_divergence` | distribution change vs previous pass | small |
| `hidden_cosine_change` | representation drift vs previous pass | small |
| `top1_stability` | how much the top token kept flipping | high (stable) |
| `entropy` | uncertainty of the output distribution | low |
| `entropy_delta` | confidence trend (this pass − prev) | positive (getting more confident) |
| `nll_delta` | next-token loss this pass vs previous | negative (loss still dropping = keep going) |
| `recurrence_count` | passes already done vs `max_loops` | close to max = time to stop |

## The formula

Normalize each signal to ~[0,1] using reference scales, then combine with
weights so a HIGH score means "converged, safe to halt":

```
converged_score =
    w_js      * (1 − js_norm)
  + w_hidden  * (1 − hidden_norm)
  + w_top1    * top1_stability
  + w_entropy * (1 − entropy_norm)
  + w_conf    * entropy_delta_norm            # rising confidence → halt
  + w_loss    * (1 − nll_improvement_norm)    # no more loss dropped → halt
  + w_loops   * (recurrence_count / max_loops)

p_halt = sigmoid( k * (converged_score − bias) )

HALT  if  p_halt >= threshold   (default = STOP)
CONTINUE if  p_halt <  threshold
```

Decision rule (redundant safety):
- Always run at least 1 pass (never halt before loop 1).
- Stop if `p_halt >= threshold` OR the score barely moved vs previous loop
  (diminishing returns: `converged_score(d) − converged_score(d−1) < 0.02`).
- `max_loops` is a hard safety cap only, not the operative bound.

## "Best results" guardrail

Halting must never sacrifice accuracy to save loops. So the chosen threshold
is constrained by, on a held-out slice: `acc >= best_depth_acc − 0.2pt`. Among
thresholds satisfying that, pick the one with the smallest mean loops.

## Calibration (constants only, no training)

1. After benchmark suite runs, log per-sequence features per loop on a
   held-out slice (ARC-Easy + HellaSwag) through the real inference path.
2. Small grid over `(w*, k, bias, threshold)` — no optimizer, minutes of GPU.
3. Pick per the "best results" rule above.
4. Bake the constants into `ThresholdController` defaults.

## Acceptance criteria

- `mean_loops` ~2 (vs ~4 today) on the budgeted eval,
- accuracy within 0.2pt of the best measured depth for each trained variant,
- easy tokens stop at 1–2, hard tokens may run deep (per-sequence variance,
  not a blanket cap).

## Backlog (implementation order, after Tier B done)

1. `--budgeted` eval mode in `run_benchmarks.py` + eval kernel (reports
   `mean_loops` AND `acc` together; runs real `decide()` path, not fixed-depth).
2. Rewrite `ThresholdController.decide` → score/threshold rule (default STOP),
   keep `InferenceEngine` interface unchanged.
3. Calibration script producing the frontier table + winning constants.
4. Bake constants; re-benchmark in budgeted mode; document outcome per variant.