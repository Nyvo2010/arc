# Phase 1 Stage B — Formula-based Halt Controller (Policy-T, no learned head)

Status: DESIGN. Target: refined closed-form HALT/CONTINUE rule, tuned constants
only (no trained parameters), evaluated in budgeted mode against the best
Tier B weights.

## Motivation (observed failure)

`src/arc/recurrence/controller.py` `ThresholdController.decide` runs nearly
every token to `max_loops` (4):

- CONTINUE is any-OR of loose thresholds (`js>0.01`, `hidden_change>0.01`,
  `top1_stab<0.95`, `entropy_delta<−0.001`) — real distributions rarely sit
  at zero motion on all four, so the OR is almost always true.
- The `entropy_delta < −0.001 → continue` branch continues when the model is
  becoming MORE confident (backwards).
- No cost gate: a loop with diminishing returns still runs.
- No calibration: thresholds are unverified constants; `compute_budget` unused.

Result: ~4 loops/token by default, even when 2–3 loops give the best results.

## Requirement (from project owner)

- Formula only. No trained/learned halt head.
- Goal: **optimize for best results, not minimum loops.** Per sequence/token,
  stop at the depth where further recurrence stops improving the answer.
  Extra loops that don't help (or hurt) must not run.

## Controller formula

Replace the OR with a scalar instability score; halt when the score drops
below a tuned threshold OR the projected marginal gain is non-positive.

```
normalized terms (0..1 scale via running reference values on the calibration set):
  js_n      = js_divergence / ref_js
  hidden_n  = hidden_cosine_change / ref_hidden
  top1_n    = (1 − top1_stability) / (1 − ref_top1_stability)
  entropy_n = entropy / ref_entropy
  conf_n    = clamp((entropy_prev − entropy_cur) / ref_entropy, 0, 1)   # confidence rising

instability(d) = w_js·js_n + w_hidden·hidden_n + w_top1·top1_n +
                 w_entropy·entropy_n − w_conf·conf_n

continue ⇔ instability(d) > θ
            AND (d == 1 OR instability(d) ≥ α·instability(d − 1))   # marginal-gain gate
            AND d < max_loops AND compute_used < budget
```

Notes:
- `α` enforces diminishing-return stop: loop d+1 only if the sequence is still
  "as unsettled" as it was at loop d (scaled by α < 1).
- `conf_n` term makes the rule stop when confidence is rising and nothing else
  is moving — the primary "I've converged" signal.
- Recurrence at depth-1 is always executed (never halt before first loop must
  be allowed via `d==1` guard on instability only... see caveat in impl).

## "Optimize for best results" objective (operating point)

For each candidate constant set `(w, θ, α, max_loops, budget)` evaluate on a
held-out slice and record `(mean_loops, acc)` per model. Choose:

1. Let `acc_max` = best accuracy achievable across all depth settings
   (from the fixed-depth sweep on the same slice).
2. Feasible set = settings with `acc ≥ acc_max − ε` (default ε = 0.2 accuracy
   points) — "no meaningful loss" trades first.
3. Pick from the feasible set: first minimize mean_loops, then maximize acc.
   If a setting has LOWER acc than a strictly-deeper-one could give, it is
   rejected even at fewer loops.

This makes fewer-loops a tiebreaker, never a reason to give up accuracy — the
halt head optimizes results, and only then compute.

Per-sequence marginal-gain alignment: on the calibration slice, mark each
sequence as "gained from loop d→d+1" if continuing changed the argmax toward
the label or reduced NLL by > tiny δ. The score surface is fitted so that
`instability(d) > θ` correlates with "sequence still gains from one more loop"
(flip-rate alignment ≥ 0.5 by construction via grid search; reported in the
spec as the calibrated confusion per depth).

## Calibration procedure (no gradients)

1. Instrument the inference path (`InferenceEngine`/adaptive `decide`) to log
   per sequence: all raw features per loop, final depth, and what depth would
   have produced the best answer for that sequence.
2. Sweep grid: `w ∈ {0..2}`, `α ∈ {0.5..0.9}`, `θ ∈ [0.01..1]`,
   `max_loops ∈ {2,3,4}`, `budget ∈ {None, 0.5×max, 0.75×max}`.
3. Score each setting on the held-out slice; pick per "best-results" rule.
4. Hard-code winning constants as `ThresholdController` defaults; log the
   frontier table (mean_loops × acc per setting) to `docs/runs/`.

## Budgeted eval mode (harness)

New mode in `scripts/run_benchmarks.py` (and eval kernel) using the real
controller path:

- `--budgeted` runs `score_choices` through the controller-based forward
  (not fixed-depth) and reports BOTH `mean_loops` (incl. per-seq distribution)
  and `acc` per task, plus `compute_full_passes` (FLOPs relative to base).
- Fixed-depth sweep stays available for the compute-accuracy frontier.
- Compare lines: base (1 pass), budgeted controller, depth-1..4 sweeps.

## Dependencies / ordering

1. Tier B 20M-token weights for all 3 variants (in flight) + benchmark suite run.
2. Budgeted eval mode implementation.
3. Calibration on ARC-Easy + HellaSwag held-out slices (per variant).
4. Bake constants; re-benchmark in budgeted mode; verify `acc ≥ acc_best − ε`
   with `mean_loops ≥ ~50% reduction` OR document that the model needs deeper
   passes (frontier says so).

## Out of scope

- Learned/neural halt head (Policy-NN) — NOT wanted.
- Module offloading / speculative halting.