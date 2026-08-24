# ARC — Adaptive Recurrence Computing

Quality-per-compute research on block-recurrent JetMoE-8B with learned halting.
Branch `stage-a-cpt`: trimmed for the three-stage training curriculum
(Stage A backbone CPT → Stage B threshold calibration → Stage C joint halt-head
training). Design and pre-declared gates live in `RESEARCH_PLAN.md` (v5).

Model-level recurrence is removed (locked decision #5); supported variants:

| Name | Scale | Adaptive | Builder |
|------|-------|----------|---------|
| base | base | False | `build_arc_model(source, scale='base')` |
| block_fixed | block | False | `scale='block', adaptive=False, recurrence=R` |
| layer_fixed | layer | False | `scale='layer', adaptive=False, recurrence=R` |
| block_adaptive | block | True | `scale='block', adaptive=True, max_loops=M` |
| layer_adaptive | layer | True | `scale='layer', adaptive=True, max_loops=M` |

All share the `ARCAdapter` interface: `model(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult`

## Adapter contract

`ARCAdapter` provides:
- `embed`, `forward_native`, `prepare`
- `forward_layer`, `forward_block`, `forward_model`
- `normalize`, `project_logits`, `final_logits`
- `num_layers`, `num_blocks`
- `lm_head_flops_per_token`, `unit_flops(scale, unit_index, seq_len, batch_size)`

(`forward_model` remains on the adapter for parity verification and FLOPs accounting.)

## Halting

Adaptive variants use HALT heads to decide continue vs halt.

Features:
- entropy, entropy_delta
- JS divergence
- top-1 stability
- hidden cosine change
- recurrence count

`ThresholdController` uses these features with configurable thresholds; HALT head probability can be blended for learned control. Stage B calibrates these thresholds on the adapted checkpoint; Stage C replaces them with a tiny learned head trained jointly (PonderNet-style expected loss + compute penalty).

## Evaluation

Loglikelihood benchmarks via `lm-evaluation-harness` through the bridge:

```bash
pip install "lm-eval[hf]"
python -m arc.benchmarks.lm_eval_bridge --source /path/to/jetmoe-8b --variant block_adaptive --tasks arc_challenge,mmlu,gsm8k --device cuda
```

Throughput/halting matrix (`RESULTS.CSV` protocol) and the future Phase 0
threshold-calibration sweep both reuse `scripts/benchmark_matrix.py`
(see `run_benchmarks_matrix.sh`).

## Quick start

```bash
pip install -e .
python -m pytest tests -q
```

## Repo layout

```
src/arc/
  models/                 # JetMoe adapter + registry + factory
  recurrence/             # fixed & adaptive runtimes, controllers, halt heads, state
  benchmarks/lm_eval_bridge.py
  inference.py            # unified inference engine
scripts/
  preflight_check.sh
  benchmark_matrix.py     # matrix harness; calibration sweeps build on this
run_benchmarks_matrix.sh
```

Weights are not shipped. Place JetMoE weights under `models/jetmoe-8b/` or pass a HF path.
Parity gate `JetMoeAdapter.verify_parity` must pass before experiments.

Docs: `RESEARCH_PLAN.md` (authoritative), `CONTINUED_PRETRAIN_PLAN.md`
(Kaggle ops details; partially superseded — see banner).
