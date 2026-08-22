# ARC — Adaptive Recurrence Computing

Production-ready 7-variant ARC for pretrained MoE Transformers with one-command Kaggle runs.

## Kaggle-ready one-command run

```bash
./kaggle_run_all.sh /kaggle/working/models/jetmoe-8b /kaggle/working/arc_results
```

Runs preflight checks, parity smoke, synthetic benchmarks, metrics and LM-Eval for all 7 variants with logs and a summary CSV.

## Variants

| Name | Scale | Adaptive | Builder |
|------|-------|----------|---------|
| base | base | False | `build_arc_model(source, scale='base')` |
| model_fixed | model | False | `scale='model', adaptive=False, recurrence=R` |
| block_fixed | block | False | `scale='block', adaptive=False, recurrence=R` |
| layer_fixed | layer | False | `scale='layer', adaptive=False, recurrence=R` |
| model_adaptive | model | True | `scale='model', adaptive=True, max_loops=M` |
| block_adaptive | block | True | `scale='block', adaptive=True, max_loops=M` |
| layer_adaptive | layer | True | `scale='layer', adaptive=True, max_loops=M` |

All share `ARCAdapter` interface: `model(input_ids, attention_mask=None, position_ids=None) -> RecurrenceResult`

## Adapter contract

`ARCAdapter` provides:
- `embed`, `forward_native`, `prepare`
- `forward_layer`, `forward_block`, `forward_model`
- `normalize`, `project_logits`, `final_logits`
- `num_layers`, `num_blocks`
- `lm_head_flops_per_token`, `unit_flops(scale, unit_index, seq_len, batch_size)`

## HALT heads

Adaptive variants use HALT heads to decide continue vs halt.

Features:
- entropy, entropy_delta
- JS divergence
- top-1 stability
- hidden cosine change
- recurrence count

`ThresholdController` uses these features with configurable thresholds; HALT head probability can be blended for learned control.

## Free community benchmarks

`hellaswag`, `arc_easy`, `arc_challenge`, `gsm8k`, `mmlu` — loglikelihood-only via `lm-evaluation-harness`.

Run via bridge:
```bash
pip install "lm-eval[hf]"
python -m arc.benchmarks.lm_eval_bridge --source /path/to/jetmoe-8b --variant model_adaptive --tasks hellaswag,arc_easy --device cuda
```

## Quick start

```bash
pip install -e .
python -m pytest tests -q
./kaggle_run_all.sh /path/to/jetmoe-8b ./results
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
  benchmark_all.py
  benchmark_metrics.py
kaggle_run_all.sh
```

Weights are not shipped. Place JetMoE weights under `models/jetmoe-8b/` or pass a HF path.
Parity gate `JetMoeAdapter.verify_parity` must pass before experiments.

Docs: see `DOCS.md`.
