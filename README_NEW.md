# ARC — Adaptive Recurrence Computing

Production-ready implementation of 7 recurrence variants for pretrained MoE Transformers.

## Quick start

```bash
pip install -e .
python -m tests.test_parity
python -m tests.test_smoke
```

Benchmarks with synthetic data:
```bash
python benchmark.py --source models/jetmoe-8b --device cpu --output results.csv
```

## Variants

| Name | Scale | Adaptive |
|---|---|---|
| base | base | False |
| model_fixed | model | False |
| block_fixed | block | False |
| layer_fixed | layer | False |
| model_adaptive | model | True |
| block_adaptive | block | True |
| layer_adaptive | layer | True |

All share `ARCAdapter` interface and return `RecurrenceResult(logits, final_hidden, state)`.

## Free community benchmarks

Use `lm-evaluation-harness` for recognized tasks:
- `hellaswag`
- `arc_easy` / `arc_challenge`
- `gsm8k`
- `mmlu`

Run via bridge:
```bash
pip install "lm-eval[hf]"
python -m arc.benchmarks.lm_eval_bridge --source /path/to/jetmoe-8b --variant model_adaptive --tasks hellaswag,arc_easy --device cuda
```

Kaggle script:
```bash
./run_lm_eval.sh /kaggle/working/models/jetmoe-8b /kaggle/working/arc_results
```

## Repo layout

- `src/arc/models/` — adapters
- `src/arc/recurrence/` — fixed & adaptive runtimes, controllers, halt heads
- `src/arc/inference.py` — unified inference engine
- `tests/` — parity, smoke, contract tests
- `benchmark.py`, `benchmark_full.py` — CSV benchmarks
- `run_benchmarks.sh`, `run_lm_eval.sh` — Kaggle ready

Weights are not shipped. Place JetMoE weights under `models/jetmoe-8b/` or pass a HF path.
