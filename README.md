# ARC — Adaptive Recurrence Computing

Production-ready 7-variant ARC for pretrained MoE Transformers with one-command Kaggle runs.

## Kaggle-ready one-command run

Attach the JetMoE-8B weights as a Kaggle dataset, enable GPU (T4 x2 or P100) and
internet in notebook settings, then use **Save & Run All (Commit)** for unattended
execution (GPU sessions up to 9-12h; ~30 GPU hrs/week free quota).

```bash
./kaggle_run_all.sh /kaggle/working/jetmoe-8b /kaggle/working/arc_results cuda
```

Runs preflight checks, real-model parity, the unified 13-config experiment matrix
(base + fixed variants at recurrence {2,3,4} + adaptive variants at max_loops=4,
with full metrics: time, tokens/s, FLOPs, executions, recurrence-per-unit, RAM),
and LM-Eval for all 7 variants with logs and a summary CSV.
The checkpoint is loaded once and reused across all matrix configurations.
Results flush incrementally so partial results survive session timeouts.

### Kaggle notebook cells

Use `%pip` in the install cell so the package is installed into the active
notebook kernel. Do not run a separate 8B parity cell; the runner performs the
real-model parity gate immediately before the matrix and reuses that model.

```python
%pip install -q -e .[lmeval]
```

```python
from pathlib import Path
from huggingface_hub import login, snapshot_download

token_file = Path("/kaggle/input/hf-secret/hf_token.txt")
if token_file.exists():
    login(token_file.read_text().strip())

model_dir = "/kaggle/working/jetmoe-8b"
snapshot_download(repo_id="jetmoe/jetmoe-8b", local_dir=model_dir)
print("Model downloaded to", model_dir)
```

```bash
!bash /kaggle/working/arc/kaggle_run_all.sh /kaggle/working/jetmoe-8b /kaggle/working/arc_results cuda
```

If the install cell changes `torch`, `transformers`, or CUDA-related packages,
restart the Kaggle session before running the download and runner cells. The
runner intentionally checks the existing environment and does not reinstall
those packages.

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

`arc_challenge`, `mmlu`, `gsm8k` via `lm-evaluation-harness`. See `KAGGLE_RUN.md`
for exactly what a full Kaggle run executes, expected runtimes, and limits.

Run via bridge:
```bash
pip install "lm-eval[hf]"
python -m arc.benchmarks.lm_eval_bridge --source /path/to/jetmoe-8b --variant model_adaptive --tasks arc_challenge,mmlu,gsm8k --device cuda
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
