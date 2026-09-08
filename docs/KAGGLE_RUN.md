# Kaggle Runbook — Stage A (random-recurrence CPT, Tier A ≈ 1M tokens)

## 0. What Tier A is
Feasibility gate (PLAN G2): `base` control + adaptive variants trained with
random recurrence on ~1M tokens (~61 steps at seq_len 1024 × batch 16).
Expected cost: 1–3 GPU-h **per variant**. Run `base` + `model_adaptive` first;
add `block/layer_adaptive` only if those behave.

## 1. One-time setup
1. Kaggle notebook settings: **Accelerator = GPU T4 x2** (or P100), **Internet = ON**,
   **Environment = latest**.
2. Upload weights as a Kaggle dataset (24 GB, too big for git):
   - Kaggle → Datasets → New Dataset → upload the contents of `models/jetmoe-8b/`
     (`config.json`, `model.safetensors.index.json`, all 4 shards, `tokenizer.json`, …).
   - Name it e.g. `jetmoe-8b`. In the notebook: Add input → your dataset.
3. Clone this repo / upload it as a second dataset or via `!git clone <url>`.

## 2. In the notebook (in order)
```bash
pip install -r requirements-kaggle.txt
ln -sfn /kaggle/input/jetmoe-8b /kaggle/working/jetmoe-8b   # match configs/kaggle.yaml model.path
bash scripts/preflight_check.sh /kaggle/working/jetmoe-8b configs/kaggle.yaml
python scripts/train_stage_a.py --config configs/kaggle.yaml --variant base --tier a
python scripts/train_stage_a.py --config configs/kaggle.yaml --variant model_adaptive --tier a
```
Or open `notebooks/kaggle_stage_a.ipynb`, which runs the same cells.

Recommended smoke first (2 steps, CPU-safe, no network):
```bash
python scripts/train_stage_a.py --config configs/kaggle.yaml --variant model_adaptive --dry-run
```

## 3. Session-timeout survival
- Metrics flush every row to `train_metrics.csv` / `eval_metrics.csv`; checkpoints
  every 30 steps + final, auto-resumed on relaunch. Re-running the same command
  continues from the latest checkpoint.
- After the run: **Save notebook version** (Output) to persist
  `/kaggle/working/checkpoints/stage-a/`, or copy it to a dataset.

## 4. G2 check after Tier A
Compare `checkpoints/stage-a/<variant>/{eval_metrics,train_metrics}.csv`:
- Treatment val_loss < its own step-0 val_loss?
- ARC-Easy probe: **not yet implemented** (`eval.arc_easy_probe: false`) — the
  Tier-B prerequisite before any G2 pass can be claimed. `src/arc/training/gates.py:evaluate_g2`
  reports "partial" until probe numbers exist.

## 5. Known limitations (do not tune around silently)
- LoRA targets are `[kv_proj]` only: JetMoE has no q/k/v/o_proj and its MoE
  experts are `ParallelExperts` (3D weights stock PEFT cannot LoRA). Preflight
  aborts if targets match zero modules.
- `adapter.unit_flops` is an analytic upper bound (MoE sparsity unmodeled);
  wall-clock in the CSVs is the measured number (PLAN §3: trust measured).
- Single-GPU script; T4x2 uses one GPU. If OOM: halve `seq_len` to 512
  (doubles step count for the same token budget automatically).
