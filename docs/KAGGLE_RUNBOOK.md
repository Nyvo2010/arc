# Kaggle Runbook — Phase 1 quality-per-compute experiments

Free-tier facts (verified 2026-08): **30 GPU-hours/week** (resets Sat 00:00 UTC), sessions up to **12h**, hardware = P100 (16GB) or T4 x2, internet toggle required (phone-verified account), `/kaggle/working` = 20GB auto-saved. JetMoE-8B int8 fits either GPU; prefer **T4 x2**.

## One-time account setup

1. Verify your account with a phone number (required for GPU + internet).
2. Accept the gated GPQA dataset terms once: <https://huggingface.co/datasets/Idavidrein/gpqa> (log in with HF).
3. Create a fine-grained HF token (read access) and add it in Kaggle: Notebook → Add-ons → Secrets → `HF_TOKEN` with *attached to this notebook*.
4. Enable **Internet: On** and **Accelerator: GPU T4 x2** in notebook settings.

## Per-session setup

```python
!pip install -q -e . "lm_eval[hf]"

import os
from huggingface_hub import login
login(os.environ["HF_TOKEN"])          # only needed for GPQA
```

The repo reaches the notebook by attaching the GitHub repo as a Kaggle dataset/utility script, or `!git clone https://github.com/Nyvo2010/arc.git && %cd arc`.

## The experiment grid

10 variants — base once + layer/block/model at R ∈ {2, 3, 4}:

```bash
# base control (R is irrelevant; runs exactly one native pass)
arc-lm-eval --model arc --model_args scale=base,path=jetmoe/jetmoe-8b \
    --tasks mmlu,gpqa_diamond_zeroshot,hellaswag --limit 500 \
    --batch_size auto:2 --device cuda:0

# recurrent variants — ONLY scale and recurrence change
for S in layer block model; do for R in 2 3 4; do
arc-lm-eval --model arc --model_args scale=$S,recurrence=$R,path=jetmoe/jetmoe-8b \
    --tasks mmlu,gpqa_diamond_zeroshot,hellaswag --limit 500 \
    --batch_size auto:2 --device cuda:0 --output_path /kaggle/working/${S}_x${R}
done; done
```

Before burning quota on a variant, run the parity gate:

```python
import sys; sys.path.append("src")
from arc.common.config import load_config
from arc.models.registry import create_adapter
from arc.models.jetmoe import verify_parity

cfg = load_config("configs/kaggle.yaml")
adapter = create_adapter(cfg["model"]["path"], block_size=cfg["model"]["block_size"])
print(verify_parity(adapter))   # must print ok=True before any experiment
```

## Scientific control protocol (non-negotiable)

Everything constant except the evaluated variant. Within one experiment set:

- identical task list, `--limit`, few-shot settings (`--num_fewshot` if used), seeds (`--seed 1234`)
- identical tokenizer (same `path`, hence same tokenizer), same int8 quantization path
- identical batch/device configuration per comparable pair where feasible
- **only `scale` and `recurrence` differ**

Record for every run: harness metrics (JSON in output dir) + `arc_model.total_flops_used` / `total_executions`. Quality-per-compute = accuracy ÷ total FLOPs.

## Quota budgeting (30 GPU-h/week)

| Item | Rough cost |
|---|---|
| Model load (int8) | ~5 min each |
| Full MMLU (14k × 4 choices) | ~2–4 h per variant |
| MMLU `--limit 500` | ~15–25 min per variant |
| GPQA diamond (198 Qs) | ~5–10 min per variant |
| HellaSwag `--limit 500` | ~10–15 min per variant |

Plan: pilot the whole grid at `--limit 100` first (~1 session), then scale the winning scales to `--limit 2000` across sessions. Use **Save Version → Run All (commit mode)** so runs continue in background; results land in `/kaggle/working` and are kept with the notebook version.

## Notes & pitfalls

- Only loglikelihood/MCQ tasks work (`mmlu*`, `gpqa*`, `hellaswag`, `arc_*`, `piqa`, `boolq`, `winogrande`, `lambada_openai`). Generative tasks need `generate()`, which ARC wrappers do not implement.
- `lm-eval ls tasks` lists everything.
- If OOM: drop to `--batch_size 1`; int8 JetMoE needs headroom beyond its ~9GB of weights.
- Session died mid-grid? Rerun only missing variants — outputs are per-variant directories.
