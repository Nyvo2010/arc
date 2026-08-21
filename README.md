# Adaptive Recurrence Computing (ARC)

Can a pretrained MoE Transformer get more quality per FLOP by **allocating recurrent depth** instead of always running a fixed computation? ARC studies three recurrence scales independently at fixed recurrence values, evaluated at matched actual compute.

- Research questions and methodology: `RESEARCH_PLAN.md`
- Implementation contract: `BUILD_PLAN.md`
- Short version: `PROJECT_DESCRIPTION.md`

## The four Phase 1 models

One shared JetMoE adapter, four runtime modes (`recurrence` is a plain integer):

| Model | Scale | Forward |
|---|---|---|
| Base | `base` | one native pass (control) |
| Layer recurrence | `layer` | each transformer layer runs `R` times, `h_{l,r+1} = F_l(h_{l,r})` |
| Block recurrence | `block` | each contiguous block of `block_size` layers runs `R` times |
| Model recurrence | `model` | the whole stack runs `R` times, `H_{t+1} = F(H_t)` |

The hidden state always chains forward; it never restarts from `h0`. The final norm + LM head run exactly once. Kaggle comparison plan: base once, then each recurrent model at `R ∈ {2, 3, 4}`, scored on quality per estimated FLOP.

## Layout

```text
src/arc/
├── models/       # ARCAdapter interface + JetMoE-8B adapter (embed / layer / block / model / logits boundaries)
├── recurrence/   # BaseLM + the three fixed-recurrence runtimes
├── compute/      # analytic FLOP accounting (active experts only)
└── common/       # YAML config loading
configs/
├── base.yaml     # local weights
└── kaggle.yaml   # hub weights
```

All real-weight loads are **8-bit quantized** (bitsandbytes `load_in_8bit`); the random-init tiny model used in tests is the only exception. There is deliberately **no benchmark code** here; Kaggle's built-in benchmarks provide quality metrics, and every forward already reports estimated FLOPs and execution counts for the compute side.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest    # CPU-only tests against a random-init tiny JetMoE
```

Notebook cell on Kaggle (attach this repo as a dataset/utility script + GPU accelerator):

```python
!pip install -q -e .

import torch
from transformers import AutoTokenizer
from arc.common.config import load_config
from arc.models.registry import create_adapter
from arc.models.jetmoe import verify_parity
from arc.recurrence import BaseLM, build_recurrent_model

cfg = load_config("configs/kaggle.yaml")
adapter = create_adapter(cfg["model"]["path"], block_size=cfg["model"]["block_size"])
print(verify_parity(adapter))  # must print ok=True before any experiment

tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"])
ids = tokenizer("81 / 9 =", return_tensors="pt").input_ids.to(
    adapter.net.embed_tokens.weight.device
)

for R in (2, 3, 4):
    for scale in ("layer", "block", "model"):
        model = build_recurrent_model(scale, adapter, recurrence=R)
        result = model(ids)
        print(f"{scale} x{R}", result.state.executions, f"{result.state.compute_used:.3e}")

base = BaseLM(adapter)(ids)  # control; quality comes from Kaggle's benchmarks
```

## Design decisions pinned so far

| Question | Decision |
|---|---|
| Layer vs block | Layer = one decoder layer. Block = contiguous segment of `block_size` layers (default 4 → 6 blocks in JetMoE-24L). Fixed per experiment set. |
| Loop count semantics | Recurrence value `R` = total executions per unit (`R=1` ≡ native). Hidden state chains forward; never restarts from h0. |
| Routing | JetMoE's native router runs on every repeated layer execution; no diversity objectives or routing changes are introduced. |
| Compute metric | FLOPs = 2×MACs, active experts only (top-k), attention score term included; lm_head counted once per forward. |

Raw research docs (`BUILD_PLAN.md`, `RESEARCH_PLAN.md`, `PROJECT_DESCRIPTION.md`) describe the full program including later adaptive phases; this repository intentionally contains only the Phase 1 models.
