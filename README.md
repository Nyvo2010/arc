# Adaptive Recurrence Computing (ARC)

Can a pretrained MoE Transformer get more quality per FLOP by **dynamically allocating recurrent depth** instead of always running a fixed computation? ARC studies three recurrence scales independently — fixed first, adaptive later — and evaluates everything at matched actual compute.

- Research questions and methodology: `RESEARCH_PLAN.md`
- Implementation contract: `BUILD_PLAN.md`
- Short version: `PROJECT_DESCRIPTION.md`

## Status

Phase 1 implemented and tested (frozen model, fixed recurrence):

- `ModelRecurrenceLM` — `H_{t+1} = F(H_t)`, whole-stack loops (`src/arc/recurrence/base.py`)
- `BlockRecurrenceLM` — repeats contiguous segments of `block_size` decoder layers
- `LayerRecurrenceLM` — repeats individual transformer layers
- `JetMoeAdapter` — exposes embed / forward_layer / forward_block / forward_model / final_logits boundaries of JetMoE-8B with exact native parity (verified by test against HF forward, tolerance 1e-5)
- Router instrumentation via hooks on both MoA and MoE gates: expert utilization, load-balance entropy, gate probabilities
- Analytic FLOP accounting (active-expert only, closed-form from config dims)
- Single-token arithmetic benchmark: prompts like `81 / 9 =`, model must answer in exactly ONE token. The JetMoE tokenizer splits digits, so answers are constrained to one digit (0-9). No chain-of-thought is possible — all reasoning must happen inside the forward pass(es), which is exactly where recurrence acts.

Adaptive controllers, training, and DeepSeekMoE adapter are later phases (`controllers/`, `training/` are stubs).

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest            # 17 tests, CPU-only, uses random-init tiny JetMoE

# Smoke run without weights (tiny random model):
.venv/bin/python scripts/run_recurrence_sweep.py --config configs/model_fixed.yaml --tiny --limit 8

# Real JetMoE-8B sweep (needs ~16GB; intended for Kaggle GPU):
.venv/bin/python scripts/run_recurrence_sweep.py --config configs/model_fixed.yaml

# Aggregate quality-vs-compute tables:
.venv/bin/python scripts/analyze_results.py
```

Raw runs land in `results/raw/*.jsonl` (meta + summary + per-problem rows); summaries aggregate into `results/processed/summary.csv`.

## Design decisions pinned so far

| Question | Decision |
|---|---|
| Layer vs block | Layer = one decoder layer. Block = contiguous segment of `block_size` layers (default 4 → 6 blocks in JetMoE-24L). Configurable, fixed per experiment set. |
| Loop count semantics | `x2` means total executions = 2 (so x1 ≡ native baseline). Hidden state chains forward; never restarts from h0. |
| Routing | Natural routing, observed only. Router runs once per executed unit as in the native architecture; no diversity objectives. |
| Compute metric | FLOPs = 2×MACs, active experts only (top-k), attention score term included; lm_head counted once per forward. |
| Hard limits | Enforced by the runtime (`RecurrentLM.forward`), not by any controller. |

## Kaggle workflow

Scripts are CLI-driven and hardware-auto-detecting; every JSONL record embeds full hardware/software metadata plus git sha. Push this repo, attach `models/jetmoe-8b` as a dataset, and run the sweep scripts in a notebook cell (`!python scripts/run_recurrence_sweep.py ...`). Checkpointing/training integration comes with the Phase 2 training work.
