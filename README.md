# Adaptive Recurrence Computing (ARC)

Can a pretrained MoE Transformer get more quality per FLOP by **allocating recurrent depth** instead of always running a fixed computation? Phase 1 studies three recurrence scales independently at fixed recurrence values, evaluated at matched actual compute on Kaggle.

- Research questions and methodology: `RESEARCH_PLAN.md`
- Implementation contract: `BUILD_PLAN.md`
- Universal adapter & benchmarking contract: `docs/ADAPTER_CONTRACT.md`
- Short version: `PROJECT_DESCRIPTION.md`
- Kaggle operations runbook: `docs/KAGGLE_RUNBOOK.md`

## The four Phase 1 models

One shared JetMoE adapter, four runtime modes. `recurrence` (`R`) is a plain integer:

| Model | `scale` | Forward |
|---|---|---|
| Base | `base` | one native pass (control) |
| Layer recurrence | `layer` | each transformer layer runs `R` times, `h_{l,r+1} = F_l(h_{l,r})` |
| Block recurrence | `block` | each contiguous block of `block_size` layers runs `R` times |
| Model recurrence | `model` | the whole stack runs `R` times, `H_{t+1} = F(H_t)` |

The hidden state always chains forward and never restarts from `h0`. Between complete model loops the final RMSNorm is applied so the chained state equals a real native traversal. The final norm + LM head run exactly once per forward.

Experiment grid: base once + each recurrent model at `R ∈ {2, 3, 4}` = **10 variants**.

## Repository architecture

```text
src/arc/
├── models/
│   ├── base.py        # ARCAdapter interface: embed / forward_layer / forward_block /
│   │                  #   forward_model / normalize / project_logits / unit_flops /
│   │                  #   lm_head_flops_per_token
│   ├── jetmoe.py      # JetMoeAdapter (JetMoE-8B boundaries), tiny test model,
│   │                  #   8-bit-only loader, verify_parity gate
│   └── registry.py    # create_adapter(source) -> ARCAdapter
├── recurrence/
│   ├── base.py        # BaseLM (native control), the three RecurrentLM runtimes,
│   │                  #   build_recurrent_model(scale, adapter, R)
│   └── state.py       # RecurrenceState: executions, loop counts, compute_used
├── compute/
│   └── flops.py       # analytic FLOPs: 2×MACs, active experts only, seq×batch scaled
├── common/config.py   # YAML config loading
├── benchmarks/
│   ├── protocol.py    # ModelVariant (the only allowed variables) +
│   │                  #   BenchmarkProtocol (frozen tasks/seeds/batch/device)
│   ├── runner.py      # run_suite: one harness config for all variants
│   └── cli.py         # arc-benchmark console entry point
└── lmeval.py          # OPTIONAL lm-evaluation-harness bridge (model type "arc")
configs/
├── base.yaml          # local weights
└── kaggle.yaml        # hub weights for Kaggle
tests/                 # CPU-only suite on a random-init tiny JetMoE
docs/KAGGLE_RUNBOOK.md # step-by-step free-tier benchmark operations
```

Everything else (benchmarks, controllers, training, other base models) is deliberately **not in this repo**; see *Development strategy* below.

All real-weight loads are **8-bit quantized** (`bitsandbytes load_in_8bit`, `device_map="auto"`); the random-init tiny model used in tests is the only exception.

## Compute accounting

Every forward reports `state.executions` and `state.compute_used` — analytic FLOPs = 2×MACs with active-expert-only cost, attention score term included, scaled by sequence length × batch size × recurrence count, plus the LM-head projection counted once. The lm-eval shim accumulates totals across a whole benchmark run (`arc_model.total_flops_used`), which is the "compute" half of quality-per-compute.

## Testing strategy

The suite is **CPU-only** (random-init 2-layer JetMoE, ~40s total) and gates correctness at three levels:

1. **Parity** (`test_parity.py`) — the decomposed adapter path must reproduce the native HF forward exactly (`< 1e-5`), including padded batches with explicit masks; model-recurrence chaining must equal two normalized native passes; `final_hidden` must be the normalized state.
2. **Recurrence semantics** (`test_recurrence.py`) — execution counts, per-unit loop counts, output changes with depth, invalid-argument rejection, x1 recurrence ≡ base logits.
3. **Compute accounting** (`test_units.py`) — FLOP formulas scale correctly with sequence length, batch size, and recurrence.
4. **Benchmark bridge** (`test_lmeval.py`) — offline loglikelihood through the real harness plus a networked `simple_evaluate` round-trip that skips gracefully offline.

Run everything:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev,lmeval]'
.venv/bin/python -m pytest
```

## Kaggle Notebook benchmark workflow

The repository provides one universal `lm-eval` backend and one fixed benchmark protocol for every architecture and variant. A Kaggle Notebook runs that protocol against Kaggle-hosted or otherwise available weights. This is a reproducible notebook experiment; it is not an automatic submission to Kaggle's hosted Benchmarks leaderboard. Full setup and quota budgeting: `docs/KAGGLE_RUNBOOK.md`.

```bash
!pip install -q -e . "lm_eval[hf]"
!arc-benchmark --path jetmoe/jetmoe-8b --limit 500 \
    --output /kaggle/working/arc-benchmark.json
```

The benchmark runner evaluates the base control plus `l`, `b`, and `m` at recurrence 2, 3, and 4. Future architectures add an adapter under `src/arc/models/` and a registry branch; the runner and inference engine do not change.

### Scientific control protocol

Within an experiment set, **everything stays constant except the evaluated model variant**: identical tokenizer, tasks, few-shot counts, limits, seeds, quantization, batch settings, and hardware. The variant may change architecture, weights, recurrence scale (`l`, `b`, or `m`), and recurrence count. Quality comes from the harness; compute comes from `total_flops_used`; report accuracy per FLOP.

## Development strategy / where things go

Phase discipline keeps attribution clean — each phase adds code only where it belongs:

| Concern | Lives in | Status |
|---|---|---|
| New base-model adapters (DeepSeekMoE etc.) | `src/arc/models/<name>.py` + registry branch | future phase |
| Adaptive HALT/CONTINUE controllers | `src/arc/controllers/` | later phase |
| Recurrence-aware LoRA training | `src/arc/training/` | after untrained comparison |
| Benchmark task definitions | never here — use lm-eval tasks or Kaggle datasets | n/a |
| Result aggregation notebooks | Kaggle notebook, outputs to `/kaggle/working` | per experiment |

When a future phase lands, recreate its package rather than growing the model files; the adapter interface in `models/base.py` is the stable contract between phases.
