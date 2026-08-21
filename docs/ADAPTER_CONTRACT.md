# Universal adapter & benchmarking contract

This document is the stable contract for **every future phase** of ARC. If code
and this document disagree, fix one of them immediately — do not let them drift.

## The scientific rule

Within one experiment set, **only the model variant changes**. Tasks, prompts,
few-shot count, limits, seeds, quantization, batching, device, and output format
are frozen by construction in `arc.benchmarks`. A future architecture must never
require editing the benchmark runner, the CLI, or the lm-eval bridge.

## Canonical recurrence scales

| Canonical name | Internal scale | Meaning |
|---|---|---|
| `base` | `base` | native one-pass control |
| `l` | `layer` | each transformer layer runs `R` times |
| `b` | `block` | each contiguous block of `block_size` layers runs `R` times |
| `m` | `model` | whole stack runs `R` times, `H_{t+1} = F(H_t)` |

The translation `l/b/m → layer/block/model` happens exactly once, in
`ArcLM.__init__` (`src/arc/lmeval.py`). Everything above that boundary uses the
canonical names only.

## ModelVariant schema (`arc.benchmarks.protocol.ModelVariant`)

The **only** fields allowed to differ between comparable runs:

| Field | Type | Default | Validation |
|---|---|---|---|
| `architecture` | str | required | must have a registry branch |
| `path` | str | required | HF repo id or `"tiny"` |
| `scale` | str | `base` | one of `base, l, b, m` |
| `recurrence` | int | 1 | `>= 1`; forced to `1` for `base` |
| `block_size` | int | 4 | block partition width |

Anything else that a future model needs (controller config for dynamic
recurrence, LoRA adapters, etc.) is added to **this dataclass**, never as loose
CLI flags — that keeps the invariant enforceable in one place.

## Frozen benchmark protocol (`BenchmarkProtocol`)

Defaults, pinned by construction:

- Tasks: `mmlu`, `mmlu_pro`, `gpqa_main_zeroshot`, `gpqa_diamond_zeroshot`
  (exact installed-harness names; guarded by a test that resolves them via
  `TaskManager.match_tasks`)
- Zero-shot: `num_fewshot=0` explicitly overrides each task's own default
  (notably `mmlu_pro`, whose yaml defaults to 5-shot CoT)
- Seeds: `1234` for random/numpy/torch/fewshot
- Limit: `500` — **applies per subtask**, so group `mmlu` (~57 subtasks) runs
  nearly the full set; budget accordingly or pilot with `--limit 100`
- Batch/device: `auto:2` on `cuda:0`
- Quantization: int8, enforced inside `models/jetmoe.py` loading (recorded in
  the artifact as `dtype: "int8"`)
- Generation: greedy-only (`do_sample=False`); sampling raises
  `NotImplementedError`

## Run flow and artifact

```
arc-benchmark --path jetmoe/jetmoe-8b [--limit N] [--architecture NAME]
      └── run_suite(variants, protocol, output_path)
            └── simple_evaluate(model="arc", model_args=variant.model_args(), **protocol.harness_args())
```

One JSON artifact per suite invocation:

```jsonc
{
  "protocol": { "...frozen settings..." },
  "runs": [
    {
      "variant":   { "architecture": "...", "path": "...", "scale": "l",
                     "recurrence": 2, "block_size": 4 },
      "protocol":  { "...same frozen settings..." },
      "protocol_fingerprint": "<sha256 of protocol dict>",
      "accounting": { "total_flops_used": 0.0, "total_executions": 0 },
      "results":   { "...raw simple_evaluate payload..." }
    }
  ]
}
```

Quality-per-compute = accuracy metrics ÷ `total_flops_used`.
`ArcLM.last_instance` carries accounting from the constructed model back into
the artifact.

## Adding a future architecture (the only permitted change)

1. Create `src/arc/models/<name>.py` implementing `ARCAdapter`
   (`src/arc/models/base.py`). Preserve embeddings, positional handling,
   norms, residuals, attention, routing, experts, head.
2. Add one branch in `create_adapter(...)` keyed on `architecture="<name>"`.
3. Provide a tiny random-init build for the CPU test suite and a parity gate
   (`verify_parity`) mirroring `jetmoe.py`.
4. Nothing else changes: not `benchmarks/`, not `recurrence/`, not `lmeval.py`.

Phase discipline stays as documented in README.md — controllers go in
`src/arc/controllers/`, training in `src/arc/training/`, etc.

### Dynamic recurrence (future)

The contract already reserves room: dynamic variants are just `ModelVariant`
values where a future field selects a controller instead of a fixed `R`.
When that phase lands, extend `ModelVariant` and add the controller package;
do **not** add new benchmark flags or runner branches.

## Kaggle readiness checklist

Per session: internet ON, accelerator GPU T4 x2, `HF_TOKEN` secret attached,
GPQA dataset terms accepted once, then:

```bash
pip install -q -e ".[lmeval]" && arc-benchmark --path jetmoe/jetmoe-8b --limit 100
```

Operational notes:

- Pilot at low `--limit` first; full grid = base + `{l,b,m} × R∈{2,3,4}` = 10
  variants.
- `mmlu_pro` is generative CoT (`max_gen_toks=2048`) and our decoding loop has
  no KV cache — expect it to dominate wall time uniformly across variants.
- GPQA needs the gated HF dataset login even though tasks are zero-shot.
- The parity gate must print `ok=True` before burning GPU quota (see
  `docs/KAGGLE_RUNBOOK.md`).

## Non-goals

Official Kaggle **Benchmarks leaderboard** registration requires separately
hosted Kaggle model versions. This repo produces reproducible notebook-run
benchmark artifacts; leaderboard submission is a separate future integration.
