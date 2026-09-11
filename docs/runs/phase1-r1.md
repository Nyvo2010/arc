# Phase 1 Tier A — Run r1 (smoke-scale, 2026-09-11)

Kaggle kernel v8. Status: COMPLETE. All three adaptive variants trained end-to-end.

## Budget caveat

A step-accounting bug counted effective batches instead of micro-steps, so r1
processed **63,488 tokens per variant (~6% of the 1M Tier A target, ~4 optimizer
steps)**. Results below are pipeline validation, NOT a feasibility signal.
Fixed in `2626f13`; full-budget repeat is r2 (kernel v9).

## Results (val on WikiText-103 validation, depth-1 eval)

| Variant | Step-0 val | Best val | Best step | G2-lite |
|---|---|---|---|---|
| model_adaptive | 1.9553 | 1.9539 | 60 | PASS (margin 0.0014) |
| block_adaptive | 1.5747 | 1.5747 | 20 | FAIL (tie — no improvement) |
| layer_adaptive | 1.5747 | 1.5721 | 60 | PASS |

Depth histogram (shared seed): {1: 17, 2: 21, 3: 16, 4: 8} over 62 micro-steps.
Train config hash: `e4da29e21de3adfd`. LoRA: 2,359,296 trainable params on 24 `kv_proj` modules, 4-bit NF4 base, fp16.

## Artifacts (Hub: Nyvo/arc-jetmoe-recurrence-phase1, private)

- `phase1-tierA-r1-model_adaptive-best/` (adapters + metrics)
- `phase1-tierA-r1-block_adaptive-best/`
- `phase1-tierA-r1-layer_adaptive-best/`

Adapter safetensors sha256 (distinct per variant, as expected):
- model: `dbfc3e00…`
- block: `dc9a444e…`
- layer: `bf11d91b…`

## Bugs found and fixed by this run

1. `src/arc/training/*` never committed → Kaggle ImportError (fixed `9c99d20`).
2. 8-bit base + `prepare_model_for_kbit_training` fp32 upcast → T4 OOM before step 0 (fixed: 4-bit NF4, manual kbit prep).
3. `prepare` fp32 norm cast vs fp16 compute → half/float matmul error (fixed: uniform fp16).
4. `device_map: auto` split across T4x2 → loss device mismatch (fixed: device-agnostic loss).
5. `device_map {"": 0}` single-GPU → load-time OOM on fp16 MoE experts (reverted to auto; experts don't quantize).
6. `state_dict()` filtering by `requires_grad` saves 876-byte stubs (fixed: PEFT LoRA state dict).
7. Token budget under-counted 16× (fixed: micro-step accounting; r2 runs the real 1M).

## Open items

- HF_TOKEN Kaggle secret not attached to pushed versions → Hub push from kernel
  skipped; r1 artifacts were uploaded from local instead. Attach the secret on
  kaggle.com (kernel → Add-ons → Secrets → HF_TOKEN) for automatic pushes.
- ARC-Easy probe still unimplemented → full G2 cannot pass; Tier B gated on r2
  loss improvement + probe landing.
