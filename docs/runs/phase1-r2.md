# Phase 1 Tier A — Run r2 (full budget, 2026-09-14)

Kaggle kernel v9. Status: COMPLETE. All three adaptive variants trained end-to-end on 1M tokens.

## Budget

- 977 micro-steps × 1024 tokens = 1,000,448 tokens per variant
- LoRA: 2,359,296 trainable params on 24 kv_proj modules
- Quantization: 4-bit NF4 base, fp16 compute
- Depth distribution: {1: 389, 2: 303, 3: 186, 4: 99}

## Results (val on WikiText-103 validation, depth-1 eval)

| Variant | Step-0 val | Best val | Best step | G2-lite |
|---|---|---|---|---|
| model_adaptive | 1.9553 | 1.9159 | 940 | PASS (−0.0394) |
| block_adaptive | 1.5747 | 1.5742 | 80 | PASS (−0.0005, barely) |
| layer_adaptive | 1.5747 | 1.5675 | 560 | PASS (−0.0072) |

## Analysis

- **model_adaptive**: Strongest improvement, loss dropped 2.0% over full run. Still improving at step 940 — would benefit from Tier B continuation.
- **block_adaptive**: Marginal improvement, best at step 80 then plateaued/degraded slightly. May have hit LoRA capacity limit early. Consider increasing LoRA rank or dropping this variant.
- **layer_adaptive**: Moderate improvement, best at step 560, slight degradation after. The layer-level depth control provides stable training.

All three variants pass G2-lite loss gate, unlocking Tier B. ARC-Easy probe not yet implemented (full G2 requires it).

## Artifacts (Hub: Nyvo/arc-jetmoe-recurrence-phase1, private)

- `phase1-tierA-r2-model_adaptive-best/` (adapters-best, trainable_state.pt, metrics)
- `phase1-tierA-r2-block_adaptive-best/`
- `phase1-tierA-r2-layer_adaptive-best/`

Adapter sha256 (distinct per variant):
- model: `7aa803f0…`
- block: `219efc14…`
- layer: `481f923f…`

## Changes since r1

- Fixed step-budget bug: micro-step counting instead of effective batches
- Fixed save_best/save_checkpoint to use PEFT LoRA state dict (not base model requires_grad stubs)
- best_state.pt now 9.5MB (real LoRA weights, not 876-byte stubs)
- device-agnostic loss computation (handles multi-GPU dispatch)
- map_location="cpu" for checkpoint loading (cross-device safe)
