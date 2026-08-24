# Continued Pre-training on Kaggle — Efficient ARC Adaptation

> **Status: partially superseded by `RESEARCH_PLAN.md` (v5).** The data strategy,
> run design, and resume/session management below remain the operational reference
> for Stage A. Superseded content: the three per-granularity adapters
> (model-level is dropped; block_rand is the treatment arm per Phase 1),
> the 500k-1M warm-up budget (replaced by the Tier A ~1M feasibility gate and
> Tier B 20-50M token budget), and the old LR schedule (now LR 1e-4 cosine,
> warmup 3%, QLoRA r=16 alpha=32 on attention projections).

Goal: adapt JetMoe-8B weights to recurrence architecture using Kaggle free GPU quota as efficiently as possible. Design for checkpointing, resume, and minimal waste.

## Constraints
* Free quota ~30 GPU hrs/week, T4 x2 / P100 16GB VRAM, session 9-12h
* Internet required for HF download, but once weights are local, runs offline
* Only /kaggle/working persists; /kaggle/input is read-only
* Never reinstall torch; use image-built CUDA

## Data strategy — free and small
* Corpus: 500k-1M tokens warm-up from OpenWebText2 or C4 subset via HF datasets streaming
* Stream in 1k batches to avoid disk blow-up
* Pre-tokenize once, save as shards under /kaggle/working/data/; reuse across runs
* Use 2048 seq length, batch 64 gradient accumulation = 256 effective

## Pre-training protocol per granularity
Three separate adapters: model_fixed, block_fixed, layer_fixed. Train sequentially, reuse base checkpoint.

Stage 1: Warm-up 500 steps LR 1e-5, freeze embeddings
Stage 2: Full 1-2 epochs LR 5e-6 cosine
Stage 3: Validation perplexity vs base on 10k held-out tokens

Success: adapted ≤ base +0.02 ppl at same compute, ARC-Easy 200 samples ≥ base at matched compute

## Kaggle-efficient run design
1. Preflight: check GPU, VRAM, data shards exist
2. Load JetMoe-8B 8-bit once, keep in memory
3. Build recurrent model with scale X, recurrence 2
4. Train with gradient checkpointing, torch.compile if available
5. Checkpoint every 500 steps to /kaggle/working/checkpoints/scale_X_step{N}.pt
6. Log metrics to wandb-free CSV: loss, ppl, lr, tokens/s, VRAM
7. Auto-resume: scan checkpoints, load latest

## Resume and session management
* Save & Run All (Commit) for unattended 9-12h
* Split training into 3 sessions per scale: warm-up, main, validation
* Use milestone alerts: log to stdout every 50 steps, monitor first 5 min then auto-cancel if tokens/s < threshold
* If session killed, next run loads latest checkpoint automatically

## Compute budget estimate
* Warm-up 500 steps ~30 min
* Main 1 epoch ~6-8h on T4 x2
* Validation 20 min
Total ~8h per scale → ~24h per cycle. Fits within weekly quota if spread over 1 week with 3 sessions.

## Output for grant application
* Loss curves, perplexity vs steps
* Quality per compute on 3 probe benchmarks after each scale
* Checkpoint hashes, config logs
* Cost report: GPU hrs used, tokens processed per hour

## Next steps
Implement training loop with resume, data streaming, and metric logging. Start with model_fixed scale, validate adaptation, then proceed to block/layer.