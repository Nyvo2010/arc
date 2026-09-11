"""Stage-A trainer: random-recurrence QLoRA CPT + equal-budget base control.

One process trains ONE variant (``base`` control or one adaptive scale).
Launch once per variant with the same seed/config; the ``base`` run uses
depth=1, adaptive runs sample depth per forward from ``DEPTH_DIST``.

Checkpoints (every ``checkpoint_every`` steps + final, auto-resumed):
  - ``trainable_state.pt`` (LoRA-only or full params for method=none)
  - ``optim_state.pt``, ``train_state.json`` (step, tokens, RNG, hashes)
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from arc.models.registry import create_adapter
from arc.training import causal_lm_loss, random_recurrence_forward, sample_depth
from arc.training.data import build_streaming_loader, cycling, synthetic_batch
from arc.training.metrics import (
    CSVLogger,
    EVAL_FIELDS,
    TRAIN_FIELDS,
    config_hash,
    gpu_mem_gb,
    gpu_name,
    sha256_file,
    write_provenance,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _linear_names(model) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def _device_loss(logits, input_ids):
    """Shifted next-token loss with labels moved to the logits device.

    With device_map auto the dispatched model may return logits on a
    different GPU than the input batch; the loss must live on one device.
    """
    return causal_lm_loss(logits, input_ids.to(logits.device))


def setup_qlora(hf_model, lora_cfg: dict):
    """Wrap ``hf_model`` with LoRA in place; return (peft_wrapper, trainable_names).

    The adapter keeps pointing at ``hf_model`` (mutated in place), so no
    rebinding is needed. Fails fast if no target matches a Linear module
    (e.g. stock LLaMA target names on JetMoE would silently train nothing).
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise RuntimeError("method=qlora requires 'peft' (pip install peft)") from e

    targets = list(lora_cfg.get("targets", []))
    matched = [n for n in _linear_names(hf_model) if any(n.endswith(t) for t in targets)]
    if not matched:
        cands = sorted({n.split(".")[-1] for n in _linear_names(hf_model)})
        raise RuntimeError(
            f"LoRA targets {targets} matched 0 Linear modules. "
            f"Available leaf names include: {cands[:20]}. "
            "JetMoE has no q/k/v/o_proj; use ['kv_proj']."
        )
    # Manual k-bit prep: stock prepare_model_for_kbit_training upcasts large
    # weights to fp32 and OOMs a 15GB T4. Freeze base, force ONE uniform dtype
    # (fp16 = bnb compute dtype) so quantized linears never see mixed dtypes,
    # enable input grads so per-loop gradient checkpointing has a grad source.
    for p in hf_model.parameters():
        p.requires_grad = False
    for n, p in hf_model.named_parameters():
        if p.ndim == 1:
            p.data = p.data.to(torch.float16)
    if hasattr(hf_model, "enable_input_require_grads"):
        hf_model.enable_input_require_grads()
    else:
        def _hook(module, inp, out):
            out.requires_grad_(True)
        try:
            hf_model.get_input_embeddings().register_forward_hook(_hook)
        except Exception:
            pass
    if hasattr(hf_model, "gradient_checkpointing_enable"):
        try:
            hf_model.gradient_checkpointing_enable()
        except Exception:
            pass
    try:
        cfg = getattr(hf_model, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    except Exception:
        pass
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapper = get_peft_model(hf_model, peft_cfg)
    trainable = [n for n, p in hf_model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    if n_params == 0:
        raise RuntimeError("QLoRA setup produced 0 trainable parameters; aborting.")
    print(f"[stage-a] LoRA on {len(matched)} modules, trainable params: {n_params:,}")
    return wrapper, trainable


def _cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < max(1, warmup_steps):
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_stage_a(
    cfg: dict,
    variant: str,
    output_dir: str | Path,
    tier: str = "a",
    max_steps: int | None = None,
    resume: bool = True,
    probe_fn=None,
    run_id: str | None = None,
) -> dict:
    """Run Stage-A CPT for one variant. Returns a summary dict."""
    from arc.models.registry import MODEL_VARIANTS

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {sorted(MODEL_VARIANTS)}")
    scale = MODEL_VARIANTS[variant]["scale"]

    cpt = cfg.get("cpt", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    seed = int(cfg.get("arc", {}).get("seed", 0))
    _set_seed(seed)

    seq_len = int(cpt.get("seq_len", 1024))
    eff_batch = int(cpt.get("effective_batch", 16))
    micro_batch = int(cpt.get("micro_batch", 1))
    accum = max(1, eff_batch // micro_batch)
    tier_tokens = int(cpt.get(f"tier_{tier}_tokens", cpt.get("tier_a_tokens", 1_000_000)))
    tokens_per_step = seq_len * eff_batch
    total_steps = max_steps or max(1, math.ceil(tier_tokens / tokens_per_step))

    log_every = int(cpt.get("log_every", 5))
    eval_every = int(cpt.get("eval_every", 20))
    ckpt_every = int(cpt.get("checkpoint_every", 30))
    per_loop_every = int(cpt.get("per_loop_loss_every", 0))
    val_batches = int(data_cfg.get("val_batches", 8))
    use_checkpoint = bool(cpt.get("gradient_checkpointing", True))

    run_dir = Path(output_dir) / variant if run_id is None else Path(output_dir) / run_id / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(run_dir, cfg, extra={"variant": variant, "scale": scale, "tier": tier})
    train_log = CSVLogger(run_dir / "train_metrics.csv", TRAIN_FIELDS)
    eval_log = CSVLogger(run_dir / "eval_metrics.csv", EVAL_FIELDS)

    # --- model ---
    source = model_cfg.get("path", "tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_map auto: 4-bit base + fp16 experts split across T4x2 (a single T4
    # cannot hold the fp16 MoE experts at load). Forward outputs may land on a
    # different GPU than the inputs, so every loss call moves labels to the
    # logits device (see _device_loss below).
    adapter = create_adapter(
        source, block_size=int(model_cfg.get("block_size", 4)),
        device_map=None if source == "tiny" else model_cfg.get("device_map", "auto"),
    )
    hf_model = adapter.hf_model
    if source == "tiny" and str(device).startswith("cuda"):
        hf_model = hf_model.to(device)
    method = cpt.get("method", "qlora")
    peft_wrapper = None
    if method == "qlora":
        peft_wrapper, _ = setup_qlora(hf_model, cpt.get("lora", cpt))
        # setup_qlora reads r/alpha/dropout/targets from the given mapping;
        # accept both nested `lora:` and flat cpt keys.
    elif method == "none":
        for p in hf_model.parameters():
            p.requires_grad = True
        print("[stage-a] method=none: full params trainable (tiny/test only)")
    else:
        raise ValueError(f"unknown cpt.method: {method}")
    trainable_params = [p for p in hf_model.parameters() if p.requires_grad]
    hf_model.train()

    optimizer = torch.optim.AdamW(
        trainable_params, lr=float(cpt.get("lr", 1e-4)), weight_decay=float(cpt.get("weight_decay", 0.0))
    )
    scheduler = _cosine_schedule(optimizer, int(total_steps * float(cpt.get("warmup_ratio", 0.03))), total_steps)

    # --- data ---
    synthetic = bool(data_cfg.get("synthetic", False)) or source == "tiny"
    if synthetic:
        vocab = int(getattr(adapter.cfg, "vocab_size", 128))
        rng_data = random.Random(seed)
        train_iter = cycling(
            lambda: ({"input_ids": synthetic_batch(vocab, seq_len, micro_batch, rng_data)} for _ in iter(int, 1))
        )
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_cfg.get("tokenizer_path", source), trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"tokenizer load failed from {source}: {e}") from e
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        specs = data_cfg.get("datasets", [{"name": "wikitext", "weight": 1.0}])
        val_specs = data_cfg.get("val_datasets", specs)
        train_iter = cycling(
            lambda: build_streaming_loader(specs, tokenizer, seq_len, micro_batch, seed=seed, split="train")
        )

    # --- resume ---
    step, tokens_processed, total_flops = 0, 0, 0.0
    state_path = run_dir / "train_state.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        step = int(state.get("step", 0))
        tokens_processed = int(state.get("tokens_processed", 0))
        total_flops = float(state.get("total_flops", 0.0))
        ts = run_dir / "trainable_state.pt"
        if ts.exists():
            hf_model.load_state_dict(torch.load(ts, map_location="cpu"), strict=False)
        os_ = run_dir / "optim_state.pt"
        if os_.exists():
            optimizer.load_state_dict(torch.load(os_, map_location="cpu"))
        print(f"[stage-a] resumed {variant} at step {step} ({tokens_processed:,} tokens)")

    depth_rng = random.Random(seed + step)
    depth_hist: dict[int, int] = {}
    best_path = run_dir / "best.json"
    if resume and best_path.exists():
        try:
            best_val_loss = float(json.loads(best_path.read_text()).get("best_val_loss", float("inf")))
        except Exception:
            best_val_loss = float("inf")
    else:
        best_val_loss = float("inf")
    t_start = time.perf_counter()
    t_last_log = t_start
    tok_last_log = tokens_processed
    running_loss = 0.0

    def save_checkpoint(final: bool = False) -> None:
        tag = "final" if final else f"step-{step}"
        torch.save(
            {k: v.cpu() for k, v in hf_model.state_dict().items() if v.requires_grad},
            run_dir / "trainable_state.pt",
        )
        if peft_wrapper is not None:
            peft_wrapper.save_pretrained(str(run_dir / "adapters"))
        torch.save(optimizer.state_dict(), run_dir / "optim_state.pt")
        state = {
            "step": step,
            "tokens_processed": tokens_processed,
            "total_flops": total_flops,
            "config_hash": config_hash(cfg),
            "depth_hist": depth_hist,
        }
        state_path.write_text(json.dumps(state, indent=2))
        state["ckpt_sha256"] = sha256_file(run_dir / "trainable_state.pt")
        state_path.write_text(json.dumps(state, indent=2))
        print(f"[stage-a] checkpoint {tag} @ step {step}")

    def save_best(val_loss: float) -> None:
        torch.save(
            {k: v.cpu() for k, v in hf_model.state_dict().items() if v.requires_grad},
            run_dir / "best_state.pt",
        )
        if peft_wrapper is not None:
            peft_wrapper.save_pretrained(str(run_dir / "adapters-best"))
        best = {
            "best_val_loss": val_loss,
            "best_val_ppl": math.exp(min(val_loss, 20.0)),
            "step": step,
            "tokens_processed": tokens_processed,
            "config_hash": config_hash(cfg),
        }
        best["ckpt_sha256"] = sha256_file(run_dir / "best_state.pt")
        best_path.write_text(json.dumps(best, indent=2))
        print(f"[stage-a] NEW BEST {variant} @ step {step}: val_loss {val_loss:.4f}")

    @torch.no_grad()
    def evaluate() -> dict:
        hf_model.eval()
        losses = []
        for _ in range(max(1, val_batches if not synthetic else 2)):
            if synthetic:
                batch = {"input_ids": synthetic_batch(vocab, seq_len, micro_batch, rng_data)}
            else:
                break  # streaming val slice evaluated below
            out = random_recurrence_forward(adapter, scale, batch["input_ids"].to(device), depth=1)
            losses.append(float(_device_loss(out["logits"], batch["input_ids"].to(device))))
        if not synthetic:
            val_gen = build_streaming_loader(
                val_specs, tokenizer, seq_len, micro_batch, seed=seed + 999, split="validation",
                val_batches=val_batches,
            )
            for b in val_gen:
                out = random_recurrence_forward(adapter, scale, b["input_ids"].to(device), depth=1)
                losses.append(float(_device_loss(out["logits"], b["input_ids"].to(device))))
        hf_model.train()
        val_loss = sum(losses) / max(1, len(losses))
        return {"val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20.0))}

    # --- loop ---
    optimizer.zero_grad()
    while step < total_steps:
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        depth = 1 if scale == "base" else sample_depth(depth_rng)
        depth_hist[depth] = depth_hist.get(depth, 0) + 1

        track = bool(per_loop_every and scale == "model" and step % per_loop_every == 0)
        out = random_recurrence_forward(
            adapter, scale, input_ids, depth=depth, track_per_loop=track, use_checkpoint=use_checkpoint
        )
        loss = _device_loss(out["logits"], input_ids) / accum
        loss.backward()
        running_loss += float(loss.detach()) * accum
        total_flops += float(out["flops_est"]) * eff_batch / micro_batch
        tokens_processed += micro_batch * seq_len

        per_loop_str = ""
        if track and out.get("per_loop_logits"):
            pls = [round(float(_device_loss(pl, input_ids)), 4) for pl in out["per_loop_logits"]]
            per_loop_str = f" per_loop={pls}"

        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(cpt.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        step += 1

        if step % log_every == 0:
            now = time.perf_counter()
            dt = max(1e-6, now - t_last_log)
            tps = (tokens_processed - tok_last_log) / dt
            avg_loss = running_loss / log_every
            train_log.log(
                {
                    "step": step, "tokens_processed": tokens_processed,
                    "loss": round(avg_loss, 4), "ppl": round(math.exp(min(avg_loss, 20.0)), 2),
                    "depth": depth, "lr": round(scheduler.get_last_lr()[0], 8),
                    "tokens_per_s": round(tps, 1), "flops_est": f"{out['flops_est']:.3e}",
                    "total_flops": f"{total_flops:.3e}", "gpu_mem_gb": gpu_mem_gb(),
                    "elapsed_s": round(now - t_start, 1),
                }
            )
            print(f"[stage-a:{variant}] step {step}/{total_steps} loss {avg_loss:.4f} depth {depth}{per_loop_str}")
            t_last_log, tok_last_log, running_loss = now, tokens_processed, 0.0

        if step % eval_every == 0 or step == total_steps:
            ev = evaluate()
            eval_log.log(
                {"step": step, "tokens_processed": tokens_processed, "val_loss": round(ev["val_loss"], 4),
                 "val_ppl": round(ev["val_ppl"], 2), "depth": 1, "elapsed_s": round(time.perf_counter() - t_start, 1)}
            )
            print(f"[stage-a:{variant}] eval step {step}: val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.1f}")
            if ev["val_loss"] < best_val_loss:
                best_val_loss = float(ev["val_loss"])
                save_best(best_val_loss)
            if probe_fn is not None:
                try:
                    print(f"[stage-a:{variant}] probe: {probe_fn(adapter, scale)}")
                except Exception as e:
                    print(f"[stage-a:{variant}] probe skipped: {e}")

        if step % ckpt_every == 0 and step < total_steps:
            save_checkpoint()

    save_checkpoint(final=True)
    summary = {
        "variant": variant, "scale": scale, "steps": step, "tokens_processed": tokens_processed,
        "total_flops": total_flops, "depth_hist": depth_hist,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "elapsed_s": round(time.perf_counter() - t_start, 1), "gpu": gpu_name(),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
