"""Trainable random-recurrence forward for Stage A (PLAN Stage A).

No halting machinery: a depth ``d`` is sampled per forward from ``DEPTH_DIST``
(``{1..4}``, mass biased toward 1-2) and the SAME depth is applied to every
unit at the chosen scale. The ``base`` scale is the equal-budget control:
a single native pass with gradients enabled.

Normalization mirrors ``arc.recurrence.adaptive`` (normalize between model
loops, final normalize + project) so train-time and eval-time representations
match. Unlike inference, nothing here runs under ``torch.no_grad``.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from arc.models.base import ARCAdapter

# PLAN Stage A: depth-randomized loops, mass biased toward 1-2.
DEPTH_DIST: dict[int, float] = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}


def sample_depth(rng: random.Random, dist: dict[int, float] | None = None) -> int:
    """Sample a recurrence depth from ``dist`` (defaults to ``DEPTH_DIST``)."""
    dist = dist or DEPTH_DIST
    depths = sorted(dist)
    weights = [float(dist[d]) for d in depths]
    return int(rng.choices(depths, weights=weights, k=1)[0])


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Shifted next-token cross-entropy. ``labels`` == ``input_ids``."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )


def _maybe_checkpoint(fn, hidden: Tensor, use_checkpoint: bool) -> Tensor:
    if not use_checkpoint or not torch.is_grad_enabled():
        return fn(hidden)
    return torch.utils.checkpoint.checkpoint(fn, hidden, use_reentrant=False)


def random_recurrence_forward(
    adapter: ARCAdapter,
    scale: str,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    depth: int | None = None,
    rng: random.Random | None = None,
    track_per_loop: bool = False,
    use_checkpoint: bool = False,
) -> dict:
    """Trainable forward with fixed random depth.

    Returns dict with ``logits``, ``final_hidden``, ``depth``,
    ``executions`` (total unit executions), ``flops_est`` (analytic upper
    bound via ``adapter.unit_flops``), and optionally ``per_loop_logits``
    (model scale only, for the mechanistic per-loop-position loss figure).
    """
    if scale not in ("base", "model", "block", "layer"):
        raise ValueError(f"unknown scale: {scale}")

    if depth is None:
        depth = 1 if scale == "base" else sample_depth(rng or random.Random())

    hidden = adapter.embed(input_ids)
    ctx = adapter.prepare(hidden, attention_mask=attention_mask, position_ids=position_ids)
    seq_len = hidden.shape[1]
    batch_size = hidden.shape[0]

    executions = 0
    flops_est = 0.0
    per_loop_logits: list[Tensor] | None = [] if (track_per_loop and scale == "model") else None

    if scale == "base":
        hidden = _maybe_checkpoint(lambda h: adapter.forward_model(h, ctx), hidden, use_checkpoint)
        executions = 1
        flops_est = float(adapter.unit_flops("model", 0, seq_len, batch_size=batch_size))
    elif scale == "model":
        for _ in range(depth):
            hidden = _maybe_checkpoint(lambda h: adapter.forward_model(h, ctx), hidden, use_checkpoint)
            hidden = adapter.normalize(hidden)
            executions += 1
            flops_est += float(adapter.unit_flops("model", 0, seq_len, batch_size=batch_size))
            if per_loop_logits is not None:
                per_loop_logits.append(adapter.project_logits(hidden))
    elif scale == "block":
        for unit in range(adapter.num_blocks()):
            for _ in range(depth):
                hidden = _maybe_checkpoint(
                    lambda h, u=unit: adapter.forward_block(u, h, ctx), hidden, use_checkpoint
                )
                executions += 1
                flops_est += float(adapter.unit_flops("block", unit, seq_len, batch_size=batch_size))
    else:  # layer
        for unit in range(adapter.num_layers()):
            for _ in range(depth):
                hidden = _maybe_checkpoint(
                    lambda h, u=unit: adapter.forward_layer(u, h, ctx), hidden, use_checkpoint
                )
                executions += 1
                flops_est += float(adapter.unit_flops("layer", unit, seq_len, batch_size=batch_size))

    final_hidden = adapter.normalize(hidden)
    logits = adapter.project_logits(final_hidden)
    flops_est += float(adapter.lm_head_flops_per_token()) * seq_len * batch_size

    return {
        "logits": logits,
        "final_hidden": final_hidden,
        "depth": depth,
        "executions": executions,
        "flops_est": flops_est,
        "per_loop_logits": per_loop_logits,
    }


class RandomRecurrenceLM(nn.Module):
    """``nn.Module`` wrapper so PEFT/optimizers see a model object.

    Holds no params of its own; gradients flow into the adapter's HF model
    (LoRA adapters after ``get_peft_model`` mutates it in place).
    """

    def __init__(self, adapter: ARCAdapter, scale: str):
        super().__init__()
        self.adapter = adapter
        self.scale = scale
        self.transformer = adapter.hf_model

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> dict:
        return random_recurrence_forward(
            self.adapter,
            self.scale,
            input_ids,
            attention_mask=attention_mask,
            depth=kwargs.get("depth"),
            rng=kwargs.get("rng"),
            track_per_loop=kwargs.get("track_per_loop", False),
            use_checkpoint=kwargs.get("use_checkpoint", False),
        )
