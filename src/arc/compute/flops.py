from __future__ import annotations

import torch

"""Analytic FLOP accounting. FLOPs = 2 x MACs; active-expert costs only."""


def jetmoe_layer_macs_per_token(cfg, seq_len: int) -> dict[str, float]:
    h = cfg.hidden_size
    kv = cfg.kv_channels * cfg.num_key_value_heads
    heads_dim = cfg.num_attention_heads * cfg.kv_channels
    k = cfg.num_experts_per_tok
    e = cfg.num_local_experts
    f = cfg.intermediate_size
    return {
        "moa_experts_active": k * 2 * h * kv,
        "kv_proj": h * 2 * kv,
        "attention_scores": k * 2 * heads_dim * seq_len,
        "mlp_gate": h * e,
        "moe_experts_active": k * 3 * h * f,
    }


def jetmoe_layer_flops_per_token(cfg, seq_len: int) -> float:
    macs = jetmoe_layer_macs_per_token(cfg, seq_len)
    return 2.0 * sum(macs.values())


def jetmoe_lm_head_flops_per_token(cfg) -> float:
    return 2.0 * cfg.hidden_size * cfg.vocab_size


def generic_numel_flops(module: torch.nn.Module) -> float:
    return 2.0 * sum(p.numel() for p in module.parameters())
