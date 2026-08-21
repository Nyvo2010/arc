from __future__ import annotations

import torch


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_p.float(), dim=-1)
    log_q = torch.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


def trajectory_metrics(last_logits_history: list[torch.Tensor]) -> list[dict]:
    """Per-step progress signals from the last-position logits of each execution."""
    out = []
    for t in range(1, len(last_logits_history)):
        prev, cur = last_logits_history[t - 1], last_logits_history[t]
        out.append(
            {
                "step": t,
                "entropy": round(float(entropy(cur).mean()), 6),
                "kl_to_prev": round(float(kl_divergence(cur, prev).mean()), 6),
                "top1_stable": bool(torch.equal(prev.argmax(-1), cur.argmax(-1))),
            }
        )
    return out
