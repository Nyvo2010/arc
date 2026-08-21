from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor


class RouterRecorder:
    """Forward hooks on all TopKGating routers; records raw router logits per call."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @classmethod
    def attach(cls, net: torch.nn.Module) -> "RouterRecorder":
        rec = cls()
        for name, module in net.named_modules():
            if type(module).__name__ == "JetMoeTopKGating":
                rec._handles.append(module.register_forward_hook(rec._make_hook(name)))
        return rec

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            router_logits = output[-1]
            self.records.append({"module": name, "logits": router_logits.detach().float().cpu()})

        return hook

    def reset(self) -> None:
        self.records.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def summarize_router_records(records: list[dict], top_k: int) -> dict:
    counts: Counter[int] = Counter()
    gate_probs: list[float] = []
    per_call: list[dict] = []
    for rec in records:
        logits = rec["logits"]
        top_logits, top_idx = logits.topk(top_k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        gate_probs.append(probs.mean().item())
        experts_flat = top_idx.flatten().tolist()
        counts.update(experts_flat)
        per_call.append(
            {
                "module": rec["module"],
                "tokens": int(logits.shape[0]),
                "selected_counts": dict(Counter(experts_flat)),
            }
        )
    if not counts:
        return {"num_calls": 0}
    num_experts = max(counts) + 1
    total = sum(counts.values())
    utilization = [counts.get(e, 0) / total for e in range(num_experts)]
    frac = torch.tensor(utilization)
    load_entropy = float(-(frac * (frac + 1e-12).log()).sum())
    return {
        "num_calls": len(records),
        "num_experts": num_experts,
        "expert_counts": {str(k): v for k, v in sorted(counts.items())},
        "utilization": [round(u, 6) for u in utilization],
        "load_balance_entropy": round(load_entropy, 6),
        "mean_gate_prob": round(sum(gate_probs) / len(gate_probs), 6),
        "per_call": per_call,
    }
