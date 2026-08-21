from __future__ import annotations

import time
import uuid
from pathlib import Path

import torch

from arc.evaluation.benchmarks import evaluate_single_token, generate_arithmetic
from arc.evaluation.logging import append_jsonl, experiment_meta
from arc.evaluation.metrics import trajectory_metrics
from arc.models.registry import create_adapter
from arc.recurrence.base import RecurrentLM, build_recurrent_model


def resolve_device(preference: str | None = None) -> str:
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_cfg: dict, tiny: bool = False, vocab_size: int | None = None):
    if tiny:
        return TinyTokenizerShim(vocab_size or 128)
    from transformers import AutoTokenizer

    path = model_cfg.get("tokenizer_path") or model_cfg.get("path")
    return AutoTokenizer.from_pretrained(path)


class TinyTokenizerShim:
    """Deterministic pseudo-tokenization for smoke tests with the random-init tiny model."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, prompt: str, return_tensors: str | None = None):
        gen = torch.Generator().manual_seed(abs(hash(prompt)) % (2**31))

        class _Batch:
            pass

        batch = _Batch()
        batch.input_ids = torch.randint(2, self.vocab_size, (1, 8), generator=gen)
        return batch

    def decode(self, ids, **kwargs) -> str:
        return str(int(ids[0]) % 10)

    def __len__(self) -> int:
        return self.vocab_size


def build_model(cfg: dict, scale: str, loops, tiny: bool = False):
    model_cfg = cfg["model"]
    device_pref = cfg.get("run", {}).get("device") or ("cpu" if tiny else None)
    dtype = "float32" if tiny else model_cfg.get("dtype")
    adapter = create_adapter(
        "tiny" if tiny else model_cfg["path"],
        block_size=model_cfg.get("block_size", 4),
        dtype=dtype,
        device=device_pref,
    )
    model = build_recurrent_model(scale, adapter, loops, block_size=model_cfg.get("block_size", 4))
    return model


def run_scale(
    cfg: dict,
    scale: str,
    name: str,
    loops,
    out_dir: str | Path,
    tiny: bool = False,
    limit: int | None = None,
) -> dict:
    bench_cfg = cfg.get("benchmark", {})
    n_problems = min(bench_cfg.get("n_problems", 200), limit or 10**9)
    problems = generate_arithmetic(
        n_problems,
        seed=bench_cfg.get("seed", 123),
        ranges={k: tuple(v) for k, v in (bench_cfg.get("ranges") or {}).items()} or None,
    )

    model: RecurrentLM = build_model(cfg, scale, loops, tiny=tiny)
    device = next(model.parameters()).device
    tokenizer = load_tokenizer(cfg["model"], tiny, vocab_size=model.adapter.cfg.vocab_size)

    experiment_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{scale}-{name}-{uuid.uuid4().hex[:6]}"
    meta = experiment_meta(cfg, extra={"experiment_id": experiment_id, "scale": scale, "name": name, "loops": loops})
    raw_path = Path(out_dir) / f"{experiment_id}.jsonl"
    append_jsonl(raw_path, [{"type": "meta", **meta}])

    eval_result = evaluate_single_token(model, tokenizer, problems, device=str(device))

    probe_ids = tokenizer(problems[0].prompt, return_tensors="pt").input_ids.to(device)
    probe = model(probe_ids)
    total_est_flops = probe.state.compute_used + model.lm_head_flops_per_token() * probe_ids.shape[1]
    latencies = [e.latency_s for e in probe.trace]

    summary = {
        "type": "summary",
        "experiment_id": experiment_id,
        "scale": scale,
        "name": name,
        "loops": loops,
        "accuracy": eval_result["accuracy"],
        "by_op": eval_result["by_op"],
        "executions": probe.state.executions,
        "est_flops_forward": round(total_est_flops, 1),
        "avg_exec_latency_s": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "router_num_calls": (probe.router_summary or {}).get("num_calls"),
        "load_balance_entropy": (probe.router_summary or {}).get("load_balance_entropy"),
    }
    append_jsonl(raw_path, [summary])
    for r in eval_result["rows"]:
        r.update({"type": "problem", "experiment_id": experiment_id, "scale": scale, "name": name})
    append_jsonl(raw_path, eval_result["rows"])
    return summary


def trajectory_probe(cfg: dict, scale: str, loops, problem_prompt: str, tiny: bool = False) -> dict:
    """Per-execution progress signals on a single prompt."""
    model: RecurrentLM = build_model(cfg, scale, loops, tiny=tiny)
    tokenizer = load_tokenizer(cfg["model"], tiny, vocab_size=model.adapter.cfg.vocab_size)
    ids = tokenizer(problem_prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    result = model(ids)
    return {
        "trajectory": trajectory_metrics(result.last_logits_history),
        "trace": [
            {
                "unit": e.unit_index,
                "iter": e.iteration,
                "hidden_delta_cos": e.hidden_delta_cos,
                "latency_s": round(e.latency_s, 6),
            }
            for e in result.trace
        ],
        "state": {
            "executions": result.state.executions,
            "compute_used_est_flops": result.state.compute_used,
            "truncated": result.truncated,
        },
        "router": {k: v for k, v in (result.router_summary or {}).items() if k != "per_call"},
        "predicted": tokenizer.decode([int(result.logits[0, -1].argmax())]).strip(),
    }
