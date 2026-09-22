"""Benchmark harness for Arc models.

Loads a model (base 4-bit JetMoE, optionally with a trained LoRA adapter) and
scores lm-eval-style tasks: multiple-choice via next-token log-likelihood and
wikitext via chunked perplexity. Every model gets the identical harness.

Scoring: for each item, ``ctx`` is tokenized once and each ``continuation`` is
tokenized separately (add_special_tokens=False) and concatenated. Log-probs of
the continuation span are summed (``acc``) and per-token averaged (``acc_norm``
— the lm-eval convention when both are reported).
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from arc.eval.tasks import TASKS
from arc.models.registry import MODEL_VARIANTS, create_adapter
from arc.training.random_recurrence import random_recurrence_forward


class _ModelState:
    """Accumulates compute/loop/time stats across scored items."""

    def __init__(self) -> None:
        self.items: int = 0
        self.executions: int = 0
        self.tokens: int = 0
        self.compute_used: float = 0.0
        self.decide_probs: float = 0.0
        self.decide_calls: int = 0
        self.wall_s: float = 0.0
        self.forward_calls: int = 0

    def add(self, state: Any, n_items: int = 1, tokens: int = 0) -> None:
        self.items += n_items
        self.executions += int(getattr(state, "executions", 0))
        self.tokens += tokens
        self.compute_used += float(getattr(state, "compute_used", 0.0))
        self.forward_calls += 1
        probs = getattr(state, "decide_probs", {})
        self.decide_probs += sum(probs.values()) if probs else 0.0
        self.decide_calls += len(probs) if probs else 0

    def add_time(self, sec: float) -> None:
        self.wall_s += max(0.0, sec)

    def stats(self) -> dict:
        n = max(1, self.items)
        wall = max(self.wall_s, 1e-6)
        return {
            "avg_loops_per_item": round(self.executions / n, 3),
            "avg_tokens_per_item": round(self.tokens / n, 1),
            "total_executions": self.executions,
            "avg_flops_per_item": round(self.compute_used / n, 1),
            "total_flops": round(self.compute_used, 1),
            "avg_decide_mean_p": round(self.decide_probs / max(1, self.decide_calls), 4),
            "elapsed_s": round(self.wall_s, 1),
            "items_per_s": round(self.items / wall, 2),
            "tokens_per_s": round(self.tokens / wall, 1),
            "flops_per_s": round(self.compute_used / wall, 1),
            "decisions_per_s": round(self.decide_calls / wall, 1),
        }


class EvalModel:
    """Thin wrapper: base or adapter-equipped model evaluated at fixed depth.

    depth=1 => single native pass (matches the base model's compute); depth>1
    uses the ARC recurrence forward (only valid for adaptive scales).
    """

    def __init__(self, key: str, base_path: str, adapter_dir: str | None = None,
                 depth: int = 1, device_map: str = "auto", budgeted: bool = False,
                 max_loops: int = 4):
        if key not in MODEL_VARIANTS:
            raise ValueError(f"unknown model key {key}")
        scale = MODEL_VARIANTS[key]["scale"]
        self.key = key
        self.scale = scale
        self.depth = int(depth)
        self.budgeted = budgeted
        self.max_loops = max_loops
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if key == "base" or not MODEL_VARIANTS[key].get("adaptive"):
            depth = 1  # base model ignores depth
        self.eff_depth = depth

        self.adapter = create_adapter(base_path, device_map=device_map)
        # Inject trained LoRA adapter weights into the (in-place) base model.
        # Capture net/head from the raw JetMoeForCausalLM BEFORE wrapping in
        # PeftModel (PeftModel.model proxies to the base CausalLM, not its
        # transformer, so embed_tokens would be unreachable afterwards).
        if adapter_dir:
            from peft import PeftModel

            net = self.adapter.net
            head = self.adapter.head
            self.adapter.hf_model = PeftModel.from_pretrained(self.adapter.hf_model, adapter_dir)
            self.adapter.net = net
            self.adapter.head = head
            self.adapter.hf_model.eval()

        # Real adaptive path (Policy-T): build the ARC controller LM.
        if self.budgeted:
            from arc.recurrence.builder import build_model

            self.arc_model = build_model(scale, self.adapter, max_loops=max_loops)
        self._last_state: Any = None
        self._chunk_states: list = []
        self._last_tokens: int = 0
        self._chunk_tokens: int = 0

    def _forward_adaptive(self, ids: Tensor):
        """Return (logits, state) through the controller path (budgeted)."""
        res = self.arc_model(ids)
        st = res.state
        # tokens processed = executions(loops total) x batch x seq_len
        self._last_tokens = int(getattr(st, "executions", 0)) * int(ids.shape[0]) * int(ids.shape[1])
        return res.logits, st

    @torch.no_grad()
    def forward_logits(self, input_ids: Tensor):
        ids = input_ids.to(self.device)
        if self.budgeted:
            logits, state = self._forward_adaptive(ids)
            self._last_state = state
            return logits
        if self.eff_depth == 1 or self.scale == "base":
            h, logits = self.adapter.forward_native(ids)
            return logits
        out = random_recurrence_forward(self.adapter, self.scale, ids, depth=self.eff_depth)
        return out["logits"]

    @torch.no_grad()
    def score_choices(self, ctx_ids: list[Tensor], cont_ids: list[Tensor]) -> list[float]:
        """Return per-choice summed loglikelihood of the continuation span."""
        # pad to the longest ctx+cont in the choice set, forward once
        max_ctx = max(int(len(c)) for c in ctx_ids) if ctx_ids else 0
        max_cont = max(int(len(c)) for c in cont_ids) if cont_ids else 0
        pad = self.adapter.hf_model.config.pad_token_id if getattr(
            self.adapter.hf_model.config, "pad_token_id", None) is not None else 0
        if pad is None:
            pad = 0
        seq_len = max_ctx + max_cont
        B = len(cont_ids)
        if B == 0:
            return [0.0] * len(cont_ids) if cont_ids else []
        ids = torch.full((B, seq_len), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((B, seq_len), dtype=torch.bool, device=self.device)
        starts = []
        for i, (c, ct) in enumerate(zip(ctx_ids, cont_ids)):
            ln = min(len(c), max_ctx)
            ids[i, :ln] = c[:ln].to(self.device)
            s = min(ln, seq_len)
            e = min(s + len(ct), seq_len)
            ids[i, s:e] = ct[: e - s].to(self.device)
            mask[i, s:e] = True
        logits = self.forward_logits(ids)
        if logits.device != self.device:
            logits = logits.to(self.device)
        # next-token logprob of the continuation span
        logp = nn.functional.log_softmax(logits.float(), dim=-1)

        shift_labels = ids[:, 1:]
        shift_logp = logp[:, :-1]
        span = mask[:, 1:]
        scores = []
        for i in range(B):
            sel = span[i]
            nz = sel.nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                scores.append(float("-inf"))
                continue
            toks = shift_labels[i][nz]
            lg = shift_logp[i][nz].gather(-1, toks.unsqueeze(-1)).squeeze(-1)
            scores.append(float(lg.sum()))
        return scores

    @torch.no_grad()
    def wikitext_nll(self, chunks: list[Tensor]) -> tuple[float, int]:
        """Mean NLL over the given token chunks. Accumulates budgeted state per chunk."""
        tot = 0.0
        cnt = 0
        for ids in chunks:
            logits = self.forward_logits(ids)
            if self.budgeted:
                self._chunk_states.append(self._last_state)
                self._chunk_tokens += self._last_tokens
            if logits.device != self.device:
                logits = logits.to(self.device)
            logp = nn.functional.log_softmax(logits.float(), dim=-1)
            tgt = ids[:, 1:].to(logits.device)
            lg = logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot += float(-lg.sum())
            cnt += max(1, int(tgt.numel()))
        return tot / max(1, cnt), cnt


def _tokenize(tokenizer, text: str) -> Tensor:
    out = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    if isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out.input_ids
    return torch.tensor(ids[0].tolist() if ids.ndim else [ids], dtype=torch.long)


class _TinyTokenizer:
    """Char-level tokenizer for the tiny test model (vocab 128). Only for
    local smoke tests; Kaggle runs use the real JetMoE tokenizer."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = None
        self.vocab_size = 128

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [ord(c) % 128 for c in str(text)]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _get_tokenizer(base_path: str):
    if base_path == "tiny":
        return _TinyTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=True)


def chunk_text(tokenizer, text: str, max_len: int = 512) -> list[Tensor]:
    """Tokenize text and split into <=max_len non-overlapping [1, n] chunks."""
    ids = _tokenize(tokenizer, text)
    n = int(len(ids))
    chunks = []
    for s in range(0, n, max_len):
        e = min(s + max_len, n)
        if e - s > 0:
            chunks.append(ids[s:e].unsqueeze(0))
    return chunks


def evaluate_model(
    key: str,
    base_path: str,
    adapter_dir: str | None,
    tasks: list[str],
    limits: dict[str, int | None] | None = None,
    limits_prefix: str = "",
    max_len: int = 512,
    batch_choices: bool = True,
    depth: int = 1,
    device_map: str = "auto",
    budgeted: bool = False,
    max_loops: int = 4,
) -> dict[str, dict]:
    """Run the task suite for one model. Returns {task: metrics}."""
    tokenizer = _get_tokenizer(base_path)
    if base_path != "tiny" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EvalModel(key=key, base_path=base_path, adapter_dir=adapter_dir,
                      depth=depth, device_map=device_map, budgeted=budgeted,
                      max_loops=max_loops)
    out = {}
    for task in tasks:
        mstate = _ModelState()
        spec = TASKS[task]
        limit = (limits or {}).get(task, spec.get("default_limit"))
        if limits_prefix:
            lk = f"{limits_prefix}:{task}"
            limit = (limits or {}).get(lk, limit)
        t0 = time.perf_counter()
        items = spec["loader"](limit)
        if spec.get("kind") == "ppl":
            chunks = []
            for line in items:
                chunks.extend(chunk_text(tokenizer, line, max_len))
                if limit and len(chunks) >= limit:
                    chunks = chunks[:limit]
                    break
            tot_nll, cnt = model.wikitext_nll(chunks)
            out[task] = {
                "task": task, "n_items": len(items), "n_tokens": cnt,
                "ppl": math.exp(tot_nll) if tot_nll < 20 else float("inf"),
                "mean_nll": round(tot_nll, 4), "elapsed_s": round(time.perf_counter() - t0, 1),
            }
            if budgeted:
                for st in model._chunk_states:
                    mstate.add(st, tokens=model._chunk_tokens)
                mstate.add_time(out[task]["elapsed_s"])
                out[task].update(mstate.stats())
            model._chunk_states = []
            model._chunk_tokens = 0
        else:
            n_correct, n_correct_norm, n_tot = 0, 0, 0
            for stem, conts, label in items:
                ctx_ids = _tokenize(tokenizer, stem)
                cont_ids = [_tokenize(tokenizer, c) for c in conts]
                scores = model.score_choices([ctx_ids] * len(cont_ids), cont_ids)
                if budgeted and getattr(model, "_last_state", None) is not None:
                    mstate.add(model._last_state, tokens=model._last_tokens)
                if batch_choices:
                    pass
                pred = max(range(len(scores)), key=lambda i: scores[i] if scores[i] != float("-inf") else float("-inf"))
                if pred == label:
                    n_correct += 1
                # acc_norm: mean per-token loglikelihood argmax
                ms = []
                for i, s in enumerate(scores):
                    n = max(1, int(len(cont_ids[i])))
                    ms.append(s / n if s != float("-inf") else float("-inf"))
                predn = max(range(len(ms)), key=lambda i: ms[i])
                if predn == label:
                    n_correct_norm += 1
                n_tot += 1
            out[task] = {
                "task": task, "n_correct": n_correct, "n_correct_norm": n_correct_norm,
                "n_tot": n_tot,
                "acc": round(n_correct / max(1, n_tot), 4),
                "acc_norm": round(n_correct_norm / max(1, n_tot), 4),
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
            if budgeted:
                mstate.add_time(out[task]["elapsed_s"])
                out[task].update(mstate.stats())
    _free_gpu(model)
    return out


def _free_gpu(model) -> None:
    """Release a model's GPU memory so the next variant can load on a single GPU."""
    import gc

    del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_results_csv(results: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_key, tasks in results.items():
        for task, m in tasks.items():
            row = dict(m)
            row["model"] = model_key
            row["task"] = task
            row.update(meta.get(model_key, {}))
            rows.append(row)
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval] wrote {len(rows)} result rows -> {path}")


def load_results_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows