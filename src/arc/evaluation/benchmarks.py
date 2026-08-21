from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass
class Problem:
    op: str
    a: int
    b: int
    prompt: str
    answer: str


OP_SYMBOLS = {"+": "+", "-": "-", "*": "*", "/": "/"}


def generate_arithmetic(
    n: int,
    seed: int = 0,
    ranges: dict[str, tuple[int, int]] | None = None,
    ops: tuple[str, ...] = ("+", "-", "*", "/"),
) -> list[Problem]:
    """Single-TOKEN-answer arithmetic: the JetMoE tokenizer splits digits, so
    every correct answer must be one digit (0-9). Prompts may still contain any
    operands; only answers are constrained. This removes chain-of-thought:
    all reasoning must happen inside the forward pass(es)."""
    rng = random.Random(seed)
    ranges = ranges or {"+": (0, 9), "-": (0, 9), "*": (0, 3), "/": (1, 9)}
    problems: list[Problem] = []
    for _ in range(n):
        op = rng.choice(ops)
        lo, hi = ranges.get(op, (0, 9))
        if op == "+":
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            while a + b > 9:
                a = rng.randint(lo, hi)
                b = rng.randint(lo, hi)
        elif op == "-":
            a = rng.randint(lo, hi)
            b = rng.randint(lo, a)
        elif op == "*":
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
        else:
            b = rng.randint(max(1, lo), hi)
            q = rng.randint(0, 9)
            a = q * b
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}[op]
        assert 0 <= result <= 9, f"answer {result} not a single token"
        sym = OP_SYMBOLS[op]
        problems.append(Problem(op=op, a=a, b=b, prompt=f"{a} {sym} {b} =", answer=str(result)))
    return problems


@torch.no_grad()
def evaluate_single_token(model, tokenizer, problems: list[Problem], device: str | None = None) -> dict:
    """Greedy single-token answers: no CoT possible, all reasoning is latent."""
    rows: list[dict] = []
    by_op: dict[str, dict] = {}
    for idx, p in enumerate(problems):
        ids = tokenizer(p.prompt, return_tensors="pt").input_ids
        if device:
            ids = ids.to(device)
        logits = model(ids).logits
        pred_id = int(logits[0, -1].argmax())
        pred_str = tokenizer.decode([pred_id]).strip()
        correct = pred_str == p.answer
        rows.append(
            {
                "idx": idx,
                "prompt": p.prompt,
                "target": p.answer,
                "predicted": pred_str,
                "correct": correct,
            }
        )
        stats = by_op.setdefault(p.op, {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += int(correct)
    accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    return {
        "n": len(rows),
        "accuracy": round(accuracy, 6),
        "by_op": {op: {"n": s["n"], "accuracy": round(s["correct"] / s["n"], 6)} for op, s in sorted(by_op.items())},
        "rows": rows,
    }


def problem_to_row(p: Problem) -> dict:
    return asdict(p)


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    return tokenizer.encode(" " + answer, add_special_tokens=False)
