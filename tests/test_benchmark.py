from __future__ import annotations

import torch

from arc.evaluation.benchmarks import evaluate_single_token, generate_arithmetic


class FixedArgmaxModel(torch.nn.Module):
    def __init__(self, vocab_size: int, token_id: int):
        super().__init__()
        self.token_id = token_id
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        logits[0, -1, self.token_id] = 10.0
        return type("Out", (), {"logits": logits})()


def test_single_token_scoring(tokenizer):
    problems = generate_arithmetic(4, seed=5)
    target = problems[0].answer
    assert len(target) == 1 and target.isdigit()
    pred_id = tokenizer.encode(target, add_special_tokens=False)[-1]

    good = FixedArgmaxModel(len(tokenizer), pred_id)
    result = evaluate_single_token(good, tokenizer, problems[:1])
    assert result["rows"][0]["predicted"] == target
    assert result["accuracy"] == 1.0

    wrong = FixedArgmaxModel(len(tokenizer), (pred_id + 1) % len(tokenizer))
    result = evaluate_single_token(wrong, tokenizer, problems[:1])
    assert result["accuracy"] == 0.0


def test_by_op_breakdown(tokenizer):
    problems = generate_arithmetic(40, seed=11)
    model = FixedArgmaxModel(len(tokenizer), 0)
    result = evaluate_single_token(model, tokenizer, problems)
    assert set(result["by_op"].keys()) == {"+", "-", "*", "/"}
    total = sum(s["n"] for s in result["by_op"].values())
    assert total == 40
