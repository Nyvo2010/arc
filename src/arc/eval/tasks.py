"""Lm-eval-style task definitions for the ARC benchmark harness.

Prompt/format conventions follow the widely used lm-evaluation-harness
formats so results are comparable to published numbers for the substrate
(JetMoE-8B). Every model in a comparison is run through this identical
harness; conventions are recorded in each result row.
"""

from __future__ import annotations

import re
from typing import Callable

from datasets import load_dataset


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _arc_format(question: str, choices: list[str]) -> tuple[str, list[str]]:
    letters = "ABCDEFGHIJ"
    stem = f"{_clean(question)}\n"
    stem += "\n".join(f"{letters[i]}. {_clean(c)}" for i, c in enumerate(choices))
    stem += "\nAnswer:"
    # Continuation is the choice letter (standard lm-eval arc format).
    # Score each letter's probability given the stem.
    return stem, [f"{letters[i]}" for i in range(len(choices))]


def _arca_task(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("ai2_arc", "ARC-Easy", split=split)
        items = []
        for row in ds:
            choices = list(row["choices"]["text"])
            text_choices = list(row["choices"]["label"])
            stem, conts = _arc_format(row["question"], choices)
            label = text_choices.index(row["answerKey"]) if row["answerKey"] in text_choices else 0
            items.append((stem, conts, label))
        return items[:limit] if limit else items

    return load


def _arc_choice_task(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        items = []
        for row in ds:
            choices = list(row["choices"]["text"])
            text_choices = list(row["choices"]["label"])
            stem, conts = _arc_format(row["question"], choices)
            label = text_choices.index(row["answerKey"]) if row["answerKey"] in text_choices else 0
            items.append((stem, conts, label))
        return items[:limit] if limit else items

    return load


def _hellaswag(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
        items = []
        for row in ds:
            ctx = _clean(row["ctx"])
            endings = [_clean(e) for e in row["endings"]]
            items.append((ctx, endings, int(row["label"])))
        return items[:limit] if limit else items

    return load


def _piqa(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("ybisk/piqa", split=split, trust_remote_code=True)
        items = []
        for row in ds:
            ctx = f"Question: {_clean(row['goal'])}\nAnswer:"
            conts = [_clean(row["sol1"]), _clean(row["sol2"])]
            items.append((ctx, conts, int(row["label"])))
        return items[:limit] if limit else items

    return load


def _winogrande(split: str, limit: int | None = None, config: str = "winogrande_xl"):
    def load():
        ds = load_dataset("allenai/winogrande", config, trust_remote_code=True, split=split)
        items = []
        for row in ds:
            sent = _clean(row["sentence"])
            conts = [_clean(row["option1"]), _clean(row["option2"])]
            # standard WinoGrande format: sentence with _ and options appended
            ctx = sent
            label = int(row["answer"]) - 1
            items.append((ctx, conts, label))
        return items[:limit] if limit else items

    return load


def _boolq(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("google/boolq", split=split, trust_remote_code=True)
        items = []
        for row in ds:
            ctx = f"Passage: {_clean(row['passage'])}\nQuestion: {_clean(row['question'])}\nAnswer:"
            conts = ["yes", "no"]
            label = 0 if row["answer"] else 1
            items.append((ctx, conts, label))
        return items[:limit] if limit else items

    return load


def _sciq(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("allenai/sciq", split=split, trust_remote_code=True)
        items = []
        for row in ds:
            ctx = f"Question: {_clean(row['question'])}\nAnswer:"
            choices = [row["correct_answer"]] + [f"{row[f'distractor{i}']}" for i in range(1, 4)]
            conts = [_clean(c) for c in choices]
            label = 0
            items.append((ctx, conts, label))
        return items[:limit] if limit else items

    return load


def _wikitext(split: str, limit: int | None = None):
    def load():
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        # yield text lines, filter empty/wiki markup stripped
        lines = [r["text"] for r in ds]
        if limit:
            lines = lines[:limit]
        return [l for l in lines if l.strip()]

    return load


TASKS: dict[str, dict] = {
    "arc_easy": {
        "loader": lambda limit: _arca_task("test", limit)(),
        "default_limit": 2376,
        "split": "test",
    },
    "arc_challenge": {
        "loader": lambda limit: _arc_choice_task("test", limit)(),
        "default_limit": 1172,
        "split": "test",
    },
    "hellaswag": {
        "loader": lambda limit: _hellaswag("validation", limit)(),
        "default_limit": 10042,
        "split": "validation",
    },
    "piqa": {
        "loader": lambda limit: _piqa("validation", limit)(),
        "default_limit": 1838,
        "split": "validation",
    },
    "winogrande": {
        "loader": lambda limit: _winogrande("validation", limit)(),
        "default_limit": 1267,
        "split": "validation",
    },
    "boolq": {
        "loader": lambda limit: _boolq("validation", limit)(),
        "default_limit": 3270,
        "split": "validation",
    },
    "sciq": {
        "loader": lambda limit: _sciq("test", limit)(),
        "default_limit": 500,
        "split": "test",
    },
    "wikitext": {
        "loader": lambda limit: _wikitext("validation", limit)(),
        "default_limit": None,
        "kind": "ppl",
        "split": "validation",
    },
}