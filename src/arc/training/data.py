"""Streaming data pipeline for Stage-A CPT (PLAN: wikitext-103 + C4 slice).

Kaggle path uses HF ``datasets`` streaming + packing into fixed ``seq_len``
chunks. Everything HF-dependent is imported lazily so unit tests and
offline smoke runs work without the dependency (synthetic fallback).
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator

import torch


def pack_token_ids(token_ids: list[int], seq_len: int) -> list[list[int]]:
    """Greedily chunk a flat id stream into full ``seq_len`` sequences."""
    packed = []
    for i in range(0, len(token_ids) - seq_len + 1, seq_len):
        packed.append(token_ids[i : i + seq_len])
    return packed


def synthetic_batch(vocab_size: int, seq_len: int, batch_size: int, rng: random.Random | None = None):
    """Deterministic random batch for tests / offline smoke runs."""
    rng = rng or random.Random(0)
    ids = [[rng.randrange(2, vocab_size) for _ in range(seq_len)] for _ in range(batch_size)]
    return torch.tensor(ids, dtype=torch.long)


def cycling(loader_iter_fn, seed: int = 0) -> Iterator:
    """Yield batches forever, rebuilding the iterator when exhausted."""
    while True:
        for batch in loader_iter_fn():
            yield batch


def build_streaming_loader(
    dataset_specs: list[dict],
    tokenizer,
    seq_len: int,
    micro_batch: int,
    seed: int = 0,
    split: str = "train",
    val_batches: int = 0,
):
    """Yield ``(input_ids,)`` packed batches from streaming HF datasets.

    ``dataset_specs``: list of ``{name, weight}``; text is taken from the
    first text-like column found (``text``/``content``). Batches are packed
    from a shared token buffer so every batch is a full ``seq_len`` chunk.
    With ``split != 'train'`` yields at most ``val_batches`` fixed batches
    (stable validation slice).
    """
    try:
        from datasets import interleave_datasets, load_dataset
    except ImportError as e:
        raise RuntimeError("HF 'datasets' package required for streaming (pip install datasets)") from e

    streams = []
    weights = []
    for spec in dataset_specs:
        ds = load_dataset(
            spec["name"], spec.get("config"), split=spec.get("split", split),
            streaming=True, trust_remote_code=True,
        )
        streams.append(ds)
        weights.append(float(spec.get("weight", 1.0)))

    stream = interleave_datasets(streams, probabilities=None, seed=seed) if len(streams) > 1 else streams[0]
    if len(streams) > 1:
        # Weighted interleave via sampling probabilities.
        total = sum(weights)
        stream = interleave_datasets(
            streams, probabilities=[w / total for w in weights], seed=seed, stopping_strategy="all_exhausted"
        )

    # Resolve the text column per-example: streaming datasets expose columns
    # lazily, so prefer 'text' then 'content' on each example.
    buffer: list[int] = []
    yielded = 0

    def gen():
        nonlocal buffer, yielded
        for ex in stream:
            text = ex.get("text", ex.get("content", ""))
            if not text or not text.strip():
                continue
            buffer.extend(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])
            while len(buffer) >= seq_len * micro_batch:
                chunk, buffer = buffer[: seq_len * micro_batch], buffer[seq_len * micro_batch :]
                batch = torch.tensor(chunk, dtype=torch.long).view(micro_batch, seq_len)
                yielded += 1
                yield {"input_ids": batch}
                if split != "train" and yielded >= val_batches:
                    return

    return gen()
