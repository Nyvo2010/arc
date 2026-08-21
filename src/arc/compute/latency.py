from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def timer(log: list[float] | None = None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if log is not None:
        log.append(elapsed)
