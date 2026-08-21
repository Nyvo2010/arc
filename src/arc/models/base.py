from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from arc.recurrence.state import RecurrenceState


@dataclass
class ForwardContext:
    position_ids: Tensor | None = None
    attention_mask: Tensor | None = None


@dataclass
class TraceEvent:
    scale: str
    unit_index: int
    iteration: int
    hidden_delta_cos: float
    est_flops: float
    latency_s: float


@dataclass
class RecurrenceResult:
    logits: Tensor
    final_hidden: Tensor
    last_logits_history: list[Tensor] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    state: Any = None
    router_summary: dict | None = None
    truncated: bool = False


class ARCAdapter(ABC):
    """Exposes native computation boundaries of a pretrained MoE transformer."""

    hf_model: Any
    block_size: int = 1

    @abstractmethod
    def embed(self, input_ids: Tensor) -> Tensor: ...

    @abstractmethod
    def prepare(self, hidden: Tensor) -> ForwardContext: ...

    @abstractmethod
    def forward_layer(self, layer_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def forward_block(self, block_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def forward_model(self, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def final_logits(self, hidden: Tensor) -> Tensor: ...

    @abstractmethod
    def last_token_logits(self, hidden: Tensor) -> Tensor: ...

    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def num_blocks(self) -> int: ...

    @abstractmethod
    def begin_step(self) -> None: ...

    @abstractmethod
    def get_router_records(self) -> list[dict]: ...

    @abstractmethod
    def unit_flops(self, scale: str, unit_index: int, seq_len: int) -> float: ...
