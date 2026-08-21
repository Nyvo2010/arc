from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class RecurrenceResult:
    logits: Tensor
    final_hidden: Tensor
    state: Any = None


class ARCAdapter(ABC):
    """Exposes native computation boundaries of a pretrained MoE transformer."""

    hf_model: Any
    block_size: int = 1

    @abstractmethod
    def embed(self, input_ids: Tensor) -> Tensor: ...

    @abstractmethod
    def forward_native(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def prepare(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> ForwardContext: ...

    @abstractmethod
    def forward_layer(self, layer_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def forward_block(self, block_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def forward_model(self, hidden: Tensor, ctx: ForwardContext) -> Tensor: ...

    @abstractmethod
    def normalize(self, hidden: Tensor) -> Tensor: ...

    @abstractmethod
    def project_logits(self, normalized_hidden: Tensor) -> Tensor: ...

    @abstractmethod
    def final_logits(self, hidden: Tensor) -> Tensor: ...

    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def num_blocks(self) -> int: ...

    @abstractmethod
    def unit_flops(
        self, scale: str, unit_index: int, seq_len: int, batch_size: int = 1
    ) -> float: ...
