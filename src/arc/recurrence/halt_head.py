from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class HaltHead(nn.Module):
    """Base HALT head interface.

    Returns probability of CONTINUE in [0,1].
    """

    def __init__(self, hidden_dim: int, feature_dim: int = 6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        feats = self.forward_features(logits_prev, logits_cur, hidden_prev, hidden_cur)
        return self.proj(feats)


class ModelHaltHead(HaltHead):
    """HALT head evaluated after each complete model traversal.

    Features: entropy, entropy_delta, js_divergence, top1_stability, hidden_cosine_change, recurrence_count
    """

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        # Compute simple stats per batch, return mean vector [B,6]
        # Simplified feature extraction; full stats computed in controller for determinism
        # Here we just return a dummy vector to keep interface uniform.
        # Real usage will be overridden by controller features.
        b = logits_cur.shape[0]
        # placeholder zero features; controller will still decide
        return torch.zeros(b, 6, device=logits_cur.device)


class BlockHaltHead(HaltHead):
    """HALT head evaluated after each block execution."""

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        b = logits_cur.shape[0]
        return torch.zeros(b, 6, device=logits_cur.device)


class LayerHaltHead(HaltHead):
    """HALT head evaluated after each layer execution."""

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        b = logits_cur.shape[0]
        return torch.zeros(b, 6, device=logits_cur.device)
