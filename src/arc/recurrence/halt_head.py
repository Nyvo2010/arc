from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
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
        # logits_cur: [B, T, V]
        B, T, V = logits_cur.shape
        device = logits_cur.device

        # entropy of current distribution
        cur_probs = F.softmax(logits_cur, dim=-1)  # [B, T, V]
        cur_logprobs = F.log_softmax(logits_cur, dim=-1)
        cur_entropy = -(cur_probs * cur_logprobs).sum(dim=-1)  # [B, T]
        entropy = cur_entropy.mean(dim=1)  # [B]

        # entropy delta and JS divergence
        if logits_prev is None:
            entropy_delta = torch.zeros_like(entropy)
            js_div = torch.zeros_like(entropy)
            top1_stab = torch.ones_like(entropy)  # no change if no previous
            hidden_cos_change = torch.zeros_like(entropy)
        else:
            # entropy previous
            prev_probs = F.softmax(logits_prev, dim=-1)
            prev_logprobs = F.log_softmax(logits_prev, dim=-1)
            prev_entropy = -(prev_probs * prev_logprobs).sum(dim=-1).mean(dim=1)  # [B]
            entropy_delta = prev_entropy - entropy  # [B]

            # JS divergence: 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5*(P+Q)
            m = 0.5 * (prev_probs + cur_probs)  # [B, T, V]
            # avoid log(0)
            eps = 1e-10
            log_m = torch.log(m + eps)
            kl_pm = (prev_probs * (torch.log(prev_probs + eps) - log_m)).sum(dim=-1)  # [B, T]
            kl_qm = (cur_probs * (torch.log(cur_probs + eps) - log_m)).sum(dim=-1)  # [B, T]
            js = 0.5 * (kl_pm + kl_qm).mean(dim=1)  # [B]
            js_div = js

            # top1 stability: fraction of tokens where argmax same
            prev_top1 = logits_prev.argmax(dim=-1)  # [B, T]
            cur_top1 = logits_cur.argmax(dim=-1)    # [B, T]
            top1_stab = (prev_top1 == cur_top1).float().mean(dim=1)  # [B]

            # hidden cosine change: 1 - cosine similarity of mean hidden states
            # hidden: [B, T, H]
            hidden_mean_prev = hidden_prev.mean(dim=1) if hidden_prev is not None else torch.zeros_like(hidden_cur.mean(dim=1))
            hidden_mean_cur = hidden_cur.mean(dim=1)
            cos_sim = F.cosine_similarity(hidden_mean_prev, hidden_mean_cur, dim=1)  # [B]
            hidden_cos_change = 1.0 - cos_sim  # [B]

        # recurrence count: we don't have it here; we'll set to 0 and let controller handle? Actually recurrence_count is not needed for halt head features? The controller uses it but halt head originally had 6 features including recurrence_count. We'll set recurrence_count to 0 as placeholder; the controller will add its own recurrence_count via build_features.
        recurrence_count = torch.zeros_like(entropy)  # [B]

        # stack features: entropy, entropy_delta, js_divergence, top1_stability, hidden_cosine_change, recurrence_count
        feats = torch.stack([
            entropy,
            entropy_delta,
            js_div,
            top1_stab,
            hidden_cos_change,
            recurrence_count,
        ], dim=1)  # [B, 6]
        return feats


class BlockHaltHead(HaltHead):
    """HALT head evaluated after each block execution."""

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        B, T, V = logits_cur.shape
        device = logits_cur.device

        cur_probs = F.softmax(logits_cur, dim=-1)
        cur_logprobs = F.log_softmax(logits_cur, dim=-1)
        cur_entropy = -(cur_probs * cur_logprobs).sum(dim=-1)
        entropy = cur_entropy.mean(dim=1)

        if logits_prev is None:
            entropy_delta = torch.zeros_like(entropy)
            js_div = torch.zeros_like(entropy)
            top1_stab = torch.ones_like(entropy)
            hidden_cos_change = torch.zeros_like(entropy)
        else:
            prev_probs = F.softmax(logits_prev, dim=-1)
            prev_logprobs = F.log_softmax(logits_prev, dim=-1)
            prev_entropy = -(prev_probs * prev_logprobs).sum(dim=-1).mean(dim=1)
            entropy_delta = prev_entropy - entropy

            m = 0.5 * (prev_probs + cur_probs)
            eps = 1e-10
            log_m = torch.log(m + eps)
            kl_pm = (prev_probs * (torch.log(prev_probs + eps) - log_m)).sum(dim=-1)
            kl_qm = (cur_probs * (torch.log(cur_probs + eps) - log_m)).sum(dim=-1)
            js = 0.5 * (kl_pm + kl_qm).mean(dim=1)
            js_div = js

            prev_top1 = logits_prev.argmax(dim=-1)
            cur_top1 = logits_cur.argmax(dim=-1)
            top1_stab = (prev_top1 == cur_top1).float().mean(dim=1)

            hidden_mean_prev = hidden_prev.mean(dim=1) if hidden_prev is not None else torch.zeros_like(hidden_cur.mean(dim=1))
            hidden_mean_cur = hidden_cur.mean(dim=1)
            cos_sim = F.cosine_similarity(hidden_mean_prev, hidden_mean_cur, dim=1)
            hidden_cos_change = 1.0 - cos_sim

        recurrence_count = torch.zeros_like(entropy)

        feats = torch.stack([
            entropy,
            entropy_delta,
            js_div,
            top1_stab,
            hidden_cos_change,
            recurrence_count,
        ], dim=1)
        return feats


class LayerHaltHead(HaltHead):
    """HALT head evaluated after each layer execution."""

    def forward_features(self, logits_prev: Tensor | None, logits_cur: Tensor, hidden_prev: Tensor | None, hidden_cur: Tensor) -> Tensor:
        B, T, V = logits_cur.shape
        device = logits_cur.device

        cur_probs = F.softmax(logits_cur, dim=-1)
        cur_logprobs = F.log_softmax(logits_cur, dim=-1)
        cur_entropy = -(cur_probs * cur_logprobs).sum(dim=-1)
        entropy = cur_entropy.mean(dim=1)

        if logits_prev is None:
            entropy_delta = torch.zeros_like(entropy)
            js_div = torch.zeros_like(entropy)
            top1_stab = torch.ones_like(entropy)
            hidden_cos_change = torch.zeros_like(entropy)
        else:
            prev_probs = F.softmax(logits_prev, dim=-1)
            prev_logprobs = F.log_softmax(logits_prev, dim=-1)
            prev_entropy = -(prev_probs * prev_logprobs).sum(dim=-1).mean(dim=1)
            entropy_delta = prev_entropy - entropy

            m = 0.5 * (prev_probs + cur_probs)
            eps = 1e-10
            log_m = torch.log(m + eps)
            kl_pm = (prev_probs * (torch.log(prev_probs + eps) - log_m)).sum(dim=-1)
            kl_qm = (cur_probs * (torch.log(cur_probs + eps) - log_m)).sum(dim=-1)
            js = 0.5 * (kl_pm + kl_qm).mean(dim=1)
            js_div = js

            prev_top1 = logits_prev.argmax(dim=-1)
            cur_top1 = logits_cur.argmax(dim=-1)
            top1_stab = (prev_top1 == cur_top1).float().mean(dim=1)

            hidden_mean_prev = hidden_prev.mean(dim=1) if hidden_prev is not None else torch.zeros_like(hidden_cur.mean(dim=1))
            hidden_mean_cur = hidden_cur.mean(dim=1)
            cos_sim = F.cosine_similarity(hidden_mean_prev, hidden_mean_cur, dim=1)
            hidden_cos_change = 1.0 - cos_sim

        recurrence_count = torch.zeros_like(entropy)

        feats = torch.stack([
            entropy,
            entropy_delta,
            js_div,
            top1_stab,
            hidden_cos_change,
            recurrence_count,
        ], dim=1)
        return feats