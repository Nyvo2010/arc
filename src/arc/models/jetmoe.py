from __future__ import annotations

import math

import torch
from torch import Tensor

from arc.compute.flops import jetmoe_layer_flops_per_token
from arc.models.base import ARCAdapter, ForwardContext


class JetMoeAdapter(ARCAdapter):
    def __init__(self, hf_model, block_size: int = 4):
        self.hf_model = hf_model
        self.net = hf_model.model
        self.head = hf_model.lm_head
        self.cfg = hf_model.config
        self.block_size = max(1, block_size)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.net.embed_tokens(input_ids)

    def forward_native(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        return hidden, self.project_logits(hidden)

    def prepare(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> ForwardContext:
        seq_len = hidden.shape[1]
        cache_position = torch.arange(seq_len, device=hidden.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self.net._update_causal_mask(
            attention_mask, hidden, cache_position, None, False
        )
        return ForwardContext(position_ids=position_ids, attention_mask=causal_mask)

    def forward_layer(self, layer_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        outputs = self.net.layers[layer_idx](
            hidden,
            position_ids=ctx.position_ids,
            attention_mask=ctx.attention_mask,
            use_cache=False,
        )
        return outputs[0]

    def forward_block(self, block_idx: int, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        start = block_idx * self.block_size
        end = min(start + self.block_size, self.num_layers())
        for i in range(start, end):
            hidden = self.forward_layer(i, hidden, ctx)
        return hidden

    def forward_model(self, hidden: Tensor, ctx: ForwardContext) -> Tensor:
        for i in range(self.num_layers()):
            hidden = self.forward_layer(i, hidden, ctx)
        return hidden

    def normalize(self, hidden: Tensor) -> Tensor:
        return self.net.norm(hidden)

    def project_logits(self, normalized_hidden: Tensor) -> Tensor:
        return self.head(normalized_hidden)

    def final_logits(self, hidden: Tensor) -> Tensor:
        return self.project_logits(self.normalize(hidden))

    def num_layers(self) -> int:
        return len(self.net.layers)

    def num_blocks(self) -> int:
        return math.ceil(self.num_layers() / self.block_size)

    def unit_flops(
        self, scale: str, unit_index: int, seq_len: int, batch_size: int = 1
    ) -> float:
        per_token = jetmoe_layer_flops_per_token(self.cfg, seq_len)
        if scale == "layer":
            n = 1
        elif scale == "block":
            start = unit_index * self.block_size
            n = min(start + self.block_size, self.num_layers()) - start
        elif scale == "model":
            n = self.num_layers()
        else:
            raise ValueError(f"unknown scale: {scale}")
        return per_token * seq_len * batch_size * n


def build_tiny_jetmoe(seed: int = 0, device: str | None = None):
    from transformers import JetMoeConfig, JetMoeForCausalLM

    cfg = JetMoeConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_key_value_heads=2,
        kv_channels=16,
        intermediate_size=96,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(seed)
    model = JetMoeForCausalLM(cfg)
    if device:
        model = model.to(device)
    model.eval()
    return model


def load_jetmoe(path: str, device_map: str | None = "auto"):
    """Load JetMoE-8B in 8-bit (bitsandbytes int8); the only supported mode."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map=device_map,
    )
    model.eval()
    return model


@torch.no_grad()
def verify_parity(adapter: JetMoeAdapter, seq_len: int = 16) -> dict:
    """Gate for real weights: the decomposed forward must reproduce the native
    HF forward within numerical tolerance before any recurrence experiment."""
    torch.manual_seed(0)
    device = adapter.net.embed_tokens.weight.device
    ids = torch.randint(2, adapter.cfg.vocab_size, (1, seq_len), device=device)
    native = adapter.hf_model(input_ids=ids).logits.float()
    h = adapter.embed(ids)
    ctx = adapter.prepare(h)
    h = adapter.forward_model(h, ctx)
    decomposed = adapter.final_logits(h).float()
    max_abs = float((native - decomposed).abs().max())
    return {"max_abs_diff": round(max_abs, 6), "ok": max_abs < 5e-2}
