from __future__ import annotations

import torch
from torch import Tensor

from arc.compute.flops import jetmoe_layer_flops_per_token
from arc.models.base import ARCAdapter, ForwardContext
from arc.routing.instrumentation import RouterRecorder


class JetMoeAdapter(ARCAdapter):
    def __init__(self, hf_model, block_size: int = 4):
        self.hf_model = hf_model
        self.net = hf_model.model
        self.head = hf_model.lm_head
        self.cfg = hf_model.config
        self.block_size = max(1, block_size)
        self._recorder = RouterRecorder.attach(self.net)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.net.embed_tokens(input_ids)

    def prepare(self, hidden: Tensor) -> ForwardContext:
        seq_len = hidden.shape[1]
        cache_position = torch.arange(seq_len, device=hidden.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = self.net._update_causal_mask(None, hidden, cache_position, None, False)
        return ForwardContext(position_ids=position_ids, attention_mask=attention_mask)

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

    def final_logits(self, hidden: Tensor) -> Tensor:
        return self.head(self.net.norm(hidden))

    def last_token_logits(self, hidden: Tensor) -> Tensor:
        return self.head(self.net.norm(hidden[:, -1:, :]))

    def num_layers(self) -> int:
        return len(self.net.layers)

    def num_blocks(self) -> int:
        import math

        return math.ceil(self.num_layers() / self.block_size)

    def begin_step(self) -> None:
        self._recorder.reset()

    def get_router_records(self) -> list[dict]:
        return self._recorder.records

    def unit_flops(self, scale: str, unit_index: int, seq_len: int) -> float:
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
        return per_token * n


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


def load_jetmoe(path: str, dtype: str | None = None, device: str | None = None):
    from transformers import AutoModelForCausalLM

    kwargs = {}
    if dtype:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device:
        model = model.to(device)
    model.eval()
    return model
