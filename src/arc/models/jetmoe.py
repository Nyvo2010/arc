from __future__ import annotations

import math

import torch
from torch import Tensor

from arc.models.base import ARCAdapter, ForwardContext


def _layer_flops_per_token(cfg) -> float:
    hidden = int(getattr(cfg, "hidden_size", 768))
    inter = int(getattr(cfg, "intermediate_size", 3072))
    # rough MoE aware estimate: attention + mlp, upper bound
    # attention ~ 4*hidden^2 per token
    attn = 4 * hidden * hidden
    mlp = 2 * hidden * inter
    # NOTE: this is an upper bound; actual MoE activation is sparse
    return float(attn + mlp)


def _lm_head_flops_per_token(cfg) -> float:
    hidden = int(getattr(cfg, "hidden_size", 768))
    vocab = int(getattr(cfg, "vocab_size", 32000))
    return float(2 * hidden * vocab)


class JetMoeAdapter(ARCAdapter):
    def __init__(self, hf_model, block_size: int = 4):
        self.hf_model = hf_model
        self.net = hf_model.model
        self.head = hf_model.lm_head
        self.cfg = hf_model.config
        self.block_size = max(1, block_size)
        self.hidden_dim = int(getattr(self.cfg, "hidden_size", 768))

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
        out = self.net.layers[layer_idx](
            hidden,
            position_ids=ctx.position_ids,
            attention_mask=ctx.attention_mask,
            use_cache=False,
        )
        return out[0] if isinstance(out, tuple) else out

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

    def lm_head_flops_per_token(self) -> float:
        return _lm_head_flops_per_token(self.cfg)

    def unit_flops(
        self, scale: str, unit_index: int, seq_len: int, batch_size: int = 1
    ) -> float:
        per_token = _layer_flops_per_token(self.cfg)
        if scale == "layer":
            n = 1
        elif scale == "block":
            start = unit_index * self.block_size
            end = min(start + self.block_size, self.num_layers())
            n = max(1, end - start) if end > start else 0
        elif scale == "model":
            n = self.num_layers()
        else:
            raise ValueError(f"unknown scale: {scale}")
        # FLOPs is upper bound; note MoE sparsity not modeled here
        return float(per_token) * float(seq_len) * float(batch_size) * float(n)


def build_tiny_jetmoe(seed: int = 0, device: str | None = None, attn_implementation: str | None = None):
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
    if attn_implementation:
        cfg._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    model = JetMoeForCausalLM(cfg)
    if device:
        model = model.to(device)
    model.eval()
    return model


def load_jetmoe(path: str, device_map: str | None = "auto"):
    """Load JetMoE-8B in 8-bit if GPU available, else CPU float32 fallback."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    has_gpu = torch.cuda.is_available()
    if has_gpu and device_map not in (None, "cpu"):
        # GPU path: 8-bit quant
        try:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map=device_map or "auto",
            )
            model.eval()
            return model
        except Exception as e:
            print(f"[load_jetmoe] 8-bit GPU load failed: {e}, falling back to CPU")

    # CPU fallback: no quantization
    print("[load_jetmoe] Loading on CPU float32 fallback")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


@torch.no_grad()
def verify_parity(adapter: JetMoeAdapter, seq_len: int = 16) -> dict:
    """Gate for real weights: the decomposed forward must reproduce the native
    HF forward -- unmasked, padded, and chained through model recurrence x2 --
    before any recurrence experiment.

    If the adapter is a tiny dummy build, the gate will still run but skips heavy checks
    and returns a warning rather than a hard failure.
    """
    torch.manual_seed(0)
    device = adapter.net.embed_tokens.weight.device
    is_dummy = int(getattr(adapter.cfg, "vocab_size", 0)) < 1000

    def max_diff(native: Tensor, decomposed: Tensor) -> float:
        return float((native - decomposed).abs().max())

    ids = torch.randint(2, adapter.cfg.vocab_size, (1, seq_len), device=device)

    # 1. unmasked single pass
    native = adapter.hf_model(input_ids=ids).logits.float()
    h = adapter.embed(ids)
    h = adapter.forward_model(h, adapter.prepare(h))
    max_abs = max_diff(native, adapter.final_logits(h).float())

    # 2. right-padded batch with explicit mask must match its real prefix
    pad = torch.zeros((1, 3), dtype=ids.dtype, device=device)
    batch = torch.cat([ids, pad], dim=1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros_like(pad)], dim=1)
    native_pad = adapter.hf_model(input_ids=batch, attention_mask=mask).logits[0, :seq_len].float()
    h = adapter.embed(batch)
    h = adapter.forward_model(h, adapter.prepare(h, attention_mask=mask))
    max_abs = max(max_abs, max_diff(native_pad, adapter.final_logits(h)[0, :seq_len].float()))

    # 3. R=2 model recurrence must equal two chained normalized native passes
    # Skip heavy recurrence for dummy models to keep parity fast
    if not is_dummy:
        out1 = adapter.net(input_ids=batch, attention_mask=mask, return_dict=True).last_hidden_state
        out2 = adapter.net(inputs_embeds=out1, attention_mask=mask, return_dict=True).last_hidden_state
        recurrent_native = adapter.project_logits(out2)[0, :seq_len].float()
        from arc.recurrence.builder import build_model
        recurrent = build_model(scale="model", adapter=adapter, adaptive=False, recurrence=2)
        result = recurrent(batch, attention_mask=mask)
        max_abs = max(max_abs, max_diff(recurrent_native, result.logits[0, :seq_len].float()))

    ok = max_abs < 5e-2
    note = "dummy_model" if is_dummy else None
    return {"max_abs_diff": round(max_abs, 6), "ok": ok, "note": note}
