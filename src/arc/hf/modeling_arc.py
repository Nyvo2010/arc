"""ARC adaptive-recurrence JetMoE for transformers trust_remote_code.

This module is deliberately SELF-CONTAINED: it is served from a Hugging Face
model repo and imported by transformers (``AutoModelForCausalLM.from_pretrained(
repo, trust_remote_code=True)``), so it cannot depend on the ``arc`` package.
The classes below mirror ``src/arc/models/jetmoe.py``,
``src/arc/recurrence/controller.py`` and ``src/arc/recurrence/adaptive.py`` at
the commit this model was trained with.

Load flow
---------
1. base MoE is loaded from ``config.base_model_id`` (4-bit NF4 on GPU, fp32 CPU)
2. the trained LoRA adapter is applied from the repo's ``adapter/`` subfolder
   via ``peft.PeftModel.from_pretrained``
3. the ARC wrapper reconstructs the exact training-time recurrence: per-unit
   (model/block/layer) forward, formula ThresholdController (Policy-T), hard
   max_loops cap, and the same RecurrenceResult contract.

Backward/control behavior matches benchmarks: default STOP bias, diminishing
returns gate, same converged_score weights.
"""
from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig


try:
    from peft import PeftModel
except Exception:  # pragma: no cover - only needed at load time
    PeftModel = None


# --------------------------------------------------------------------------
# tiny API mirror (no arc imports allowed here)
# --------------------------------------------------------------------------
class ForwardContext:
    def __init__(self, position_ids=None, attention_mask=None):
        self.position_ids = position_ids
        self.attention_mask = attention_mask


class RecurrenceResult(ModelOutput):
    logits: torch.FloatTensor = None
    final_hidden: torch.FloatTensor = None
    state: object = None


class RecurrenceState:
    def __init__(self, scale):
        self.scale = scale
        self.compute_used = 0.0
        self.executions = 0
        self.unit_loop_counts = {}
        self.decide_scores = {}
        self.decide_probs = {}
        self.current_unit = 0

    def record_execution(self, unit_index):
        self.executions += 1
        self.unit_loop_counts[unit_index] = self.unit_loop_counts.get(unit_index, 0) + 1

    def as_dict(self):
        return {
            "scale": self.scale,
            "compute_used": self.compute_used,
            "executions": self.executions,
            "unit_loop_counts": dict(self.unit_loop_counts),
            "decide_probs": dict(self.decide_probs),
        }


# --------------------------------------------------------------------------
# JetMoE adapter (mirrors src/arc/models/jetmoe.py)
# --------------------------------------------------------------------------
def _layer_flops_per_token(cfg) -> float:
    hidden = int(getattr(cfg, "hidden_size", 768))
    inter = int(getattr(cfg, "intermediate_size", 3072))
    attn = 4 * hidden * hidden
    mlp = 2 * hidden * inter
    return float(attn + mlp)


def _lm_head_flops_per_token(cfg) -> float:
    hidden = int(getattr(cfg, "hidden_size", 768))
    vocab = int(getattr(cfg, "vocab_size", 32000))
    return float(2 * hidden * vocab)


class JetMoeAdapter:
    def __init__(self, hf_model, block_size: int = 4):
        self.hf_model = hf_model
        self.net = hf_model.model
        self.head = hf_model.lm_head
        self.cfg = hf_model.config
        self.block_size = max(1, block_size)
        self.hidden_dim = int(getattr(self.cfg, "hidden_size", 768))

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.net.embed_tokens(input_ids)

    def forward_native(self, input_ids, attention_mask=None, position_ids=None):
        output = self.net(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        return hidden, self.project_logits(hidden)

    def prepare(self, hidden, attention_mask=None, position_ids=None):
        seq_len = hidden.shape[1]
        cache_position = torch.arange(seq_len, device=hidden.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self.net._update_causal_mask(
            attention_mask, hidden, cache_position, None, False
        )
        return ForwardContext(position_ids=position_ids, attention_mask=causal_mask)

    def forward_layer(self, layer_idx, hidden, ctx):
        out = self.net.layers[layer_idx](
            hidden,
            position_ids=ctx.position_ids,
            attention_mask=ctx.attention_mask,
            use_cache=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def forward_block(self, block_idx, hidden, ctx):
        start = block_idx * self.block_size
        end = min(start + self.block_size, self.num_layers())
        for i in range(start, end):
            hidden = self.forward_layer(i, hidden, ctx)
        return hidden

    def forward_model(self, hidden, ctx):
        for i in range(self.num_layers()):
            hidden = self.forward_layer(i, hidden, ctx)
        return hidden

    def normalize(self, hidden):
        return self.net.norm(hidden)

    def project_logits(self, normalized_hidden):
        return self.head(normalized_hidden)

    def num_layers(self):
        return len(self.net.layers)

    def num_blocks(self):
        return math.ceil(self.num_layers() / self.block_size)

    def lm_head_flops_per_token(self):
        return _lm_head_flops_per_token(self.cfg)

    def unit_flops(self, scale, unit_index, seq_len, batch_size=1):
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
        return float(per_token) * float(seq_len) * float(batch_size) * float(n)


# --------------------------------------------------------------------------
# ThresholdController (mirrors src/arc/recurrence/controller.py)
# --------------------------------------------------------------------------
class ThresholdController:
    def __init__(
        self,
        max_loops: int = 4,
        compute_budget: float | None = None,
        ref_js: float = 0.05,
        ref_hidden: float = 0.1,
        ref_entropy_delta: float = 0.1,
        k: float = 12.0,
        bias: float = 0.6,
        halt_threshold: float = 0.45,
        min_gain: float = 0.02,
        w_js: float = 0.25,
        w_hidden: float = 0.25,
        w_top1: float = 0.20,
        w_entropy: float = 0.15,
        w_conf: float = 0.15,
    ):
        self.max_loops = max_loops
        self.compute_budget = compute_budget
        self.ref_js = ref_js
        self.ref_hidden = ref_hidden
        self.ref_entropy_delta = ref_entropy_delta
        self.k = k
        self.bias = bias
        self.halt_threshold = halt_threshold
        self.min_gain = min_gain
        self.w_js = w_js
        self.w_hidden = w_hidden
        self.w_top1 = w_top1
        self.w_entropy = w_entropy
        self.w_conf = w_conf
        self._vocab_size = None

    def _log_vocab(self, logits):
        if self._vocab_size is None:
            self._vocab_size = int(logits.shape[-1])
        return float(math.log(max(2, self._vocab_size)))

    @staticmethod
    def _softmax(logits):
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _entropy(p):
        eps = 1e-12
        return float(-(p * torch.log(p + eps)).sum(dim=-1).mean().item())

    @staticmethod
    def _js_divergence(p, q):
        eps = 1e-12
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
        kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
        return float(((kl_pm + kl_qm) * 0.5).mean().item())

    @staticmethod
    def _cosine_change(h_prev, h_cur):
        if h_prev is None:
            return 1.0
        eps = 1e-12
        h_prev_n = h_prev / (h_prev.norm(dim=-1, keepdim=True) + eps)
        h_cur_n = h_cur / (h_cur.norm(dim=-1, keepdim=True) + eps)
        cos = (h_prev_n * h_cur_n).sum(dim=-1).mean().item()
        return float(1.0 - max(min(cos, 1.0), -1.0))

    def build_features(
        self, logits_prev, logits_cur, hidden_prev, hidden_cur, recurrence_count, compute_used
    ):
        self._vocab_size = int(logits_cur.shape[-1])
        p_cur = self._softmax(logits_cur)
        entropy = self._entropy(p_cur)
        if logits_prev is not None:
            p_prev = self._softmax(logits_prev)
            entropy_prev = self._entropy(p_prev)
            entropy_delta = entropy - entropy_prev
            js = self._js_divergence(p_cur, p_prev)
            top1_stability = float((p_cur.argmax(dim=-1) == p_prev.argmax(dim=-1)).float().mean().item())
        else:
            entropy_delta = 0.0
            js = 0.0
            top1_stability = 0.0
        hidden_change = self._cosine_change(hidden_prev, hidden_cur)
        return {
            "entropy": entropy,
            "entropy_delta": entropy_delta,
            "js_divergence": js,
            "top1_stability": top1_stability,
            "hidden_cosine_change": hidden_change,
            "recurrence_count": recurrence_count,
            "compute_used": compute_used,
        }

    def converged_score(self, features, logits=None):
        log_v = self._log_vocab(logits) if logits is not None else 1.0
        js_n = min(features["js_divergence"] / max(self.ref_js, 1e-9), 1.0)
        hidden_n = min(features["hidden_cosine_change"] / max(self.ref_hidden, 1e-9), 1.0)
        entropy_n = min(features["entropy"] / max(log_v, 1e-9), 1.0)
        conf_n = min(max(-features["entropy_delta"] / max(self.ref_entropy_delta, 1e-9), 0.0), 1.0)
        score = (
            self.w_js * (1.0 - js_n)
            + self.w_hidden * (1.0 - hidden_n)
            + self.w_top1 * features["top1_stability"]
            + self.w_entropy * (1.0 - entropy_n)
            + self.w_conf * conf_n
        )
        return float(score)

    @staticmethod
    def _p_halt(score, k, bias):
        return 1.0 / (1.0 + math.exp(-k * (score - bias)))

    def decide(self, features, state):
        if features["recurrence_count"] >= self.max_loops:
            return False
        if self.compute_budget is not None and features["compute_used"] >= self.compute_budget:
            return False
        score = self.converged_score(features)
        key = getattr(state, "current_unit", 0)
        prev = getattr(state, "decide_scores", {})
        pscore = prev.get(key)
        prev[key] = score
        if features["recurrence_count"] >= 2 and pscore is not None:
            if score - pscore < self.min_gain:
                return False
        if self.max_loops is not None and features["recurrence_count"] >= self.max_loops:
            return False
        p_halt = self._p_halt(score, self.k, self.bias)
        if getattr(state, "decide_probs", None) is not None:
            state.decide_probs[key] = state.decide_probs.get(key, 0.0) + p_halt
        halt = p_halt >= self.halt_threshold
        return not halt


# --------------------------------------------------------------------------
# ARC recurrence LMs (mirrors src/arc/recurrence/adaptive.py + builder.py)
# --------------------------------------------------------------------------
class AdaptiveRecurrentLM(torch.nn.Module):
    scale = "model"

    def __init__(self, adapter, controller):
        super().__init__()
        self.adapter = adapter
        self.controller = controller
        self.transformer = adapter.hf_model

    def num_units(self):
        raise NotImplementedError

    def execute_unit(self, unit_index, hidden, ctx):
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        adapter = self.adapter
        hidden = adapter.embed(input_ids)
        ctx = adapter.prepare(hidden, attention_mask=attention_mask, position_ids=position_ids)
        seq_len = hidden.shape[1]
        batch_size = hidden.shape[0]
        state = RecurrenceState(scale=self.scale)

        with torch.no_grad():
            for unit_index in range(self.num_units()):
                state.current_unit = unit_index
                logits_prev = None
                hidden_prev = None
                rec_count = 0
                while True:
                    est = adapter.unit_flops(self.scale, unit_index, seq_len, batch_size=batch_size)
                    hidden = self.execute_unit(unit_index, hidden, ctx)
                    if self.scale == "model":
                        hidden = adapter.normalize(hidden)
                    state.compute_used += est
                    state.record_execution(unit_index)
                    rec_count += 1
                    hidden_for_logits = adapter.normalize(hidden) if self.scale != "model" else hidden
                    logits_cur = adapter.project_logits(hidden_for_logits)
                    features = self.controller.build_features(
                        logits_prev=logits_prev,
                        logits_cur=logits_cur,
                        hidden_prev=hidden_prev,
                        hidden_cur=hidden,
                        recurrence_count=rec_count,
                        compute_used=state.compute_used,
                    )
                    logits_prev = logits_cur.detach()
                    hidden_prev = hidden.detach()
                    if not self.controller.decide(features, state):
                        break
                    if rec_count >= self.controller.max_loops:
                        break

            final_hidden = adapter.normalize(hidden)
            logits = adapter.project_logits(final_hidden)
            state.compute_used += adapter.lm_head_flops_per_token() * seq_len * batch_size

        return RecurrenceResult(logits=logits, final_hidden=final_hidden, state=state)


class ModelAdaptiveRecurrenceLM(AdaptiveRecurrentLM):
    scale = "model"

    def num_units(self):
        return 1

    def execute_unit(self, unit_index, hidden, ctx):
        return self.adapter.forward_model(hidden, ctx)


class BlockAdaptiveRecurrenceLM(AdaptiveRecurrentLM):
    scale = "block"

    def num_units(self):
        return self.adapter.num_blocks()

    def execute_unit(self, unit_index, hidden, ctx):
        return self.adapter.forward_block(unit_index, hidden, ctx)


class LayerAdaptiveRecurrenceLM(AdaptiveRecurrentLM):
    scale = "layer"

    def num_units(self):
        return self.adapter.num_layers()

    def execute_unit(self, unit_index, hidden, ctx):
        return self.adapter.forward_layer(unit_index, hidden, ctx)


# --------------------------------------------------------------------------
# Hub-facing wrapper (trust_remote_code entry point)
# --------------------------------------------------------------------------
try:
    from .configuration_arc import ARCAutoConfig
except Exception:  # pragma: no cover - sibling module only exists in the repo
    ARCAutoConfig = PretrainedConfig


def _build_quant_config(quantize: str, device_type: str):
    if quantize not in ("4bit", "8bit") or device_type != "cuda":
        return None
    from transformers import BitsAndBytesConfig

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


device_type = "cuda" if torch.cuda.is_available() else "cpu"


def _load_base(config, **kwargs):
    from transformers import AutoModelForCausalLM

    kwargs.pop("trust_remote_code", None)
    quant = _build_quant_config(getattr(config, "quantize", "4bit"), device_type)
    device_map = kwargs.pop("device_map", "auto")
    torch_dtype = getattr(torch, getattr(config, "torch_dtype", "float16"), torch.float16) if quant else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=quant,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            **kwargs,
        )
    base.eval()
    return base


class ARCJetMoeForCausalLM(PreTrainedModel):
    config_class = ARCAutoConfig
    base_model_prefix = "arc"

    def __init__(self, config, adapter=None, controller=None):
        super().__init__(config)
        self.scale = getattr(config, "scale", "model")
        self.adapter = adapter
        self.controller = controller
        if adapter is not None:
            if self.scale == "model":
                self.arc = ModelAdaptiveRecurrenceLM(adapter, controller)
            elif self.scale == "block":
                self.arc = BlockAdaptiveRecurrenceLM(adapter, controller)
            elif self.scale == "layer":
                self.arc = LayerAdaptiveRecurrenceLM(adapter, controller)
            else:
                raise ValueError(f"unknown scale {self.scale!r}")
        else:
            self.arc = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        """Reconstruct base MoE + trained LoRA + ARC wrapper from the repo.

        Unlike standard PreTrainedModel.from_pretrained, weights are NOT
        re-serialized into the repo: the base comes from ``config.base_model_id``
        and the trainable weights from this repo's ``adapter/`` subfolder.
        """
        if config is None:
            config = ARCAutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif not isinstance(config, ARCAutoConfig):
            config = ARCAutoConfig(**config.to_dict())

        # locate the local snapshot dir of THIS repo (the adapter subfolder)
        from transformers.utils.hub import cached_file

        local_cfg = cached_file(
            pretrained_model_name_or_path,
            "config.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=True,
        )
        repo_dir = os.path.dirname(local_cfg) if local_cfg else None
        adapter_dir = os.path.join(repo_dir, "adapter") if repo_dir else None
        if adapter_dir is None or not (adapter_dir and os.path.isdir(adapter_dir)):
            raise RuntimeError(
                f"ARCHub: expected 'adapter/' (adapter_config.json + adapter_model.safetensors) "
                f"in {pretrained_model_name_or_path} (resolved {repo_dir})"
            )

        base = _load_base(config, **kwargs)

        if PeftModel is None:
            raise RuntimeError("ARCJetMoeForCausalLM requires peft (pip install peft)")
        peft = PeftModel.from_pretrained(base, adapter_dir)
        peft.eval()

        block_size = int(getattr(config, "block_size", 4))
        adapter = JetMoeAdapter(peft, block_size=block_size)

        ck = dict(getattr(config, "controller_kwargs", {}) or {})
        ck.setdefault("max_loops", int(getattr(config, "max_loops", 4)))
        ck.setdefault("compute_budget", getattr(config, "compute_budget", None))
        controller = ThresholdController(**ck)

        model = cls(config, adapter=adapter, controller=controller)
        model.config.update({"architectures": ["ARCJetMoeForCausalLM"]})
        model.eval()
        return model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        use_cache: bool | None = None,
        past_key_values=None,
        labels: Tensor | None = None,
        **kwargs,
    ):
        if self.arc is None:
            raise RuntimeError("ARCJetMoeForCausalLM not initialized (use from_pretrained)")
        result = self.arc(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return result

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "position_ids": kwargs.get("position_ids"),
        }

    def get_output_embeddings(self):
        return self.adapter.head if self.adapter is not None else None

    @property
    def can_generate(self):
        return True

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save config + the LoRA adapter only (the base MoE is referenced by id)."""
        import json

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        adapter_dir = os.path.join(save_directory, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        if self.adapter is not None:
            peft = self.adapter.hf_model
            if hasattr(peft, "save_pretrained"):
                peft.save_pretrained(adapter_dir)
                # make adapter self-describing: point at the remote base id
                acfg = os.path.join(adapter_dir, "adapter_config.json")
                if os.path.exists(acfg):
                    data = json.load(open(acfg))
                    data["base_model_name_or_path"] = self.config.base_model_id
                    json.dump(data, open(acfg, "w"), indent=2)
        return save_directory