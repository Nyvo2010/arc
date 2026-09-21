"""AutoConfig for the ARC adaptive-recurrence JetMoE model (trust_remote_code).

Mirrors the training-time design:
  base_model_id          HF id of the base MoE (jetmoe/jetmoe-8b)
  scale                 model | block | layer  (adaptive unit granularity)
  block_size            transformer layers per recurrence unit (block scale)
  max_loops             hard safety cap on recurrence loops per unit
  compute_budget        optional global FLOP cap (None = disabled)
  controller_kwargs     formula constants for ThresholdController (Policy-T)
  quantize              '4bit' | '8bit' | null  (base load dtype on GPU)

The file is self-contained on purpose: it is loaded by transformers'
trust_remote_code mechanism and cannot depend on the ``arc`` package.
"""
from __future__ import annotations

from transformers import PretrainedConfig


class ARCAutoConfig(PretrainedConfig):
    model_type = "arc-jetmoe"

    def __init__(
        self,
        base_model_id: str = "jetmoe/jetmoe-8b",
        scale: str = "model",
        block_size: int = 4,
        max_loops: int = 4,
        compute_budget: float | None = None,
        controller_kwargs: dict | None = None,
        quantize: str = "4bit",
        torch_dtype: str = "float16",
        max_length: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_id = base_model_id
        self.scale = scale
        self.block_size = int(block_size)
        self.max_loops = int(max_loops)
        self.compute_budget = compute_budget
        self.controller_kwargs = dict(controller_kwargs or {})
        self.quantize = quantize
        self.torch_dtype = torch_dtype
        self.max_length = int(max_length)
        self.architectures = kwargs.get("architectures") or ["ARCJetMoeForCausalLM"]