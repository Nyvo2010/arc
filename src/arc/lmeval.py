"""lm-evaluation-harness backend for the four ARC Phase 1 models.

Registers model type ``"arc"`` with EleutherAI's lm-evaluation-harness so
every benchmark task (mmlu, gpqa, hellaswag, arc, ...) can drive our
BaseLM / recurrent wrappers directly.

Scientific control protocol: within one experiment set, EVERY evaluation
parameter must stay identical across variants -- same tokenizer, same
tasks, few-shot counts, limits, seeds, quantization and hardware. The ONLY
things that may change between runs are ``scale`` and ``recurrence``.
This module therefore pins the adapter/quantization/tokenizer path by
construction; callers must only vary those two arguments.

Only loglikelihood-style (multiple choice) tasks are supported; generative
tasks would require HF generate(), which the ARC wrappers do not implement.
"""

from types import SimpleNamespace

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from torch import Tensor


class ArcCausalLMShim(torch.nn.Module):
    """Minimal causal-LM facade over an ARC model so HFLM can drive it."""

    def __init__(self, arc_lm):
        super().__init__()
        self.arc_lm = arc_lm
        self.config = arc_lm.adapter.cfg
        self.main_input_name = "input_ids"
        self.forward_count = 0
        self.total_flops_used = 0.0
        self.total_executions = 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: Tensor | None = None, **_) -> SimpleNamespace:
        result = self.arc_lm(input_ids)
        self.forward_count += 1
        # state.compute_used already includes the lm head for this forward
        self.total_flops_used += float(result.state.compute_used)
        self.total_executions += int(result.state.executions)
        return SimpleNamespace(logits=result.logits)

    def tie_weights(self) -> None:
        pass


def _build_arc_model(
    scale: str,
    recurrence: int,
    path: str,
    device_map,
    block_size: int | None = None,
    architecture: str = "jetmoe",
):
    from arc.models.registry import create_adapter
    from arc.recurrence.base import BaseLM, build_recurrent_model

    kwargs = {} if block_size is None else {"block_size": block_size}
    adapter = create_adapter(path, device_map=device_map, architecture=architecture, **kwargs)
    if scale == "base":
        return BaseLM(adapter)
    return build_recurrent_model(scale, adapter, recurrence)


@register_model("arc")
class ArcLM(HFLM):
    """HFLM subclass that evaluates ARC models.

    Example (notebook / python):

        import arc.lmeval
        from lm_eval import simple_evaluate
        results = simple_evaluate(
            model="arc",
            model_args=dict(scale="model", recurrence=2, path="jetmoe/jetmoe-8b"),
            tasks=["mmlu"], limit=500,
        )

    Example (CLI via the arc-lm-eval entry point):

        arc-lm-eval --model arc \
            --model_args scale=model,recurrence=2,path=jetmoe/jetmoe-8b \
            --tasks mmlu,gpqa_diamond_zeroshot --limit 500
    """

    last_instance: "ArcLM | None" = None

    def __init__(
        self,
        scale: str = "base",
        recurrence: int = 1,
        path: str = "tiny",
        block_size: int | None = None,
        architecture: str = "jetmoe",
        device_map: str | None = None,
        tokenizer_path: str | None = None,
        tokenizer=None,
        **hf_kwargs,
    ):
        """ARC-specific args are popped first; everything else (batch_size,
        max_length, add_bos_token, ...) is forwarded verbatim to HFLM so the
        lm-eval CLI can keep passing --batch_size itself."""
        scale = {"l": "layer", "b": "block", "m": "model"}.get(scale, scale)
        if scale not in ("base", "layer", "block", "model"):
            raise ValueError(f"unknown scale: {scale}")
        arc_lm = _build_arc_model(
            scale, int(recurrence), path, device_map, block_size, architecture
        )
        self.arc_model = ArcCausalLMShim(arc_lm)
        ArcLM.last_instance = self
        # lets HFLM fall back to AutoTokenizer.from_pretrained(path)
        self.arc_model.name_or_path = path
        tok_src = tokenizer if tokenizer is not None else tokenizer_path
        super().__init__(
            pretrained=self.arc_model,
            backend="causal",
            tokenizer=tok_src,
            **hf_kwargs,
        )


def cli() -> None:
    """Console entry point: importing this module registers 'arc', then the
    standard lm-eval CLI takes over."""
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
