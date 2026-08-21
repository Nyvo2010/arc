from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from arc.benchmarks.protocol import BenchmarkProtocol, ModelVariant


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_suite(
    variants: Iterable[ModelVariant],
    protocol: BenchmarkProtocol | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Run identical harness settings for each model variant."""
    import arc.lmeval  # noqa: F401 - registers the universal lm-eval model
    from lm_eval import simple_evaluate

    protocol = protocol or BenchmarkProtocol()
    variants = tuple(variants)
    if not variants:
        raise ValueError("at least one model variant is required")

    runs = []
    for variant in variants:
        evaluated = simple_evaluate(
            model="arc",
            model_args=variant.model_args(),
            **protocol.harness_args(),
        )
        model = arc.lmeval.ArcLM.last_instance
        accounting = {}
        if model is not None:
            accounting = {
                "total_flops_used": model.arc_model.total_flops_used,
                "total_executions": model.arc_model.total_executions,
            }
        runs.append(
            {
                "variant": variant.__dict__,
                "protocol": protocol.as_dict(),
                "protocol_fingerprint": _fingerprint(protocol.as_dict()),
                "accounting": accounting,
                "results": evaluated,
            }
        )

    output = {"protocol": protocol.as_dict(), "runs": runs}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    return output
