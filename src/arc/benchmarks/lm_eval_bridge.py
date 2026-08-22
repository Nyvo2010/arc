"""
lm-evaluation-harness bridge for ARC variants.

Provides a thin HF-like loglikelihood scorer so ARC models can be evaluated
with EleutherAI lm-evaluation-harness on free community benchmarks without
changing the ARC scope.

Supported tasks (free & widely recognized):
  hellaswag, arc_easy, arc_challenge, gsm8k, mmlu

Usage:
  pip install lm-eval[hf]
  python -m arc.benchmarks.lm_eval_bridge --variant model_adaptive --source tiny --tasks hellaswag,arc_easy
"""

from __future__ import annotations

import argparse
import sys
sys.path.insert(0, "src")

import torch
from arc.inference import InferenceEngine
from arc.models.registry import MODEL_VARIANTS


class ARCLoglikelihoodWrapper:
    """
    Minimal wrapper exposing the interface required by lm-evaluation-harness
    for loglikelihood tasks. Implements:
      - loglikelihood(toks, choices)
    The wrapper builds an ARC model per call and caches the engine.
    """
    def __init__(self, source: str, variant: str, device: str = "cpu", max_loops: int = 4, seed: int = 0):
        self.engine = InferenceEngine(
            source=source,
            variant=variant,
            device_map="auto" if device != "cpu" else None,
            max_loops=max_loops,
            seed=seed,
        )
        self.device = device

    def loglikelihood(self, requests):
        """
        requests: list of (prompt_tokens, continuation_tokens)
        Returns list of (logprob, is_greedy)
        """
        results = []
        for prompt, cont in requests:
            input_ids = torch.tensor([prompt + cont], dtype=torch.long)
            if self.device != "cpu":
                input_ids = input_ids.to(self.device)
            res = self.engine(input_ids)
            logits = res.logits
            # simple logprob for the continuation tokens
            # We compute sum of log softmax over the continuation positions
            with torch.no_grad():
                log_probs = torch.log_softmax(logits, dim=-1)
                # gather logprob for true continuation tokens
                # continuation tokens align with logits[:, -len(cont):]
                start = logits.shape[1] - len(cont)
                if start < 0:
                    # fallback: use whole logits
                    target = torch.tensor([prompt + cont], dtype=torch.long, device=logits.device)
                else:
                    target = torch.tensor([cont], dtype=torch.long, device=logits.device)
                # Compute mean logprob for the continuation
                # For simplicity, take the last len(cont) positions
                selected = log_probs[0, -len(cont):, :]
                # Gather
                if len(cont) > 0:
                    token_ids = torch.tensor(cont, device=logits.device).unsqueeze(0)
                    gathered = torch.gather(selected.unsqueeze(0), 2, token_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, selected.shape[-1]))
                else:
                    gathered = torch.tensor([0.0])
                logprob = float(gathered.sum().item()) if len(cont) > 0 else 0.0
            results.append((logprob, True))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--variant", required=True, choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Lazy import to avoid hard dependency when not running eval
    try:
        from lm_eval import evaluator
    except Exception as e:
        raise SystemExit(f"lm-eval not installed: {e}")

    model = ARCLoglikelihoodWrapper(
        source=args.source,
        variant=args.variant,
        device=args.device,
        max_loops=args.max_loops,
        seed=args.seed,
    )

    results = evaluator.evaluate(
        model=model,
        tasks=args.tasks.split(","),
        device=args.device,
        batch_size=1,
    )
    print(results)


if __name__ == "__main__":
    main()
