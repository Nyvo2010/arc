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
            # Build full input for model
            input_ids = torch.tensor([prompt + cont], dtype=torch.long)
            if self.device != "cpu":
                input_ids = input_ids.to(self.device)
            res = self.engine(input_ids)
            logits = res.logits  # [1, T, V]
            # Compute log softmax over vocab
            log_probs = torch.log_softmax(logits, dim=-1)
            # Align logits to continuation tokens
            # logits at position t predicts token t+1; for full sequence prompt+cont,
            # the continuation tokens correspond to logits at positions len(prompt) .. len(prompt)+len(cont)-1
            start = len(prompt)
            end = start + len(cont)
            if len(cont) == 0:
                logprob = 0.0
            else:
                # Clip to available logits
                valid_len = max(0, min(len(cont), logits.shape[1] - start))
                if valid_len <= 0:
                    logprob = float("-inf")
                else:
                    target_ids = torch.tensor(cont[:valid_len], dtype=torch.long, device=log_probs.device).unsqueeze(0)
                    # log_probs shape [1, T, V]; select slice
                    selected = log_probs[0, start:start+valid_len, :]  # [valid_len, V]
                    # gather logprobs for true tokens
                    gathered = selected[torch.arange(valid_len, device=log_probs.device), target_ids.squeeze(0)]
                    logprob = float(gathered.sum().item())
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
