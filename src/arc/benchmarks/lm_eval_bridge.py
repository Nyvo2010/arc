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
import json
import sys
sys.path.insert(0, "src")

import torch
from lm_eval.api.model import LM
from arc.inference import InferenceEngine
from arc.models.registry import MODEL_VARIANTS


class ARCLoglikelihoodWrapper(LM):
    """
    Minimal wrapper exposing the interface required by lm-evaluation-harness
    for loglikelihood tasks. Implements:
      - loglikelihood(toks, choices)
    The wrapper builds an ARC model per call and caches the engine.
    """
    def __init__(self, source: str, variant: str, device: str = "cpu", max_loops: int = 4, seed: int = 0,
                 recurrence: int = 1, **kwargs):
        super().__init__()
        self.engine = InferenceEngine(
            source=source,
            variant=variant,
            device_map="auto" if device != "cpu" else None,
            max_loops=max_loops,
            recurrence=recurrence,
            seed=seed,
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self._device = self.engine.adapter.net.embed_tokens.weight.device

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def _encode_pair(self, context: str, continuation: str):
        if context:
            full = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
            prefix = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        else:
            full = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
            prefix = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]
            full = prefix + full
        return full, len(prefix)

    def loglikelihood(self, requests):
        """
        requests: list of (prompt_tokens, continuation_tokens)
        Returns list of (logprob, is_greedy)
        """
        results = []
        for request in requests:
            prompt, cont = request.args
            full_ids, start = self._encode_pair(prompt, cont)
            # Build full input for model
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self._device)
            res = self.engine(input_ids)
            logits = res.logits  # [1, T, V]
            # Compute log softmax over vocab
            log_probs = torch.log_softmax(logits, dim=-1)
            # Position t predicts the token at t + 1.
            logit_start = max(start - 1, 0)
            continuation_ids = full_ids[start:]
            if not continuation_ids:
                logprob = 0.0
            else:
                # Clip to available logits
                valid_len = max(0, min(len(continuation_ids), logits.shape[1] - logit_start))
                if valid_len <= 0:
                    logprob = float("-inf")
                else:
                    target_ids = torch.tensor(continuation_ids[:valid_len], dtype=torch.long, device=log_probs.device)
                    # log_probs shape [1, T, V]; select slice
                    selected = log_probs[0, logit_start:logit_start + valid_len, :]  # [valid_len, V]
                    # gather logprobs for true tokens
                    gathered = selected[torch.arange(valid_len, device=log_probs.device), target_ids]
                    logprob = float(gathered.sum().item())
            greedy = bool(torch.equal(logits[0, logit_start:logit_start + valid_len].argmax(dim=-1), target_ids)) if continuation_ids and valid_len > 0 else True
            results.append((logprob, greedy))
        return results

    def loglikelihood_rolling(self, requests):
        return [score for score, _ in self.loglikelihood(
            [type("Request", (), {"args": ("", request.args[0])})() for request in requests]
        )]

    def generate_until(self, requests):
        # ARC-faithful greedy decoding through the recurrent wrapper.
        # NOTE: no KV cache exists for recurrence models, so cost scales with
        # output length; gsm8k-style generation is expensive by construction.
        results = []
        for request in requests:
            prompt, gen_kwargs = request.args
            ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device)
            max_new = int(gen_kwargs.get("max_gen_toks", 32))
            stop = list(gen_kwargs.get("until", []))
            generated: list[int] = []
            finished = False
            for _ in range(max_new):
                logits = self.engine(ids).logits[:, -1, :]
                next_id = int(logits.argmax(dim=-1).item())
                eos = getattr(self.engine.adapter.hf_model.config, "eos_token_id", None)
                if next_id == eos:
                    break
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=self._device)], dim=1)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(marker and marker in text for marker in stop):
                    finished = True
                    break
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if not finished:
                for marker in stop:
                    if marker and marker in text:
                        text = text[: text.index(marker)]
            results.append(text)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--variant", required=True, choices=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recurrence", type=int, default=None,
                        help="Fixed recurrence R for *_fixed variants (e.g. 2, 3, 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per task; use to trim benchmark cost")
    parser.add_argument("--output", default=None,
                        help="Path to write results JSON incrementally (per task)")
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
        recurrence=args.recurrence or 1,
    )

    tasks = args.tasks.split(",")
    merged: dict = {"variant": args.variant, "recurrence": args.recurrence, "results": {}, "configs": {}}
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(merged, indent=2))

    # Evaluate one task at a time so completed scores survive a later
    # timeout; the JSON on disk is updated after every task.
    for i, task in enumerate(tasks, 1):
        print(f"[bridge] task {i}/{len(tasks)}: {task}", flush=True)
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            device=args.device,
            batch_size=1,
            limit=args.limit,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
        )
        if results is None:
            continue
        merged["results"].update(results.get("results", {}))
        merged["configs"].update(results.get("configs", {}))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(merged, f, indent=2)
        print(f"[bridge] {args.variant} {task}: "
              f"{merged['results'].get(task)}", flush=True)

    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
