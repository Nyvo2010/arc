from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor

from arc.inference import InferenceEngine
from arc.models.registry import MODEL_VARIANTS


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts_from_file(path: Path) -> List[Tensor]:
    # Expect JSONL with {"input_ids": [int,...]}
    prompts = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            ids = torch.tensor(obj["input_ids"], dtype=torch.long).unsqueeze(0)
            prompts.append(ids)
    return prompts


def evaluate_variant(
    variant: str,
    source: str,
    device: str,
    prompts: List[Tensor],
    max_loops: int = 4,
    recurrence: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    engine = InferenceEngine(
        source=source,
        variant=variant,
        device_map="auto" if device != "cpu" else None,
        max_loops=max_loops,
        recurrence=recurrence,
        seed=seed,
    )
    # move prompts to device lazily inside loop
    total_time = 0.0
    total_flops = 0.0
    total_tokens = 0
    executions_sum = 0
    results = []

    for ids in prompts:
        ids_dev = ids.to(device) if device != "cpu" else ids
        metrics = engine.measure(ids_dev)
        total_time += metrics["elapsed_s"]
        total_flops += metrics["compute_used"]
        total_tokens += metrics["tokens"]
        executions_sum += metrics["executions"]
        # store per-prompt metrics for full trace
        results.append({
            "compute_used": metrics["compute_used"],
            "executions": metrics["executions"],
            "elapsed_s": metrics["elapsed_s"],
            "tokens": metrics["tokens"],
            "unit_loop_counts": metrics["unit_loop_counts"],
        })

    avg_time = total_time / len(prompts) if prompts else 0.0
    avg_flops = total_flops / len(prompts) if prompts else 0.0
    avg_exec = executions_sum / len(prompts) if prompts else 0.0
    tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0

    cfg = MODEL_VARIANTS[variant]
    return {
        "variant": variant,
        "scale": cfg["scale"],
        "adaptive": cfg["adaptive"],
        "num_prompts": len(prompts),
        "avg_time_s": avg_time,
        "tokens_per_s": tokens_per_s,
        "total_flops": total_flops,
        "avg_flops_per_prompt": avg_flops,
        "avg_executions": avg_exec,
        "per_prompt": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL file with input_ids")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results_full.csv")
    parser.add_argument("--variants", nargs="+", default=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--recurrence", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = load_prompts_from_file(Path(args.prompts))

    rows = []
    for v in args.variants:
        print(f"Evaluating {v}")
        res = evaluate_variant(v, args.source, args.device, prompts, args.max_loops, args.recurrence, args.seed)
        # summary row
        rows.append({
            "variant": res["variant"],
            "scale": res["scale"],
            "adaptive": res["adaptive"],
            "num_prompts": res["num_prompts"],
            "avg_time_s": res["avg_time_s"],
            "tokens_per_s": res["tokens_per_s"],
            "total_flops": res["total_flops"],
            "avg_flops_per_prompt": res["avg_flops_per_prompt"],
            "avg_executions": res["avg_executions"],
        })
        # detailed per-prompt dump
        detail_path = Path(args.output).with_name(f"{args.output}.detail.{v}.jsonl")
        with detail_path.open("w") as f:
            for i, p in enumerate(res["per_prompt"]):
                f.write(json.dumps({"prompt_index": i, **p}) + "\n")

    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant","scale","adaptive","num_prompts","avg_time_s","tokens_per_s","total_flops","avg_flops_per_prompt","avg_executions"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary {args.output}")


if __name__ == "__main__":
    main()
