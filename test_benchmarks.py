from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import csv
import json
import time
from typing import Any, Dict, List

import torch

from arc.inference import InferenceEngine
from arc.models.registry import MODEL_VARIANTS
from arc.models.jetmoe import JetMoeAdapter


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts(path: Path) -> List[torch.Tensor]:
    prompts = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            ids = torch.tensor(obj["input_ids"], dtype=torch.long).unsqueeze(0)
            prompts.append(ids)
    return prompts


def run_variant(variant: str, source: str, device: str, prompts: List[torch.Tensor], seed: int = 0) -> Dict[str, Any]:
    cfg = MODEL_VARIANTS[variant]
    engine = InferenceEngine(
        source=source,
        variant=variant,
        device_map="auto" if device != "cpu" else None,
        seed=seed,
        max_loops=4,
        recurrence=2,
    )

    total_time = 0.0
    total_flops = 0.0
    total_tokens = 0
    per_prompt = []

    for i, ids in enumerate(prompts):
        ids_dev = ids.to(device) if device != "cpu" else ids
        t0 = time.perf_counter()
        metrics = engine.measure(ids_dev)
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        total_flops += metrics["compute_used"]
        total_tokens += metrics["tokens"]

        per_prompt.append({
            "prompt_index": i,
            "elapsed_s": metrics["elapsed_s"],
            "compute_used": metrics["compute_used"],
            "executions": metrics["executions"],
            "tokens": metrics["tokens"],
            "unit_loop_counts": metrics["unit_loop_counts"],
        })

    return {
        "variant": variant,
        "scale": cfg["scale"],
        "adaptive": cfg["adaptive"],
        "num_prompts": len(prompts),
        "avg_time_s": total_time / len(prompts) if prompts else 0.0,
        "tokens_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "total_flops": total_flops,
        "avg_flops_per_prompt": total_flops / len(prompts) if prompts else 0.0,
        "per_prompt": per_prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Production benchmark for ARC 7 variants")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--variants", nargs="+", default=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    prompts = load_prompts(Path(args.prompts))

    summary_rows = []
    for v in args.variants:
        print(f"Running {v}...")
        res = run_variant(v, args.source, args.device, prompts, args.seed)

        summary_rows.append({
            "variant": res["variant"],
            "scale": res["scale"],
            "adaptive": res["adaptive"],
            "num_prompts": res["num_prompts"],
            "avg_time_s": res["avg_time_s"],
            "tokens_per_s": res["tokens_per_s"],
            "total_flops": res["total_flops"],
            "avg_flops_per_prompt": res["avg_flops_per_prompt"],
        })

        detail_path = Path(args.output).with_name(f"{args.output}.detail.{v}.jsonl")
        with detail_path.open("w") as f:
            for p in res["per_prompt"]:
                f.write(json.dumps(p) + "\n")

    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant","scale","adaptive","num_prompts","avg_time_s","tokens_per_s","total_flops","avg_flops_per_prompt"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
