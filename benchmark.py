from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from arc.models.registry import MODEL_VARIANTS
from arc.models.factory import build_arc_model
from arc.recurrence.state import RecurrenceState


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts(num_prompts: int = 32, seq_len: int = 128, vocab_size: int = 128) -> list[Tensor]:
    rng = torch.Generator().manual_seed(0)
    prompts = []
    for _ in range(num_prompts):
        ids = torch.randint(2, vocab_size, (1, seq_len), generator=rng)
        prompts.append(ids)
    return prompts


def measure_variant(variant_name: str, source: str, device: str, max_loops: int = 4, recurrence: int = 1):
    cfg = MODEL_VARIANTS[variant_name]
    scale = cfg["scale"]
    adaptive = cfg["adaptive"]

    model, adapter = build_arc_model(
        source=source,
        scale=scale,
        adaptive=adaptive,
        max_loops=max_loops,
        recurrence=recurrence,
        architecture="jetmoe",
    )
    model.eval()
    adapter.hf_model.eval()

    # move to device if possible
    if device != "cpu":
        try:
            model = model.to(device)
            adapter.hf_model.to(device)
        except Exception:
            device = "cpu"

    prompts = load_prompts(num_prompts=32, seq_len=128, vocab_size=128)
    total_time = 0.0
    total_flops = 0.0
    total_tokens = 0
    loop_counts = []

    with torch.no_grad():
        for ids in prompts:
            if device != "cpu":
                ids = ids.to(device)
            t0 = time.perf_counter()
            result = model(ids)
            t1 = time.perf_counter()
            total_time += t1 - t0

            state: RecurrenceState = result.state
            total_flops += float(state.compute_used)
            total_tokens += ids.numel()
            loop_counts.append(state.executions)

    avg_time_per_prompt = total_time / len(prompts)
    avg_flops = total_flops / len(prompts)
    avg_loops = sum(loop_counts) / len(loop_counts)

    return {
        "variant": variant_name,
        "scale": scale,
        "adaptive": adaptive,
        "avg_time_s": avg_time_per_prompt,
        "tokens_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "total_flops": total_flops,
        "avg_flops_per_prompt": avg_flops,
        "avg_executions": avg_loops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path or identifier for base model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--variants", nargs="+", default=list(MODEL_VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--recurrence", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    rows = []
    for v in args.variants:
        if v not in MODEL_VARIANTS:
            print(f"skip unknown variant {v}")
            continue
        cfg = MODEL_VARIANTS[v]
        if cfg["adaptive"]:
            rec = args.max_loops
        else:
            rec = args.recurrence
        print(f"Running {v} scale={cfg['scale']} adaptive={cfg['adaptive']}")
        metrics = measure_variant(v, args.source, args.device, max_loops=args.max_loops, recurrence=rec)
        rows.append(metrics)

    out_path = Path(args.output)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant","scale","adaptive","avg_time_s","tokens_per_s","total_flops","avg_flops_per_prompt","avg_executions"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
