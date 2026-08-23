from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import psutil
import torch
from torch import Tensor

from arc.models.registry import MODEL_VARIANTS, create_adapter
from arc.models.factory import build_arc_model
from arc.recurrence.state import RecurrenceState
from arc.models.jetmoe import verify_parity

FIELDNAMES = [
    "variant", "scale", "adaptive", "recurrence", "max_loops",
    "num_prompts", "seq_len",
    "avg_time_s", "p50_time_s", "max_time_s",
    "tokens_per_s",
    "total_flops", "avg_flops_per_prompt",
    "avg_executions", "max_executions",
    "avg_recurrence_per_unit",
    "avg_ram_gb", "max_ram_gb",
    "tokens_in_total", "tokens_out_total",
    "device", "seed",
]


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts(num_prompts: int = 32, seq_len: int = 128, vocab_size: int = 128) -> list[Tensor]:
    rng = torch.Generator().manual_seed(0)
    return [
        torch.randint(2, vocab_size, (1, seq_len), generator=rng)
        for _ in range(num_prompts)
    ]


def ram_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _input_device(model, adapter, requested_device: str) -> torch.device:
    """Return the device expected by a regular or Accelerate-dispatched model."""
    hf_model = getattr(adapter, "hf_model", None)
    if hf_model is not None and getattr(hf_model, "hf_device_map", None):
        embedding = getattr(getattr(adapter, "net", None), "embed_tokens", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None and weight.device.type != "meta":
            return weight.device
    return torch.device(requested_device)


def config_grid(recurrences: list[int], max_loops: int) -> list[dict]:
    """base; fixed x R in {2,3,4}; adaptive at max_loops."""
    grid = [{"variant": "base", "recurrence": 1, "max_loops": max_loops}]
    for v, cfg in MODEL_VARIANTS.items():
        if v == "base":
            continue
        if cfg["adaptive"]:
            grid.append({"variant": v, "recurrence": 1, "max_loops": max_loops})
        else:
            for r in recurrences:
                grid.append({"variant": v, "recurrence": r, "max_loops": max_loops})
    return grid


def measure_config(variant_name: str, source: str, device: str,
                   recurrence: int, max_loops: int,
                   num_prompts: int, seq_len: int, seed: int,
                   adapter=None, block_size: int = 4) -> dict:
    cfg = MODEL_VARIANTS[variant_name]
    scale = cfg["scale"]
    adaptive = cfg["adaptive"]

    if adapter is None:
        model, adapter = build_arc_model(
            source=source,
            scale=scale,
            block_size=block_size,
            adaptive=adaptive,
            max_loops=max_loops,
            recurrence=recurrence,
            architecture="jetmoe",
        )
    else:
        from arc.recurrence.builder import build_model
        model = build_model(
            scale=scale,
            adapter=adapter,
            adaptive=adaptive,
            max_loops=max_loops,
            recurrence=recurrence,
        )
    model.eval()
    if hasattr(adapter, "hf_model"):
        adapter.hf_model.eval()

    dispatched = bool(getattr(getattr(adapter, "hf_model", None), "hf_device_map", None))
    if device != "cpu" and not dispatched:
        try:
            model = model.to(device)
            if hasattr(adapter, "hf_model"):
                adapter.hf_model.to(device)
        except Exception as e:
            print(f"[matrix] GPU move failed ({e}), staying on CPU")
            device = "cpu"

    input_device = _input_device(model, adapter, device)
    prompts = load_prompts(num_prompts=num_prompts, seq_len=seq_len)
    times: list[float] = []
    flops: list[float] = []
    executions: list[int] = []
    recurrence_sums: list[float] = []
    total_tokens_in = 0

    with torch.no_grad():
        for ids in prompts:
            ids = ids.to(input_device)
            t0 = time.perf_counter()
            result = model(ids)
            if input_device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            state: RecurrenceState = result.state
            times.append(t1 - t0)
            flops.append(float(state.compute_used))
            executions.append(int(state.executions))
            loop_counts = state.unit_loop_counts or {}
            n_units = max(len(loop_counts), 1)
            recurrence_sums.append(sum(loop_counts.values()) / n_units)
            total_tokens_in += int(ids.numel())

    total_time = sum(times)
    # logits per prompt: seq positions x vocab
    vocab = result.logits.shape[-1]
    tokens_out = num_prompts * seq_len * vocab

    row = {
        "variant": variant_name,
        "scale": scale,
        "adaptive": adaptive,
        "recurrence": recurrence,
        "max_loops": max_loops,
        "num_prompts": num_prompts,
        "seq_len": seq_len,
        "avg_time_s": total_time / len(times),
        "p50_time_s": sorted(times)[len(times) // 2],
        "max_time_s": max(times),
        "tokens_per_s": total_tokens_in / total_time if total_time > 0 else 0.0,
        "total_flops": sum(flops),
        "avg_flops_per_prompt": sum(flops) / len(flops),
        "avg_executions": sum(executions) / len(executions),
        "max_executions": max(executions),
        "avg_recurrence_per_unit": sum(recurrence_sums) / len(recurrence_sums),
        "avg_ram_gb": 0.0,
        "max_ram_gb": 0.0,
        "tokens_in_total": total_tokens_in,
        "tokens_out_total": tokens_out,
        "device": device,
        "seed": seed,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="ARC full experiment matrix with complete metrics")
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--recurrences", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--num_prompts", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--skip_parity", action="store_true")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("[matrix] CUDA not available, falling back to cpu")
        device = "cpu"

    set_seed(args.seed)
    adapter = create_adapter(
        args.source,
        block_size=args.block_size,
        device_map="auto" if device != "cpu" else None,
    )
    if not args.skip_parity:
        parity = verify_parity(adapter, seq_len=min(16, args.seq_len))
        print(f"[matrix] Real-model parity: {parity}", flush=True)
        if not parity["ok"]:
            raise RuntimeError(f"Real-model parity failed: {parity}")
    grid = config_grid(args.recurrences, args.max_loops)
    print(f"[matrix] {len(grid)} configs: {[(g['variant'], g['recurrence']) for g in grid]}")

    rows = []
    for g in grid:
        label = f"{g['variant']}_R{g['recurrence']}"
        print(f"[matrix] === {label} ===", flush=True)
        ram_before = ram_gb()
        try:
            row = measure_config(
                g["variant"], args.source, device,
                g["recurrence"], g["max_loops"],
                args.num_prompts, args.seq_len, args.seed,
                adapter=adapter, block_size=args.block_size,
            )
        except Exception as e:
            print(f"[matrix] ERROR {label}: {e}", flush=True)
            rows.append({
                **{k: "" for k in FIELDNAMES},
                "variant": g["variant"], "recurrence": g["recurrence"],
                "max_loops": g["max_loops"], "error": str(e),
            })
            continue
        row["avg_ram_gb"] = round((ram_gb() + ram_before) / 2, 4)
        row["max_ram_gb"] = round(max(ram_gb(), gpu_mem_gb()), 4)
        rows.append(row)
        write_csv(out / "matrix_results.csv", rows)
        import gc
        gc.collect()

    write_csv(out / "matrix_results.csv", rows)
    with (out / "matrix_results.json").open("w") as f:
        json.dump({"config": {**vars(args), "device": device}, "rows": rows}, f, indent=2)
    print(f"[matrix] Wrote {out}/matrix_results.csv and .json")


def write_csv(path: Path, rows: list[dict]):
    fields = FIELDNAMES + ["error"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
