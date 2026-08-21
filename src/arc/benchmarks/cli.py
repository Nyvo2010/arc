from __future__ import annotations

import argparse

from arc.benchmarks.protocol import BenchmarkProtocol, ModelVariant
from arc.benchmarks.runner import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed ARC Kaggle benchmark suite")
    parser.add_argument("--path", required=True)
    parser.add_argument("--architecture", default="jetmoe")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="/kaggle/working/arc-benchmark.json")
    args = parser.parse_args()
    variants = [ModelVariant(args.architecture, args.path)]
    variants += [
        ModelVariant(args.architecture, args.path, scale=scale, recurrence=recurrence)
        for scale in ("l", "b", "m")
        for recurrence in (2, 3, 4)
    ]
    run_suite(variants, BenchmarkProtocol(limit=args.limit), args.output)


if __name__ == "__main__":
    main()
