"""Print a quality-per-compute table from an arc-benchmark.json artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text())
    fingerprint = {run["protocol_fingerprint"] for run in data["runs"]}
    if len(fingerprint) != 1:
        raise SystemExit("protocol fingerprints differ -- runs are not comparable")

    rows = []
    for run in data["runs"]:
        variant = run["variant"]
        accounting = run["accounting"]
        metrics = run["results"].get("results", {})
        accs = [m.get("acc,none") for m in metrics.values() if m.get("acc,none") is not None]
        mean_acc = sum(accs) / len(accs) if accs else float("nan")
        flops = accounting.get("total_flops_used", 0.0)
        rows.append((variant, mean_acc, flops))

    rows.sort(key=lambda r: -(r[1] / r[2] if r[2] else 0))
    print(f"{'variant':<14} {'mean_acc':>9} {'GFLOPs':>12} {'acc/GFLOP':>11}")
    for variant, acc, flops in rows:
        gflops = flops / 1e9
        ratio = acc / gflops if gflops else float("nan")
        name = f"{variant['scale']}{variant['recurrence'] if variant['scale'] != 'base' else ''}"
        print(f"{name:<14} {acc:>9.4f} {gflops:>12.1f} {ratio:>11.4f}")


if __name__ == "__main__":
    main()
