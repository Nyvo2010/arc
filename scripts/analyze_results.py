from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def summaries(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "summary":
                    rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw JSONL runs into quality-vs-compute tables")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--out", default="results/processed/summary.csv")
    args = parser.parse_args()

    df = summaries(Path(args.raw_dir))
    if df.empty:
        print("no summary records found")
        return

    cols = ["experiment_id", "scale", "name", "accuracy", "est_flops_forward", "executions", "avg_exec_latency_s"]
    df = df[[c for c in cols if c in df.columns]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    print("\nquality-compute frontier per scale (sorted by compute):")
    for scale, g in df.groupby("scale"):
        g = g.sort_values("est_flops_forward")
        frontier, best = [], -1.0
        for _, r in g.iterrows():
            if r["accuracy"] > best:
                best = r["accuracy"]
                frontier.append(r)
        print(f"  {scale}: " + " -> ".join(f"{r['name']}({r['accuracy']:.3f}@{r['est_flops_forward']:.2e})" for r in frontier))


if __name__ == "__main__":
    main()
