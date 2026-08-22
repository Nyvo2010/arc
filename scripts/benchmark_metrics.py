from __future__ import annotations
import argparse, csv, json, time, os, psutil
from pathlib import Path
from arc.models.registry import MODEL_VARIANTS
from arc.inference import InferenceEngine
import torch

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', required=True)
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    prompts = [torch.randint(2, 128, (1,128)) for _ in range(32)]

    rows=[]
    for v in MODEL_VARIANTS:
        print(f"[benchmark_metrics] {v}")
        engine = InferenceEngine(source=args.source, variant=v, device_map='auto' if args.device!='cpu' else None)
        times=[]; flops=[]; rams=[]
        for ids in prompts:
            ids = ids.to(args.device) if args.device!='cpu' else ids
            r0=ram_gb()
            m=engine.measure(ids)
            r1=ram_gb()
            times.append(m['elapsed_s'])
            flops.append(m['compute_used'])
            rams.append(max(r0,r1))
        rows.append({
            'variant': v,
            'scale': MODEL_VARIANTS[v]['scale'],
            'adaptive': MODEL_VARIANTS[v]['adaptive'],
            'avg_time_s': sum(times)/len(times),
            'avg_flops': sum(flops)/len(flops),
            'avg_ram_gb': sum(rams)/len(rams),
            'max_ram_gb': max(rams)
        })
    csv_path = out/'metrics_summary.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['variant','scale','adaptive','avg_time_s','avg_flops','avg_ram_gb','max_ram_gb'])
        w.writeheader()
        w.writerows(rows)
    print(f"[benchmark_metrics] Wrote {csv_path}")

if __name__=='__main__':
    main()
