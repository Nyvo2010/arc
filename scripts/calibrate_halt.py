#!/usr/bin/env python3
"""Halt-head calibration by trajectory replay (one GPU pass, many configs).

Why this exists
---------------
A normal calibration sweep runs the whole benchmark once per controller
config: with a 7-task suite and 40 items/task that is ~280 forward passes per
config, so a 200-config grid is ~56k passes. That is why previous sweeps were
tiny (4-8 configs).

The trick here: the recurrent trajectory is INDEPENDENT of the halt decision.
Hidden state at loop t is a pure function of the input, so running the model
to max_loops=N and recording, per item and per loop index, (a) the controller
features and (b) the per-choice loglikelihoods, gives us EVERY controller
config at once. A halt policy is just "stop at the first loop where decide()
returns False", and both the features and the answer at every loop are already
in the recording. So we pay for one pass and replay thousands of policies for
free.

Correctness caveat (deliberate, documented): the recorded features are the
ones a full-depth run produces. A real adaptive run that halts early never
computes deeper loops, so its feature *values* at those depths are identical
(they are a function of the hidden state, which is halt-independent) - only
the fact that it stopped earlier differs. The replay reproduces real adaptive
runs exactly for any policy that halts at or before the collected depth.

Splits: calibration runs on CALIB_TASKS (TRAIN splits, disjoint from the
reported benchmark splits) so the selected operating point is not fit to the
test numbers.

Usage:
    # Phase 1 (GPU): record trajectories
    python scripts/calibrate_halt.py collect \
        --base /kaggle/working/jetmoe-8b \
        --model model_adaptive \
        --adapters model_adaptive=/kaggle/input/.../model/adapter \
        --max_loops 8 --out /kaggle/output/traj/model.jsonl

    # Phase 2 (CPU): replay the grid
    python scripts/calibrate_halt.py sweep \
        --traj /kaggle/output/traj/model.jsonl \
        --out /kaggle/output/traj/model-sweep.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _torch():
    """Lazy torch import. The ``sweep`` half is pure CPU replay of recorded
    trajectories and must run without torch (it runs on a laptop after the
    GPU kernel has written the trajectory files)."""
    import torch
    from torch import nn
    return torch, nn


# --------------------------------------------------------------------------
# Phase 1: collect trajectories
# --------------------------------------------------------------------------
def record_item(model, ids, span_mask, max_loops: int, cont_ids=None, torch=None, nn=None) -> dict:
    """Run the recurrent model to ``max_loops`` recording per-loop state.

    Returns a JSON-serialisable record for ONE benchmark item (all its
    choices are rows of the batch, as in the real harness).
    """
    adapter = model.adapter
    scale = model.scale
    hidden = adapter.embed(ids)
    ctx = adapter.prepare(hidden)
    seq_len = int(ids.shape[1])
    B = int(ids.shape[0])

    if scale == "model":
        units = [0]
    elif scale == "block":
        units = list(range(adapter.num_blocks()))
    elif scale == "layer":
        units = list(range(adapter.num_layers()))
    else:
        raise ValueError(scale)

    shift_labels = ids[:, 1:]
    span = span_mask[:, 1:]

    def choice_scores(logits) -> list[float]:
        logp = nn.functional.log_softmax(logits.float(), dim=-1)
        shift_logp = logp[:, :-1]
        out = []
        for i in range(B):
            nz = span[i].nonzero(as_tuple=True)[0].cpu()
            if nz.numel() == 0:
                out.append(float("-inf"))
                continue
            toks = shift_labels[i][nz].cpu()
            sub = shift_logp[i][nz].detach().cpu()
            lg = sub.gather(-1, toks.unsqueeze(-1))
            out.append(float(lg.squeeze(-1).sum()))
        return out

    unit_records = []
    logits_prev: Tensor | None = None
    hidden_prev: Tensor | None = None
    for u in units:
        rec = {"unit": u, "loops": []}
        loop_i = 0
        while loop_i < max_loops:
            if scale == "model":
                hidden = adapter.forward_model(hidden, ctx)
                hidden = adapter.normalize(hidden)
            elif scale == "block":
                hidden = adapter.forward_block(u, hidden, ctx)
            else:
                hidden = adapter.forward_layer(u, hidden, ctx)

            h_for_logits = hidden if scale == "model" else adapter.normalize(hidden)
            logits = adapter.project_logits(h_for_logits)

            feats = model.controller.build_features(
                logits_prev=logits_prev,
                logits_cur=logits,
                hidden_prev=hidden_prev,
                hidden_cur=hidden,
                recurrence_count=loop_i + 1,
                compute_used=0.0,
            )
            rec["loops"].append({
                "loop": loop_i,
                "js": feats.js_divergence,
                "hid": feats.hidden_cosine_change,
                "ent": feats.entropy,
                "dent": feats.entropy_delta,
                "top1": feats.top1_stability,
                "nan": feats.nan,
                "scores": choice_scores(logits),
            })
            logits_prev = logits
            hidden_prev = hidden
            loop_i += 1
        unit_records.append(rec)

    return {
        "units": unit_records,
        "n_tokens": [max(1, int(ct.shape[0])) for ct in cont_ids],
        "lm_head_flops": float(adapter.lm_head_flops_per_token()) * seq_len * B,
        "unit_est": [float(adapter.unit_flops(scale, u, seq_len, batch_size=B)) for u in units],
    }


def cmd_collect(args) -> None:
    torch, nn = _torch()
    from arc.eval.harness import EvalModel, _get_tokenizer, _tokenize
    from arc.eval.tasks import CALIB_TASKS, TASKS

    tasks = [t for t in args.tasks.split(",") if t]
    registry = CALIB_TASKS if args.task_set == "calib" else TASKS
    tokenizer = _get_tokenizer(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for key in args.model.split(","):
        if not key:
            continue
        adapter_dir = None
        for part in args.adapters.split(","):
            if "=" in part and part.split("=", 1)[0].strip() == key:
                adapter_dir = part.split("=", 1)[1].strip()

        model = EvalModel(
            key=key, base_path=args.base, adapter_dir=adapter_dir,
            depth=1, device_map=args.device_map, budgeted=True,
            max_loops=args.max_loops, controller_kwargs={},
        )
        out_path = Path(args.out)
        if out_path.suffix == ".jsonl" and len(args.model.split(",")) > 1:
            out_path = out_path.with_name(f"{out_path.stem}-{key}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_items = 0
        with out_path.open("w") as f:
            for task in tasks:
                spec = registry[task]
                limit = args.limit or spec.get("default_limit", 400)
                items = spec["loader"](limit)
                for stem, conts, label in items:
                    ctx_ids = _tokenize(tokenizer, stem)
                    cont_ids = [_tokenize(tokenizer, c) for c in conts]
                    ids, span_mask = build_batch(model, ctx_ids, cont_ids, torch=torch)
                    with torch.no_grad():
                        rec = record_item(model, ids, span_mask, args.max_loops,
                                          cont_ids=cont_ids, torch=torch, nn=nn)
                    rec["task"] = task
                    rec["label"] = int(label)
                    f.write(json.dumps(rec) + "\n")
                    n_items += 1
                print(f"[collect:{key}] {task}: {len(items)} items", flush=True)

        print(f"[collect:{key}] wrote {n_items} trajectories -> {out_path}", flush=True)
        del model
        torch.cuda.empty_cache()


def build_batch(model, ctx_ids, cont_ids, torch=None):
    """Replicate harness batching: pad choices, return (ids, cont span mask).

    ``ctx_ids`` is a single 1-D tensor (the harness wraps it per choice), so its
    length is read with ``shape`` - a truthiness test on a multi-element
    tensor raises.
    """
    device = model.device
    max_ctx = int(ctx_ids.shape[0]) if ctx_ids is not None else 0
    max_cont = max(int(c.shape[0]) for c in cont_ids) if cont_ids else 0
    pad_id = getattr(model.adapter.hf_model.config, "pad_token_id", None)
    pad = 0 if pad_id is None else int(pad_id)
    seq_len = max_ctx + max_cont
    B = len(cont_ids)
    ids = torch.full((B, seq_len), pad, dtype=torch.long, device=device)
    mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
    for i, (c, ct) in enumerate(zip([ctx_ids] * B, cont_ids)):
        ln = min(int(c.shape[0]), max_ctx)
        ids[i, :ln] = c[:ln].to(device)
        s = min(ln, seq_len)
        e = min(s + int(ct.shape[0]), seq_len)
        ids[i, s:e] = ct[: e - s].to(device)
        mask[i, s:e] = True
    return ids, mask
    seq_len = max_ctx + max_cont
    B = len(cont_ids)
    ids = torch.full((B, seq_len), pad, dtype=torch.long, device=device)
    mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
    for i, (c, ct) in enumerate(zip([ctx_ids] * B, cont_ids)):
        ln = min(int(c.shape[0]), max_ctx)
        ids[i, :ln] = c[:ln].to(device)
        s = min(ln, seq_len)
        e = min(s + int(ct.shape[0]), seq_len)
        ids[i, s:e] = ct[: e - s].to(device)
        mask[i, s:e] = True
    return ids, mask


# --------------------------------------------------------------------------
# Phase 2: offline replay of the controller grid
# --------------------------------------------------------------------------
DEFAULTS = dict(
    ref_js=0.05, ref_hidden=0.1, ref_entropy_delta=0.1,
    k=12.0, bias=0.6, halt_threshold=0.45, min_gain=0.02,
    w_js=0.25, w_hidden=0.25, w_top1=0.20, w_entropy=0.15, w_conf=0.15,
)


def converged_score(feat: dict, log_v: float, p: dict) -> float:
    def b(x):
        return 0.0 if not math.isfinite(x) else max(0.0, min(float(x), 1.0))
    js_n = b(feat["js"] / max(p["ref_js"], 1e-9))
    hid_n = b(feat["hid"] / max(p["ref_hidden"], 1e-9))
    ent_n = b(feat["ent"] / max(log_v, 1e-9))
    conf_n = b(-feat["dent"] / max(p["ref_entropy_delta"], 1e-9))
    return (
        p["w_js"] * (1.0 - js_n)
        + p["w_hidden"] * (1.0 - hid_n)
        + p["w_top1"] * b(feat["top1"])
        + p["w_entropy"] * (1.0 - ent_n)
        + p["w_conf"] * conf_n
    )


def decide(feat, p, score, prev_score, cap):
    """Mirror ThresholdController.decide for the cap/threshold/min_gain logic."""
    if feat["nan"]:
        return False
    if feat["loop"] + 1 >= cap:
        return False
    if feat["loop"] + 1 >= 2 and prev_score is not None:
        if score - prev_score < p["min_gain"]:
            return False
    p_halt = 1.0 / (1.0 + math.exp(-p["k"] * (score - p["bias"])))
    return not (p_halt >= p["halt_threshold"])


def replay(recs, p, cap, log_v):
    """Return (acc, acc_norm, avg_loops, avg_gflop, halt_hist, nan_items)."""
    correct = correct_norm = total = 0
    loops_sum = gflop_sum = 0.0
    hist = Counter()
    nan_items = 0
    for rec in recs:
        stop_loop = None
        total_compute = rec["lm_head_flops"]
        for ui, unit in enumerate(rec["units"]):
            prev = None
            stop_in_unit = False
            for feat in unit["loops"]:
                score = converged_score(feat, log_v, p)
                if decide(feat, p, score, prev, cap):
                    prev = score
                    total_compute += rec["unit_est"][ui]
                    continue
                # halts here
                prev = score
                total_compute += rec["unit_est"][ui]
                stop_in_unit = True
                stop_loop = (ui, feat)
                break
            if stop_in_unit:
                # remaining units still run at least one pass in the real model
                for rj in range(ui + 1, len(rec["units"])):
                    total_compute += rec["unit_est"][rj]
                    for feat in rec["units"][rj]["loops"]:
                        stop_loop = (rj, feat)
                break
        if stop_loop is None:
            ui = len(rec["units"]) - 1
            stop_loop = (ui, rec["units"][ui]["loops"][-1])
        if any(f["nan"] for u in rec["units"] for f in u["loops"]):
            nan_items += 1
        scores = stop_loop[1]["scores"]
        n_tok = rec.get("n_tokens") or [1] * len(scores)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == rec["label"]:
            correct += 1
        ms = [s / n_tok[i] if s != float("-inf") else float("-inf")
              for i, s in enumerate(scores)]
        predn = max(range(len(ms)), key=lambda i: ms[i])
        if predn == rec["label"]:
            correct_norm += 1
        total += 1
        loops = stop_loop[0] + stop_loop[1]["loop"] + 1
        hist[loops] += 1
        loops_sum += loops
        gflop_sum += total_compute
    return (
        100.0 * correct / total,
        100.0 * correct_norm / total,
        loops_sum / total,
        gflop_sum / total / 1e9,
        ";".join(f"{k}@{hist[k]}" for k in sorted(hist)),
        nan_items,
    )


def cmd_sweep(args) -> None:
    recs = [json.loads(l) for l in open(args.traj) if l.strip()]
    log_v = math.log(args.vocab_size)
    print(f"[sweep] {len(recs)} trajectories, vocab={args.vocab_size}")

    biases = [float(x) for x in args.biases.split(",")]
    ks = [float(x) for x in args.ks.split(",")]
    thrs = [float(x) for x in args.thresholds.split(",")]
    gains = [float(x) for x in args.min_gains.split(",")]
    caps = [int(x) for x in args.caps.split(",")]

    rows = []
    for cap in caps:
        for b, k, thr, mg in itertools.product(biases, ks, thrs, gains):
            p = dict(DEFAULTS, bias=b, k=k, halt_threshold=thr, min_gain=mg)
            acc, accn, loops, gflop, hist, nan = replay(recs, p, cap, log_v)
            if nan > args.max_nan_items:
                continue
            distinct = len(hist.split(";")) if hist else 0
            rows.append({
                "cap": cap, "bias": b, "k": k, "halt_threshold": thr, "min_gain": mg,
                "acc_pct": round(acc, 2),
                "acc_norm_pct": round(accn, 2),
                "avg_loops": round(loops, 2),
                "gflop_per_item": round(gflop, 1),
                "acc_norm_per_gflop": round(accn / gflop, 4) if gflop else 0.0,
                "distinct_loop_levels": distinct,
                "halt_hist": hist,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {len(rows)} rows -> {out}")

    # Pareto front (max acc, min gflop), then a spread-constrained front.
    def front(rows_, require_spread):
        cand = [r for r in rows_ if (r["distinct_loop_levels"] >= 2 or not require_spread)]
        out = []
        for r in cand:
            dominated = any(
                c is not r
                and c["acc_pct"] >= r["acc_pct"]
                and c["gflop_per_item"] <= r["gflop_per_item"]
                and (c["acc_pct"] > r["acc_pct"] or c["gflop_per_item"] < r["gflop_per_item"])
                for c in cand
            )
            if not dominated:
                out.append(r)
        out.sort(key=lambda x: x["gflop_per_item"])
        return out

    for label, rs in (("ALL", front(rows, False)), ("SPREAD>=2", front(rows, True))):
        print(f"\n=== Pareto by acc_norm ({label}) ===")
        print(f"{'cap':>3s} {'bias':>5s} {'k':>4s} {'thr':>5s} {'mgain':>6s} {'accN':>5s} {'loops':>6s} {'GFLOP':>6s} {'accN/G':>6s}  hist")
        for r in rs[:20]:
            print(f"{r['cap']:>3d} {r['bias']:>5.2f} {r['k']:>4.0f} {r['halt_threshold']:>5.2f} "
                  f"{r['min_gain']:>6.3f} {r['acc_norm_pct']:>5.1f} {r['avg_loops']:>6.2f} "
                  f"{r['gflop_per_item']:>6.1f} {r['acc_norm_per_gflop']:>6.3f}  {r['halt_hist'][:44]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--base", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--adapters", default="")
    c.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,piqa,boolq,sciq")
    c.add_argument("--task_set", default="calib", choices=["calib", "bench"])
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--max_loops", type=int, default=8)
    c.add_argument("--device_map", default="auto")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser("sweep")
    s.add_argument("--traj", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--vocab_size", type=int, default=16000)
    s.add_argument("--biases", default="0.30,0.40,0.50,0.60")
    s.add_argument("--ks", default="8,14")
    s.add_argument("--thresholds", default="0.35,0.50,0.65")
    s.add_argument("--min_gains", default="0.0,0.03,0.06")
    s.add_argument("--caps", default="2,4,6")
    s.add_argument("--max_nan_items", type=int, default=0)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
