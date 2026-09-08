"""G2 gate evaluation for Stage A (PLAN §6).

G2: Tier A passes if recurrent val loss < its own step-0 val loss AND
ARC-Easy-val is within 2 pts of the ``base`` control. Reads the CSV logs
written by the trainer; ARC-Easy probe rows are optional (missing probe =
partial verdict, never a silent pass).
"""

from __future__ import annotations

import csv
from pathlib import Path


def _read_rows(path: str | Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate_g2(
    treatment_eval_csv: str | Path,
    control_eval_csv: str | Path,
    arc_easy_treatment: float | None = None,
    arc_easy_control: float | None = None,
    tolerance_pts: float = 2.0,
) -> dict:
    """Return ``{'pass': bool, 'reasons': [...], 'numbers': {...}}``."""
    reasons: list[str] = []
    t_rows = _read_rows(treatment_eval_csv)
    c_rows = _read_rows(control_eval_csv)
    if not t_rows or not c_rows:
        return {"pass": False, "reasons": ["empty eval log"], "numbers": {}}

    step0 = float(t_rows[0]["val_loss"])
    best = min(float(r["val_loss"]) for r in t_rows)
    loss_ok = best < step0
    reasons.append(f"treatment best val_loss {best:.4f} {('<' if loss_ok else '>=')} step-0 {step0:.4f}")
    numbers = {"treatment_step0_val_loss": step0, "treatment_best_val_loss": best}

    if arc_easy_treatment is None or arc_easy_control is None:
        reasons.append("ARC-Easy probe missing: G2 partial, cannot fully pass")
        return {"pass": False, "reasons": reasons, "numbers": numbers}

    gap = float(arc_easy_treatment) - float(arc_easy_control)
    probe_ok = gap >= -tolerance_pts
    reasons.append(f"ARC-Easy gap {gap:+.2f} pts (tolerance -{tolerance_pts})")
    numbers.update(
        {"arc_easy_treatment": arc_easy_treatment, "arc_easy_control": arc_easy_control, "gap": gap}
    )
    return {"pass": bool(loss_ok and probe_ok), "reasons": reasons, "numbers": numbers}
