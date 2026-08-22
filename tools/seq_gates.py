"""Gate (b): knee-cell honesty and plateau non-inflation for the seq head.

The mid-year ranking failure lives in the knee-entry population: margin
0.05-0.10 V with elevated within-day dV/dT (beta_30 >= 0.014), median failure
~25 days after the cutoff. A model that prices those cells at their realized
rate (~0.2-0.35) can rank them; one that prices them like the plateau cannot.
The dual failure is plateau inflation: rows far above the threshold getting
boosted past twice their empirical rate, which is exactly the volume poison
that killed the public submission.

Measured on the out-of-fold stride predictions from ``tools/train_seq.py``
(rows restricted to the 42-day decision with a full window remaining, i.e.
``decision_horizon == 42``, so "empirical" is a genuine 42-day rate).

    python tools/seq_gates.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WORK = Path(
    os.environ.get(
        "SEQ_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\seq",
    )
)

KNEE_MARGIN = (0.05, 0.10)
KNEE_BETA = 0.014
KNEE_BAND = (0.20, 0.35)
PLATEAU_INFLATION_LIMIT = 2.0

MARGIN_EDGES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 10.0)
BETA_EDGES = (-1e9, 0.007, 0.014, 1e9)


def cell_table(margin, beta, p, truth) -> list[dict]:
    rows = []
    for mi in range(len(MARGIN_EDGES) - 1):
        for bi in range(len(BETA_EDGES) - 1):
            mask = (
                (margin >= MARGIN_EDGES[mi])
                & (margin < MARGIN_EDGES[mi + 1])
                & (beta >= BETA_EDGES[bi])
                & (beta < BETA_EDGES[bi + 1])
            )
            n = int(mask.sum())
            if n < 30:
                continue
            mean_p = float(p[mask].mean())
            emp = float(truth[mask].mean())
            rows.append(
                {
                    "margin": f"{MARGIN_EDGES[mi]}-{MARGIN_EDGES[mi+1]}",
                    "beta30": f"{BETA_EDGES[bi]:.3f}-{BETA_EDGES[bi+1]:.3f}",
                    "n": n,
                    "mean_p": round(mean_p, 4),
                    "empirical": round(emp, 4),
                    "ratio": round(mean_p / max(emp, 1e-4), 2) if emp > 0 else None,
                    "sum_p": round(float(p[mask].sum()), 1),
                    "events": int(truth[mask].sum()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_WORK / "seq_oof.npz")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/seq_gate_knee.json")
    args = parser.parse_args()

    data = np.load(args.oof, allow_pickle=True)
    oof = data["oof"].astype(float)
    truth = data["truth"].astype(int)
    margin = data["margin"].astype(float)
    beta30 = data["beta30"].astype(float)
    horizon = data["decision_horizon"].astype(float)

    # Full 42-day windows only: rows whose decision window is clipped by the
    # observation end would depress both p and the empirical rate mechanically.
    full = horizon >= 42.0
    oof_f, truth_f = oof[full], truth[full]
    margin_f, beta_f = margin[full], beta30[full]

    knee = (
        (margin_f >= KNEE_MARGIN[0])
        & (margin_f < KNEE_MARGIN[1])
        & np.isfinite(beta_f)
        & (beta_f >= KNEE_BETA)
    )
    knee_mean_p = float(oof_f[knee].mean()) if knee.any() else float("nan")
    knee_emp = float(truth_f[knee].mean()) if knee.any() else float("nan")
    knee_pass = KNEE_BAND[0] <= knee_mean_p <= KNEE_BAND[1]

    # Plateau: comfortably above the threshold. Inflation = mean p over
    # empirical rate; the gate trips if any populous plateau cell exceeds 2x.
    finite_beta = np.where(np.isfinite(beta_f), beta_f, 0.0)
    table = cell_table(margin_f, finite_beta, oof_f, truth_f)
    plateau_cells = [
        row
        for row in table
        if float(row["margin"].split("-")[0]) >= 0.15 and row["n"] >= 200
    ]
    inflated = [
        row
        for row in plateau_cells
        if row["events"] >= 3 and row["ratio"] is not None
        and row["ratio"] > PLATEAU_INFLATION_LIMIT
    ]
    # Cells with almost no events: judge on absolute mass instead of a ratio
    # against a zero denominator -- mean p above the knee band on a no-event
    # plateau cell is inflation regardless of the undefined ratio.
    silent_inflated = [
        row
        for row in plateau_cells
        if row["events"] < 3 and row["mean_p"] > 0.05
    ]
    plateau_pass = not inflated and not silent_inflated

    out = {
        "knee_cell": {
            "definition": f"margin {KNEE_MARGIN} x beta30>={KNEE_BETA}, decision window = full 42d",
            "n": int(knee.sum()),
            "mean_p": round(knee_mean_p, 4),
            "empirical": round(knee_emp, 4),
            "events": int(truth_f[knee].sum()),
            "band": KNEE_BAND,
            "pass": bool(knee_pass),
        },
        "plateau": {
            "cells": plateau_cells,
            "inflated_cells": inflated,
            "silent_inflated_cells": silent_inflated,
            "limit": PLATEAU_INFLATION_LIMIT,
            "pass": bool(plateau_pass),
        },
        "full_table": table,
        "gate_b_pass": bool(knee_pass and plateau_pass),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2))

    print(json.dumps({k: out[k] for k in ("knee_cell",)}, indent=2))
    print("plateau cells:")
    for row in plateau_cells:
        print(f"  {row}")
    print(f"GATE (b): knee {'PASS' if knee_pass else 'FAIL'} "
          f"(mean_p {knee_mean_p:.3f} in {KNEE_BAND}? empirical {knee_emp:.3f}); "
          f"plateau {'PASS' if plateau_pass else 'FAIL'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
