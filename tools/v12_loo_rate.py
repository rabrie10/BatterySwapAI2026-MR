"""Rate-at-rank from the pooled-LOO harness predictions, per scenario third.

The 0.214 mid-block top-12 number in docs/V11_TRANSFER_FINDINGS.md was 5-fold
OOF; this computes the same table under the stricter leave-one-building-out
regime for every arm and incumbent with stored fits, so the comparison is
apples-to-apples and fully out-of-building. Calibrated per-fold with the
production RemainingCalibration procedure, exactly as ts.pooled_loo does.

    python tools/v12_loo_rate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import transfer_stress as ts
import v12_transfer as vt

from bsai.calibrate import RemainingCalibration

BLOCKS = ((0, 16, "early"), (16, 32, "mid"), (32, 48, "late"))
TOP_KS = (5, 12, 15)


def pooled_loo_pcal(scen, fits, variant, loo_folds):
    p_cal = np.full(len(scen["due"]), np.nan)
    for fold, heldout in loo_folds.items():
        record = fits.get((fold, variant))
        if record is None:
            return None
        held = np.isin(scen["building"], heldout)
        train = ~held
        calibration = RemainingCalibration.fit(
            scen["remaining"][train],
            record["p_raw"][train].astype(float),
            scen["due"][train].astype(float),
        )
        p_cal[held] = np.clip(
            record["p_raw"][held].astype(float)
            * calibration.factor_for(scen["remaining"][held]),
            0.0,
            1.0,
        )
    return None if np.isnan(p_cal).any() else p_cal


def rate_table(scen, p):
    ordered_due = {}
    for s in np.unique(scen["scenario_index"]):
        rows = np.flatnonzero(scen["scenario_index"] == s)
        order = rows[np.argsort(-p[rows], kind="stable")]
        ordered_due[int(s)] = scen["due"][order].astype(float)
    out = {}
    for lo, hi, label in BLOCKS:
        members = [d for s, d in ordered_due.items() if lo <= s < hi]
        out[label] = {
            f"top_{k}": round(float(np.mean([d[:k].mean() for d in members])), 3)
            for k in TOP_KS
        }
    out["pooled"] = {
        f"top_{k}": round(
            float(np.mean([d[:k].mean() for d in ordered_due.values()])), 3
        )
        for k in TOP_KS
    }
    return out


def main() -> None:
    prep = joblib.load(vt.DEFAULT_WORK / "v12_prep.joblib")
    scen = prep["scen"]
    fits = vt.load_fits(vt.DEFAULT_WORK, prep)
    loo_folds = ts.make_loo_folds(prep)

    results = {}
    for variant in ("v12b", "v12b_k1", "v12b_k3", "v12b_noany", "v12b_noraw", "v12", "v7", "cens"):
        p_cal = pooled_loo_pcal(scen, fits, variant, loo_folds)
        if p_cal is None:
            continue
        results[variant] = rate_table(scen, p_cal)

    print(json.dumps(results, indent=2))
    out = REPO_ROOT / "outputs/v12_loo_rate_at_rank.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
