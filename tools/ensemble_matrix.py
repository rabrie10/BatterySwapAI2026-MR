"""Build the OOF prediction matrix for the ensemble study.

One row per (scenario, battery) on the 19,890-row scenario frame
(outputs/research_rowfeat.parquet), with every decorrelated ranker the
session produced as a column:

    p_cens      incumbent cens GBDT, production 5-fold OOF, calibrated
                (outputs/frame_oof_cal.parquet 'p' == research_rowfeat 'p_cal')
    p_cens_raw  same model, raw level (research_rowfeat 'p')
    p_tp        two-phase changepoint filter p42, fold-dispatched OOF,
                per-fold RemainingCalibration applied inside predict_rows
                (outputs/twophase_model_oof.joblib)
    p_qh        quantile head 42d exceedance, by-building OOF dispatch
                (outputs/qhead_folds.joblib on the transfer-stress scen frame)
    raw_min3 / raw_slope7 / margin / staleness / beta30   raw-channel signals
    p_censhard_<fold>  cens refit p_raw from the transfer-stress hard-holdout
                fits (stride 8 / max_iter 150), one column per hard fold,
                aligned to this frame; held buildings in the sidecar JSON.

Sanity anchors printed at build time (must reproduce the recorded findings):
    cens  pooled AP ~0.308 cal / ~0.267 raw, mid-block top-12 ~0.214
    tp    pooled AP ~0.384, mid-block top-12 ~0.292
    qhead pooled AP ~0.29

Usage:
    OMP_NUM_THREADS=2 python tools/ensemble_matrix.py \
        [--out outputs/ensemble_matrix.parquet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TRANSFER_WORK = Path(
    os.environ.get(
        "TRANSFER_STRESS_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\transfer_stress",
    )
)

HARD_FOLDS = (
    "hard_large5",
    "hard_small10",
    "hard_mosteol5",
    "hard_hirate6",
    "hard_betashift5",
)


def top_k_rate(frame: pd.DataFrame, column: str, low: int, high: int, k: int = 12) -> float:
    rates = []
    for _, rows in frame[(frame.scenario >= low) & (frame.scenario <= high)].groupby(
        "scenario"
    ):
        top = rows.nlargest(k, column)
        rates.append(float(top.due.sum()) / k)
    return float(np.mean(rates)) if rates else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/research_rowfeat.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ensemble_matrix.parquet"))
    parser.add_argument("--sidecar", type=Path, default=Path("outputs/ensemble_matrix_meta.json"))
    args = parser.parse_args()

    started = time.time()
    frame = pd.read_parquet(REPO_ROOT / args.frame)
    n = len(frame)
    print(f"frame: {n} rows, {int(frame.due.sum())} due, "
          f"{frame.scenario.nunique()} scenarios", flush=True)

    matrix = frame[
        [
            "scenario", "battery", "building", "due", "remaining", "margin",
            "staleness", "beta30", "cutoff_ord", "raw_min3", "raw_slope7",
        ]
    ].copy()
    matrix["p_cens_raw"] = frame.p.fillna(0.0).to_numpy(dtype=float)
    matrix["p_cens"] = frame.p_cal.fillna(0.0).to_numpy(dtype=float)

    # ---- two-phase p42 (fold-dispatched OOF, calibration applied inside) ----
    tp_model = joblib.load(REPO_ROOT / "outputs/twophase_model_oof.joblib")
    margin = frame.margin.to_numpy(dtype=float)
    remaining = frame.remaining.to_numpy(dtype=float)
    devices = frame.battery.to_numpy(dtype=object)
    finite = np.isfinite(margin)
    safe_margin = np.where(finite, margin, 9.0)
    p42 = tp_model.predict_rows(safe_margin, remaining, devices, horizons=np.array([42.0]))[:, 0]
    matrix["p_tp"] = np.where(finite, p42, 0.0)
    print(f"twophase p42 done ({time.time()-started:.0f}s)", flush=True)

    # ---- quantile head (by-building OOF dispatch on transfer-stress scen) ----
    prep_path = TRANSFER_WORK / "prep.joblib"
    scen = joblib.load(prep_path)["scen"]
    key = pd.DataFrame(
        {
            "scenario": scen["scenario_index"],
            "battery": scen["device"],
            "row": np.arange(len(scen["due"])),
        }
    )
    qb = joblib.load(REPO_ROOT / "outputs/qhead_folds.joblib")["by_building"]
    has = scen["has_row"]
    h_eff = np.clip(np.minimum(42.0, scen["remaining"]), 0.0, None)
    p_qh = np.zeros(len(scen["due"]), dtype=float)
    for building in np.unique(scen["building"]):
        model = qb.get(str(building))
        if model is None:
            print(f"  WARNING: no qhead fold model for {building}", flush=True)
            continue
        mask = (scen["building"] == building) & has
        if not mask.any():
            continue
        raw = model.probabilities(scen["features"][mask].astype(np.float32), h_eff[mask])
        p_qh[mask] = np.where(h_eff[mask] <= 0.0, 0.0, raw)
    key["p_qh"] = p_qh
    # sanity: scen due must match frame due through the key merge
    merged = matrix.merge(key, on=["scenario", "battery"], how="left", validate="1:1")
    if merged.p_qh.isna().any():
        raise SystemExit(f"qhead join missed {int(merged.p_qh.isna().sum())} rows")
    matrix["p_qh"] = merged.p_qh.to_numpy(dtype=float)
    scen_due = np.zeros(n, dtype=int)
    scen_due[merged.row.to_numpy(dtype=int)] = 0  # placeholder to keep flake quiet
    check = int((scen["due"][merged.row.to_numpy(dtype=int)] != matrix.due.to_numpy()).sum())
    print(f"qhead joined, due mismatches vs scen frame: {check} ({time.time()-started:.0f}s)",
          flush=True)

    # ---- cens hard-holdout refits (transfer-stress records, stride8/it150) ----
    meta: dict = {"hard_folds": {}}
    row_of = merged.row.to_numpy(dtype=int)
    for fold in HARD_FOLDS:
        record = joblib.load(TRANSFER_WORK / "fits" / f"{fold}__cens.joblib")
        p_raw = np.asarray(record["p_raw"], dtype=float)
        matrix[f"p_censhard_{fold}"] = p_raw[row_of]
        meta["hard_folds"][fold] = list(record["heldout"])
    print(f"hard-fold cens refits attached ({time.time()-started:.0f}s)", flush=True)

    out = REPO_ROOT / args.out
    matrix.to_parquet(out, index=False)
    (REPO_ROOT / args.sidecar).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(matrix)} rows x {len(matrix.columns)} cols)")

    # ---- sanity anchors ----
    due = matrix.due.to_numpy(dtype=int)
    for col, label in (
        ("p_cens", "cens cal"),
        ("p_cens_raw", "cens raw"),
        ("p_tp", "twophase p42"),
        ("p_qh", "qhead"),
    ):
        ap = average_precision_score(due, matrix[col].to_numpy())
        mid = top_k_rate(matrix, col, 16, 31)
        opn = top_k_rate(matrix, col, 0, 15)
        late = top_k_rate(matrix, col, 32, 47)
        print(f"  {label:14s} AP {ap:.4f}  top12 open/mid/late "
              f"{opn:.3f}/{mid:.3f}/{late:.3f}")
    for fold in HARD_FOLDS:
        held = matrix.building.isin(meta["hard_folds"][fold]).to_numpy()
        ap = average_precision_score(due[held], matrix[f"p_censhard_{fold}"].to_numpy()[held])
        print(f"  cens-hard {fold:18s} held AP {ap:.4f} (n={int(held.sum())}, "
              f"due={int(due[held].sum())})")
    print(f"total {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
