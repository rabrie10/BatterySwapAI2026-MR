"""Gate (c): does the quantile head survive hard grouped building holdouts?

Reuses the transfer-stress harness's checkpointed prep (stride-8 frame,
scenario frame, censor-aware window bank, hard fold definitions) and its
stored cens fits, so the comparison is on byte-identical held-out rows.

Per hard fold: fit the quantile head on the cens-target windows of the
training buildings (max_iter 150, the harness's fidelity), predict the 42-day
probability on the held-out buildings' scenario rows, fit the production-style
RemainingCalibration on the training buildings' rows, and measure PR-AUC and
sum-p inflation. Gate: mean hard-holdout PR-AUC >= cens's 0.428 and calibrated
sum-p inflation <= x1.15.

    python tools/qhead_transfer.py
    python tools/qhead_transfer.py --aux    # with the fixed-horizon 14/28 sets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from sklearn.metrics import average_precision_score

from bsai.calibrate import RemainingCalibration
from bsai.quantile_head import QuantileHeadModel

# The harness's own fold definitions, target derivation and work location.
from transfer_stress import DEFAULT_WORK, make_hard_folds, variant_rows

CENS_REFERENCE_PR_AUC = 0.428  # docs/V11_TRANSFER_FINDINGS.md, 5 hard holdouts


def predict_scenario(model: QuantileHeadModel, scen: dict) -> np.ndarray:
    """42-day probability per scenario row: min(42, remaining) horizon."""
    p = np.zeros(len(scen["due"]), dtype=float)
    mask = scen["has_row"]
    h_eff = np.clip(np.minimum(42.0, scen["remaining"][mask]), 0.0, None)
    raw = model.probabilities(scen["features"][mask].astype(np.float32), h_eff)
    p[mask] = np.where(h_eff <= 0.0, 0.0, raw)
    return p


def holdout_metrics(scen: dict, p_raw: np.ndarray, heldout: list[str]) -> dict:
    held = np.isin(scen["building"], heldout)
    train = ~held
    n_scen = scen["n_scenarios"]
    calibration = RemainingCalibration.fit(
        scen["remaining"][train], p_raw[train], scen["due"][train].astype(float)
    )
    p_cal = np.clip(p_raw * calibration.factor_for(scen["remaining"]), 0.0, 1.0)

    due_h = scen["due"][held].astype(int)
    p_raw_h, p_cal_h = p_raw[held], p_cal[held]
    realized = float(due_h.sum())
    out = {
        "n_rows": int(held.sum()),
        "n_due": int(realized),
        "sum_p_raw_per_scenario": round(float(p_raw_h.sum() / n_scen), 3),
        "sum_p_cal_per_scenario": round(float(p_cal_h.sum() / n_scen), 3),
        "realized_per_scenario": round(realized / n_scen, 3),
        "inflation_raw": round(float(p_raw_h.sum() / max(realized, 1.0)), 3),
        "inflation_cal": round(float(p_cal_h.sum() / max(realized, 1.0)), 3),
    }
    if realized > 0:
        out["pr_auc"] = round(float(average_precision_score(due_h, p_raw_h)), 4)
        order = np.argsort(-p_raw_h, kind="stable")
        for k in (5, 10, 15):
            if due_h.size >= k:
                out[f"precision_at_{k}"] = round(float(due_h[order[:k]].mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--aux", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/qhead_transfer.json")
    )
    args = parser.parse_args()

    started = time.time()
    prep = joblib.load(args.work / "prep.joblib")
    bank, scen = prep["bank"], prep["scen"]
    folds = make_hard_folds(prep)
    rows, target = variant_rows(bank, "cens")
    print(
        f"prep loaded: {int(rows.sum())} cens windows, "
        f"{len(scen['due'])} scenario rows, {len(folds)} hard folds",
        flush=True,
    )

    fits_dir = args.work / "fits"
    results: dict = {
        "config": {"max_iter": args.max_iter, "aux": bool(args.aux)},
        "folds": {},
    }
    params = {"max_iter": args.max_iter}

    for fold, heldout in folds.items():
        train = rows & ~np.isin(bank["building"], list(heldout))
        fit_start = time.time()
        model = QuantileHeadModel.fit(
            bank["design"][train], target[train], prep["climatology"], params=params
        )
        if args.aux:
            aux = {}
            horizon_column = bank["design"][:, -1]
            for aux_horizon in (14, 28):
                subset = train & (horizon_column == aux_horizon)
                aux[aux_horizon] = QuantileHeadModel.fit_aux(
                    bank["design"][subset, :-1], target[subset], params=params
                )
            model.aux = aux
        p_qhead = predict_scenario(model, scen)
        entry = {
            "heldout": list(heldout),
            "n_train_windows": int(train.sum()),
            "fit_seconds": round(time.time() - fit_start, 1),
            "qhead": holdout_metrics(scen, p_qhead, list(heldout)),
        }
        cens_path = fits_dir / f"{fold}__cens.joblib"
        if cens_path.exists():
            record = joblib.load(cens_path)
            entry["cens"] = holdout_metrics(
                scen, record["p_raw"].astype(float), list(heldout)
            )
        results["folds"][fold] = entry
        print(
            f"  {fold}: qhead PR-AUC {entry['qhead'].get('pr_auc')} "
            f"(cens {entry.get('cens', {}).get('pr_auc')}), "
            f"infl cal x{entry['qhead']['inflation_cal']} "
            f"(cens x{entry.get('cens', {}).get('inflation_cal')}), "
            f"{time.time() - started:.0f}s",
            flush=True,
        )

    def mean_of(variant: str, key: str) -> float | None:
        values = [
            entry[variant][key]
            for entry in results["folds"].values()
            if variant in entry and key in entry[variant]
        ]
        return round(float(np.mean(values)), 4) if values else None

    summary = {
        "qhead_mean_pr_auc": mean_of("qhead", "pr_auc"),
        "cens_mean_pr_auc_same_rows": mean_of("cens", "pr_auc"),
        "cens_reference_pr_auc": CENS_REFERENCE_PR_AUC,
        "qhead_mean_inflation_raw": mean_of("qhead", "inflation_raw"),
        "qhead_mean_inflation_cal": mean_of("qhead", "inflation_cal"),
        "cens_mean_inflation_raw": mean_of("cens", "inflation_raw"),
        "cens_mean_inflation_cal": mean_of("cens", "inflation_cal"),
    }
    summary["pass_pr_auc"] = bool(
        (summary["qhead_mean_pr_auc"] or 0.0) >= CENS_REFERENCE_PR_AUC
    )
    summary["pass_inflation"] = bool(
        (summary["qhead_mean_inflation_cal"] or 9.9) <= 1.15
    )
    results["summary"] = summary
    print(json.dumps(summary, indent=2), flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.report} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
