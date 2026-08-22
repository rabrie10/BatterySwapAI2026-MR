"""Gate (c): hard-holdout transfer stress for the sequence head.

Reuses the exact grouped "hard" folds of ``tools/transfer_stress.py`` (largest
buildings, smallest ten, most-EOL, highest-rate, biggest beta-scale shift) and
its measurement protocol: refit with the held-out buildings excluded, predict
the 48-scenario deployment rows on held-out buildings only, fit the
RemainingCalibration on the training buildings' rows (production procedure),
and read PR-AUC plus sum-p inflation.

Reference (same protocol, stride 8 / max_iter 150, outputs/transfer_stress.json):

    cens PR-AUC by fold  0.4968 0.6026 0.2879 0.3387 0.4129   mean 0.4278
    cens inflation_cal   0.963  0.576  1.211  0.897  0.757    mean 0.881

Gate: mean PR-AUC >= 0.428 and mean calibrated sum-p inflation <= 1.15.

Fidelity: windows subsampled to stride-8 (every other stride-4 cutoff per
device) and 2 epochs against the OOF run's 3 -- the same direction of
handicap the reference runs took against production fits.

    python tools/seq_transfer.py
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

import torch

from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

from bsai.calibrate import RemainingCalibration
from bsai.seq_head import probability_at
from train_seq import train_net
from transfer_stress import make_hard_folds

DEFAULT_WORK = Path(
    os.environ.get(
        "SEQ_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\seq",
    )
)

CENS_REFERENCE = {
    "hard_large5": {"pr_auc": 0.4968, "inflation_cal": 0.963},
    "hard_small10": {"pr_auc": 0.6026, "inflation_cal": 0.576},
    "hard_mosteol5": {"pr_auc": 0.2879, "inflation_cal": 1.211},
    "hard_hirate6": {"pr_auc": 0.3387, "inflation_cal": 0.897},
    "hard_betashift5": {"pr_auc": 0.4129, "inflation_cal": 0.757},
}
GATE_PR_AUC = 0.428
GATE_INFLATION = 1.15


def within_scenario_rank(scen_index: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)
    for s in np.unique(scen_index):
        mask = scen_index == s
        out[mask] = rankdata(p[mask], method="average") / mask.sum()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_WORK / "seq_pack.joblib")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/seq_transfer.json")
    args = parser.parse_args()

    torch.set_num_threads(3)
    started = time.time()
    pack = joblib.load(args.pack)
    train, deploy = pack["train"], pack["deploy"]
    frame = train["frame"]

    # Stride-8-equivalent cutoff subsample: every other stride-4 row per device.
    keep = np.zeros(len(frame), dtype=bool)
    for device in np.unique(frame.device):
        rows = np.flatnonzero(frame.device == device)
        base = frame.cutoff[rows].min()
        keep[rows[(frame.cutoff[rows] - base) % 8 == 0]] = True
    print(f"stride-8 subsample: {int(keep.sum())}/{len(frame)} cutoffs", flush=True)

    order = np.argsort(train["window_row"], kind="stable")
    w_row = train["window_row"][order]
    w_h = train["window_horizon"][order]
    w_t = train["window_target"][order]
    ptr = np.searchsorted(w_row, np.arange(len(frame) + 1)).astype(np.int64)

    prep = {
        "building_sizes": train["building_sizes"],
        "building_eol": train["building_eol"],
        "frame": frame,
    }
    folds = make_hard_folds(prep)

    n_scen = int(deploy["n_scenarios"])
    due = deploy["due"].astype(int)
    remaining = deploy["remaining"].astype(float)
    margins = deploy["margin"].astype(float)
    h_eff = np.clip(np.minimum(42.0, remaining), 0.0, None)

    results: dict = {"folds": {}, "config": {
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "width": args.width, "stride8_cutoffs": int(keep.sum()),
    }}
    pr_aucs, inflations = [], []

    for fold, heldout in folds.items():
        held_rows = np.isin(deploy["building"], list(heldout))
        train_rows = np.flatnonzero(
            keep & ~np.isin(frame.building, list(heldout)) & (ptr[1:] > ptr[:-1])
        )
        print(f"{fold}: train {train_rows.size} cutoffs, heldout {list(heldout)}", flush=True)
        net = train_net(
            train_rows, train, ptr, w_h, w_t,
            epochs=args.epochs, batch=args.batch, lr=args.lr,
            width=args.width, seed=args.seed, log_every=60,
        )
        p_raw = probability_at(
            net, deploy["windows"], margins, h_eff
        )
        calibration = RemainingCalibration.fit(
            remaining[~held_rows], p_raw[~held_rows], due[~held_rows].astype(float)
        )
        p_cal = np.clip(p_raw * calibration.factor_for(remaining), 0.0, 1.0)

        due_h = due[held_rows]
        realized = float(due_h.sum())
        scen_h = deploy["scenario_index"][held_rows]
        rank_h = within_scenario_rank(scen_h, p_raw[held_rows])
        entry = {
            "heldout": list(heldout),
            "n_rows": int(held_rows.sum()),
            "n_due": int(realized),
            "sum_p_raw_per_scenario": round(float(p_raw[held_rows].sum() / n_scen), 3),
            "sum_p_cal_per_scenario": round(float(p_cal[held_rows].sum() / n_scen), 3),
            "realized_per_scenario": round(realized / n_scen, 3),
            "inflation_raw": round(float(p_raw[held_rows].sum() / max(realized, 1.0)), 3),
            "inflation_cal": round(float(p_cal[held_rows].sum() / max(realized, 1.0)), 3),
            "pr_auc_level": round(
                float(average_precision_score(due_h, p_raw[held_rows])), 4
            ),
            "pr_auc_rank": round(
                float(average_precision_score(due_h, rank_h)), 4
            ),
            "calibration_factors": [round(f, 3) for f in calibration.factors],
            "cens_reference": CENS_REFERENCE.get(fold),
        }
        results["folds"][fold] = entry
        pr_aucs.append(entry["pr_auc_level"])
        inflations.append(entry["inflation_cal"])
        print(
            f"  PR-AUC {entry['pr_auc_level']} (cens {CENS_REFERENCE.get(fold, {}).get('pr_auc')}), "
            f"infl_cal {entry['inflation_cal']} "
            f"({time.time()-started:.0f}s)",
            flush=True,
        )

    mean_pr = float(np.mean(pr_aucs))
    mean_infl = float(np.mean(inflations))
    worst_infl = float(np.max(inflations))
    gate_pr = mean_pr >= GATE_PR_AUC
    gate_infl = mean_infl <= GATE_INFLATION
    results["summary"] = {
        "mean_pr_auc": round(mean_pr, 4),
        "cens_mean_pr_auc": 0.4278,
        "mean_inflation_cal": round(mean_infl, 3),
        "worst_inflation_cal": round(worst_infl, 3),
        "cens_mean_inflation_cal": 0.881,
        "gate_pr_auc_pass": bool(gate_pr),
        "gate_inflation_pass": bool(gate_infl),
        "gate_c_pass": bool(gate_pr and gate_infl),
        "seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["summary"], indent=2))
    print(f"GATE (c): {'PASS' if (gate_pr and gate_infl) else 'FAIL'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
