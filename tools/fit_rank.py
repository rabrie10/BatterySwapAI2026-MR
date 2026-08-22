"""Fit the within-scenario rank -> realised-rate calibration.

Motivation, measured on the public split: absolute probability levels inflate
on unseen buildings (production expected-due 8.45/scenario in-fold against at
least 11.4 deduced on public, realised ~9.5), which silently defeated a
level-based swap budget. Ranks transfer where levels do not, so the level is
re-derived from the rank.

Fits on the calibrated out-of-fold frame (tools/dump_frame.py without --raw).
Note: the rank curve is fitted pooled over all folds' rows; at 60 monotone
rates from ~20k rows it has too little capacity for building leakage to
matter, but it is not strictly out-of-fold.

    python tools/fit_rank.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.calibrate import RankCalibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/frame_oof_cal.parquet"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--model", type=Path, default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v7_folds_rank.joblib"))
    parser.add_argument("--model-out", type=Path, default=Path("models/v7_rank.joblib"))
    parser.add_argument("--max-rank", type=int, default=60)
    args = parser.parse_args()

    frame = pd.read_parquet(args.frame)
    calibration = RankCalibration.fit(
        frame.scenario.to_numpy(),
        frame.p.to_numpy(),
        frame.due.to_numpy(dtype=float),
        max_rank=args.max_rank,
    )
    print(calibration.describe())

    # Reliability of the rank-mapped probability, by scenario third.
    mapped = np.empty(len(frame))
    for scenario, group in frame.groupby("scenario"):
        factor = calibration.factors(group.p.to_numpy())
        mapped[group.index] = np.clip(group.p.to_numpy() * factor, 0.0, 1.0)
    frame = frame.assign(mapped=mapped)
    for low, high, label in [(0, 16, "early"), (16, 32, "mid"), (32, 48, "late")]:
        block = frame[(frame.scenario >= low) & (frame.scenario < high)]
        n = block.scenario.nunique()
        print(
            f"  {label:>5}: mapped {block.mapped.sum()/n:6.2f}/scenario  "
            f"raw {block.p.sum()/n:6.2f}  actual {block.due.sum()/n:6.2f}"
        )

    bundle = joblib.load(args.folds)
    bundle["rank_calibration"] = calibration
    joblib.dump(bundle, args.folds_out)
    shipped = joblib.load(args.model)
    shipped.rank_calibration = calibration
    joblib.dump(shipped, args.model_out)
    print(f"wrote {args.folds_out} and {args.model_out}")


if __name__ == "__main__":
    main()
