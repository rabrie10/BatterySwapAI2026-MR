"""Clamp the remaining-observation calibration factors at <= cap.

The boost side (1.71x / 2.34x at long remaining observation) balances the
aggregate due count by inflating mid-probability batteries into the top
bucket, where reliability measures 0.36 realized against 0.92 predicted.
This control keeps the shrink side (which changes decisions in the closing
scenarios) and removes the boost side, to price what the boost actually
contributes to decisions rather than to a count statistic.

    python tools/clamp_calibration.py --cap 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v7_folds_clamped.joblib"))
    parser.add_argument("--cap", type=float, default=1.0)
    args = parser.parse_args()

    bundle = joblib.load(args.folds)
    seen = set()
    for building, model in bundle["by_building"].items():
        calibration = getattr(model, "calibration", None)
        if calibration is None or id(model) in seen:
            continue
        seen.add(id(model))
        before = tuple(calibration.factors)
        calibration.factors = tuple(min(f, args.cap) for f in before)
        print(f"{building}: {before} -> {calibration.factors}")
    joblib.dump(bundle, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
