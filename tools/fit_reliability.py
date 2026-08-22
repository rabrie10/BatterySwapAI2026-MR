"""Fit the isotonic reliability calibration, out-of-fold by building.

Measured at scenario cutoffs with the shipped multiplicative calibration, the
model's reliability is anti-monotone at the top: rows predicted above 0.7
realise 0.36 while rows predicted 0.5-0.7 realise 0.41, and everything below
0.35 under-predicts by 1.3-5x. The aggregate count balances (451 predicted vs
454 realised) because the multiplicative factors were fitted to balance it --
the shape is what broke. The planner prices swap-versus-defer per battery, so
the shape is what it acts on.

This replaces the multiplicative correction with an isotonic map from predicted
probability to realised frequency inside three coarse remaining-observation
bands, fitted on raw out-of-fold predictions, with the same fold discipline:
each fold's calibration never sees its own buildings.

    python tools/fit_reliability.py --volatility-scale 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from fit_calibration import block_table, collect

from bsai.calibrate import DwellAdjust, ReliabilityCalibration, RemainingCalibration
from bsai.smoothing import SmoothingCache

_EPOCH = __import__("pandas").Timestamp("1970-01-01")


def attach_evidence(frame, dataset: Path):
    """Add per-row margin and dwell (days since first below 2.45 V)."""
    import pandas as pd
    import json

    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    del raw
    scen = json.loads((dataset / "scenarios.json").read_text())
    starts = {
        i: int((pd.Timestamp(s["start_time"]).normalize() - _EPOCH) / pd.Timedelta(days=1))
        for i, s in enumerate(scen)
    }
    margin = np.full(len(frame), np.nan)
    dwell = np.full(len(frame), np.nan)
    for battery, group in frame.groupby("battery"):
        series = cache.devices.get(battery)
        if series is None:
            continue
        values = series.smooth_voltage
        below = np.flatnonzero(~np.isnan(values) & (values < 2.45))
        first_below = int(below[0]) if below.size else None
        for row in group.itertuples():
            index = min(starts[row.scenario_index] - series.origin, len(series) - 1)
            if index < 0:
                continue
            prefix = values[: index + 1]
            valid = np.flatnonzero(~np.isnan(prefix))
            if valid.size == 0:
                continue
            margin[row.Index] = prefix[valid[-1]] - 2.4
            dwell[row.Index] = (
                index - first_below
                if first_below is not None and first_below <= index
                else -1.0
            )
    frame["margin"] = margin
    frame["dwell"] = dwell
    return frame


def bucket_table(frame, column: str) -> str:
    edges = [0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]
    out = [f"{'bucket':>14}{'n':>7}{'predicted':>11}{'actual':>9}{'ratio':>8}"]
    values = frame[column].to_numpy()
    due = frame.due.to_numpy()
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (values >= low) & (values < high)
        if inside.sum() == 0:
            continue
        predicted = float(values[inside].mean())
        actual = float(due[inside].mean())
        ratio = predicted / max(actual, 1e-9)
        out.append(
            f"[{low:.2f},{high:.2f}){inside.sum():>7}{predicted:>11.4f}{actual:>9.4f}{ratio:>8.2f}"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--model", type=Path, default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v7_folds_reliability.joblib"))
    parser.add_argument("--model-out", type=Path, default=Path("models/v7_wiener_reliability.joblib"))
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=Path("outputs/v10_reliability.json"))
    parser.add_argument(
        "--style",
        choices=("isotonic", "remaining"),
        default="isotonic",
        help="second stage after the dwell adjustment: isotonic reliability map "
        "or the original multiplicative remaining-observation factors",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frame = collect(args.dataset, args.folds, args.volatility_scale)
    frame = attach_evidence(frame, args.dataset)
    print()
    print("=== BEFORE (raw model) ===")
    print(block_table(frame, "predicted"))
    print(bucket_table(frame, "predicted"))

    bundle = joblib.load(args.folds)
    fold_of_building = {
        building: id(model) for building, model in bundle["by_building"].items()
    }
    frame["fold"] = frame.building.map(fold_of_building)

    dwelled = np.empty(len(frame))
    corrected = np.empty(len(frame))
    per_fold_dwell: dict[int, DwellAdjust] = {}
    per_fold: dict[int, ReliabilityCalibration] = {}
    for fold in frame.fold.dropna().unique():
        others = frame[frame.fold != fold]
        dwell_adjust = DwellAdjust.fit(
            others.margin.to_numpy(),
            others.dwell.to_numpy(),
            others.predicted.to_numpy(),
            others.due.to_numpy(),
        )
        per_fold_dwell[fold] = dwell_adjust
        others_dwelled = np.clip(
            others.predicted.to_numpy()
            * dwell_adjust.factor_for(others.margin.to_numpy(), others.dwell.to_numpy()),
            0.0,
            1.0,
        )
        fit_stage = (
            ReliabilityCalibration.fit
            if args.style == "isotonic"
            else RemainingCalibration.fit
        )
        calibration = fit_stage(
            others.remaining.to_numpy(),
            others_dwelled,
            others.due.to_numpy(),
        )
        per_fold[fold] = calibration
        mask = (frame.fold == fold).to_numpy()
        fold_dwelled = np.clip(
            frame.predicted.to_numpy()[mask]
            * dwell_adjust.factor_for(
                frame.margin.to_numpy()[mask], frame.dwell.to_numpy()[mask]
            ),
            0.0,
            1.0,
        )
        dwelled[mask] = fold_dwelled
        if args.style == "isotonic":
            factor = calibration.factor_for(fold_dwelled, frame.remaining.to_numpy()[mask])
        else:
            factor = calibration.factor_for(frame.remaining.to_numpy()[mask])
        corrected[mask] = np.clip(fold_dwelled * factor, 0.0, 1.0)
    frame["dwelled"] = dwelled
    frame["corrected"] = corrected

    print()
    print("=== AFTER DWELL (out-of-fold) ===")
    print(bucket_table(frame, "dwelled"))
    print()
    print("=== AFTER BOTH (out-of-fold dwell + isotonic) ===")
    print(block_table(frame, "corrected"))
    print(bucket_table(frame, "corrected"))

    production_dwell = DwellAdjust.fit(
        frame.margin.to_numpy(),
        frame.dwell.to_numpy(),
        frame.predicted.to_numpy(),
        frame.due.to_numpy(),
    )
    production_dwelled = np.clip(
        frame.predicted.to_numpy()
        * production_dwell.factor_for(frame.margin.to_numpy(), frame.dwell.to_numpy()),
        0.0,
        1.0,
    )
    production = (
        ReliabilityCalibration.fit if args.style == "isotonic" else RemainingCalibration.fit
    )(
        frame.remaining.to_numpy(),
        production_dwelled,
        frame.due.to_numpy(),
    )
    print()
    print("=== production (fitted on everything) ===")
    print(production_dwell.describe())
    print(production.describe())

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    for building, model in bundle["by_building"].items():
        model.volatility_scale = args.volatility_scale
        model.dwell_adjust = per_fold_dwell.get(fold_of_building[building], production_dwell)
        model.calibration = per_fold.get(fold_of_building[building], production)
    joblib.dump(bundle, args.folds_out)

    shipped = joblib.load(args.model)
    shipped.volatility_scale = args.volatility_scale
    shipped.dwell_adjust = production_dwell
    shipped.calibration = production
    joblib.dump(shipped, args.model_out)
    print(f"\nwrote {args.folds_out} and {args.model_out}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    stage = (
        {"curves": [{"xs": list(xs), "ys": list(ys)} for xs, ys in production.curves]}
        if args.style == "isotonic"
        else {"factors": list(production.factors)}
    )
    args.report.write_text(
        json.dumps(
            {
                "style": args.style,
                "edges": list(production.edges),
                "volatility_scale": args.volatility_scale,
                "dwell": {
                    "margin_cap": production_dwell.margin_cap,
                    "edges": list(production_dwell.edges),
                    "factors": list(production_dwell.factors),
                },
                **stage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
