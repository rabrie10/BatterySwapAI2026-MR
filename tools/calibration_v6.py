"""Reliability of the forecast at scenario cutoffs, which is where it is used.

Training cutoffs and scenario cutoffs are not the same population: training
samples every device every few days along its whole life, while a scenario asks
about every alive device on one date. A model can look well calibrated on the
first and be badly off on the second, and the second is what the planner acts
on.

No planning happens here, so this is cheap enough to run on every model change.

    python tools/calibration_v6.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v6_folds.joblib"))
    parser.add_argument("--model", type=Path, default=Path("models/v6_hazard.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/v6_calibration.json"))
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--by-building", action="store_true",
                        help="report the shift fold by fold, not just pooled")
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    if args.production:
        forecaster = HazardForecaster(joblib.load(args.model))
    else:
        bundle = joblib.load(args.folds)
        forecaster = HazardForecaster(
            OofHazardModel(
                by_building=bundle["by_building"],
                building_of=building_of,
                climatology=bundle["climatology"],
            )
        )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    for scenario, locs, cut, not_dead in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        start = pd.Timestamp(scenario["start_time"])
        horizon = int(scenario["settings"].planning_window_days)
        horizon_end = start + pd.Timedelta(days=horizon)
        forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probabilities = forecaster.last_probabilities
        due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)
        for battery, probability in probabilities.items():
            rows.append(
                {
                    "scenario": scenario["name"],
                    "battery": battery,
                    "probability": float(probability),
                    "due": battery in due,
                    "building": building_of.get(battery, ""),
                }
            )
        print(
            f"  {scenario['name']:>5}  predicted={probabilities.sum():6.2f}  "
            f"actual={len(due):3d}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    report = _reliability(frame)
    if args.by_building:
        # A single shrink factor is only defensible if every held-out building
        # set is biased the same way. If the sign flips between folds, the level
        # simply does not transfer and no scalar correction is honest.
        per_building = frame.groupby("building").agg(
            predicted=("probability", "sum"), actual=("due", "sum")
        )
        per_building["ratio"] = per_building["predicted"] / per_building["actual"].clip(lower=1)
        report["by_building"] = {
            str(index): {
                "predicted": round(float(row.predicted), 1),
                "actual": int(row.actual),
                "ratio": round(float(row.ratio), 2),
            }
            for index, row in per_building.sort_values("actual", ascending=False).iterrows()
        }
    print()
    print(json.dumps(report, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))


def _reliability(frame: pd.DataFrame) -> dict:
    edges = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]
    frame = frame.copy()
    frame["bucket"] = pd.cut(frame["probability"], edges, right=False)
    grouped = frame.groupby("bucket", observed=True).agg(
        n=("due", "size"), predicted=("probability", "mean"), actual=("due", "mean")
    )
    grouped["due_count"] = frame.groupby("bucket", observed=True)["due"].sum()
    buckets = [
        {
            "range": str(index),
            "n": int(row.n),
            "predicted": round(float(row.predicted), 4),
            "actual": round(float(row.actual), 4),
            "ratio": round(float(row.predicted / row.actual), 2)
            if row.actual > 0
            else None,
            "due_count": int(row.due_count),
        }
        for index, row in grouped.iterrows()
    ]
    per_scenario = frame.groupby("scenario").agg(
        predicted=("probability", "sum"), actual=("due", "sum")
    )
    return {
        "n_rows": int(len(frame)),
        "total_predicted": round(float(frame["probability"].sum()), 1),
        "total_actual": int(frame["due"].sum()),
        "overall_ratio": round(
            float(frame["probability"].sum() / max(frame["due"].sum(), 1)), 3
        ),
        "mean_predicted_per_scenario": round(float(per_scenario["predicted"].mean()), 2),
        "mean_actual_per_scenario": round(float(per_scenario["actual"].mean()), 2),
        "buckets": buckets,
    }


if __name__ == "__main__":
    main()
