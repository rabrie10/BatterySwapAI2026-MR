"""Is the signal present but under-weighted, or genuinely absent?

The error profile says a false positive is an older device falling slowly with
no acceleration against its own history and no rise in within-day resistance --
and that every one of those features is already in the model. If that is right,
a thin layer fitted directly against the outcome should beat the first-passage
probability on its own, because the passage formula compresses everything into
(margin, drift, sigma) and cannot express "this drift estimate is the kind that
tends to be wrong".

This measures that before anything is wired into the planner. It collects the
scenario-cutoff population once, then compares the raw probability against a
low-capacity stack on out-of-fold folds grouped by building. Logistic, not
boosted: there are only about 450 events in this population and a flexible model
would fit the fold rather than the phenomenon.

    python tools/stack_probe.py
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.features import FEATURE_NAMES, DeviceView, feature_row
from bsai.forecaster import HazardForecaster
from bsai.shape import align_to
from bsai.validation import OofHazardModel

_EPOCH = pd.Timestamp("1970-01-01")

# Chosen from tools/error_profile.py by how well each separates a false positive
# from a true one, not by intuition.
STACK_FEATURES = [
    "age_days",
    "slope_30",
    "knee_slope_vs_history",
    "beta_rise",
    "voltage_compensated",
    "beta_30",
    "temp_outlook_42",
    "v_range_30",
]


def collect(dataset: Path, folds: Path) -> pd.DataFrame:
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(folds)
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    locations, timeseries, eol_times, scenarios = load_dataset(dataset)
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    rows: list[dict] = []

    for scenario, locs, cut, not_dead in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        start = pd.Timestamp(scenario["start_time"]).normalize()
        horizon = int(scenario["settings"].planning_window_days)
        forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probability = forecaster.last_probabilities
        due = set(
            not_dead[
                not_dead.notna() & (not_dead <= start + pd.Timedelta(days=horizon))
            ].index
        )
        origin_ordinal = int((start - _EPOCH) / pd.Timedelta(days=1))

        for battery in locs["battery"].astype(str):
            series = forecaster.cache.devices.get(battery)
            if series is None:
                continue
            position = min(series.index_of(origin_ordinal), len(series) - 1)
            if position < 0:
                continue
            view = DeviceView(series.smooth_voltage, series.smooth_temperature)
            shape = align_to(
                forecaster.shape_cache.devices.get(battery), series.origin, len(series)
            )
            values = feature_row(
                view, position, series.origin + position, forecaster._context, shape
            )
            if values is None:
                continue
            record = {
                "building": building_of.get(battery, ""),
                "p": float(probability.get(battery, 0.0)),
                "due": int(battery in due),
            }
            for name in STACK_FEATURES:
                record[name] = float(values[columns[name]])
            rows.append(record)
        print(f"  {scenario['name']:>5}", flush=True)
    return pd.DataFrame(rows)


def precision_at(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argsort(-scores)[:k]
    return float(labels[top].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/v8_stack.parquet"))
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_stack_probe.json"))
    args = parser.parse_args()

    if args.cache.exists():
        frame = pd.read_parquet(args.cache)
        print(f"reusing {args.cache} ({len(frame)} rows)")
    else:
        frame = collect(args.dataset, args.folds)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.cache)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    for name in STACK_FEATURES:
        frame[name] = frame[name].fillna(frame[name].median())
    labels = frame["due"].to_numpy()
    raw = frame["p"].to_numpy()

    # Out-of-fold by building, matching how the model itself is validated.
    buildings = frame["building"].to_numpy()
    unique = sorted(set(buildings))
    assignment = {b: i % 5 for i, b in enumerate(unique)}
    fold = np.array([assignment[b] for b in buildings])

    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    design = np.column_stack([logit] + [frame[n].to_numpy() for n in STACK_FEATURES])
    mean, scale = design.mean(axis=0), design.std(axis=0) + 1e-9
    design = (design - mean) / scale

    stacked = np.zeros(len(frame))
    for f in range(5):
        train, test = fold != f, fold == f
        model = LogisticRegression(max_iter=2000, C=0.5)
        model.fit(design[train], labels[train])
        stacked[test] = model.predict_proba(design[test])[:, 1]

    # A control: the same stack given only the probability, so any gain from the
    # extra features is separated from the gain of simply refitting the level.
    only = np.zeros(len(frame))
    for f in range(5):
        train, test = fold != f, fold == f
        model = LogisticRegression(max_iter=2000, C=0.5)
        model.fit(design[train][:, :1], labels[train])
        only[test] = model.predict_proba(design[test][:, :1])[:, 1]

    print()
    print(f"rows {len(frame)}   events {labels.sum()}   base rate {labels.mean():.4f}")
    print()
    print(f"{'':>22}{'AUC':>9}{'PR-AUC':>9}{'p@250':>9}{'p@500':>9}{'p@750':>9}")
    results = {}
    for label, scores in [
        ("wiener (raw)", raw),
        ("relevel only", only),
        ("stacked", stacked),
    ]:
        row = dict(
            auc=roc_auc_score(labels, scores),
            pr_auc=average_precision_score(labels, scores),
            p250=precision_at(scores, labels, 250),
            p500=precision_at(scores, labels, 500),
            p750=precision_at(scores, labels, 750),
        )
        results[label] = row
        print(
            f"{label:>22}{row['auc']:>9.4f}{row['pr_auc']:>9.4f}"
            f"{row['p250']:>9.3f}{row['p500']:>9.3f}{row['p750']:>9.3f}"
        )
    print()
    gain = results["stacked"]["pr_auc"] - results["wiener (raw)"]["pr_auc"]
    print(f"PR-AUC gain from the extra features: {gain:+.4f}")
    print("A gain here means the signal was present and under-weighted.")
    print("No gain means the passage model already extracts it, and 2d is dead.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
