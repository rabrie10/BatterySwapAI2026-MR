"""What do the mistakes look like?

Precision at the operating point is 0.31 locally and 0.47 on the leaderboard,
against first place's 0.74. Two thirds of the remaining gap is early cost, and
early cost is wasted swaps. So the question that decides where to spend effort
is not "which model is better in the abstract" but "what distinguishes the
batteries we swap wrongly from the ones we swap rightly".

For every battery above the swap threshold this compares three groups on the
features the model actually uses -- true positives, false positives, and the
false negatives we miss -- and reports how separable they are. A feature that
separates false positives from true positives is a feature worth adding; one
that does not is not, however plausible it sounds.

    python tools/error_profile.py
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

from bsai.features import FEATURE_NAMES, DeviceView, feature_row
from bsai.forecaster import HazardForecaster
from bsai.shape import align_to
from bsai.validation import OofHazardModel

_EPOCH = pd.Timestamp("1970-01-01")

# The features most likely to carry a per-device or knee signal, plus the level
# and slope that dominate the ranking today.
WATCH = [
    "voltage",
    "voltage_compensated",
    "slope_30",
    "slope_90",
    "slope_180",
    "knee_slope_vs_history",
    "knee_recent_vs_baseline",
    "knee_worst_14d_drop",
    "beta_30",
    "beta_rise",
    "v_range_30",
    "v_range_rise",
    "age_days",
    "staleness",
    "temp_outlook_42",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--threshold", type=float, default=0.26)
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_errors.json"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
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
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}

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
        due_index = not_dead[
            not_dead.notna() & (not_dead <= start + pd.Timedelta(days=horizon))
        ].index
        due = set(due_index)
        origin_ordinal = int((start - _EPOCH) / pd.Timedelta(days=1))

        for battery in locs["battery"].astype(str):
            p = float(probability.get(battery, 0.0))
            is_due = battery in due
            if p < args.threshold and not is_due:
                continue  # neither swapped nor missed: not a decision we got wrong
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
            record = {"p": p, "due": is_due, "swapped": p >= args.threshold}
            for name in WATCH:
                record[name] = float(values[columns[name]])
            rows.append(record)
        print(f"  {scenario['name']:>5}", flush=True)

    frame = pd.DataFrame(rows)
    tp = frame[frame.swapped & frame.due]
    fp = frame[frame.swapped & ~frame.due]
    fn = frame[~frame.swapped & frame.due]
    print()
    print(f"true positives {len(tp)}   false positives {len(fp)}   missed {len(fn)}")
    print(f"precision {len(tp)/max(len(tp)+len(fp),1):.3f}   "
          f"recall {len(tp)/max(len(tp)+len(fn),1):.3f}")
    print()
    print("=== median by group, and how separable FP is from TP ===")
    print(f"{'feature':>26}{'TP':>11}{'FP':>11}{'missed':>11}{'FP vs TP AUC':>14}")
    separability = {}
    for name in WATCH:
        a, b = tp[name].dropna(), fp[name].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        # rank-based separability: 0.5 means the feature cannot tell them apart
        combined = np.concatenate([a.to_numpy(), b.to_numpy()])
        ranks = pd.Series(combined).rank().to_numpy()
        auc = (ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        auc = max(auc, 1 - auc)
        separability[name] = float(auc)
        print(
            f"{name:>26}{a.median():>11.4f}{b.median():>11.4f}"
            f"{fn[name].median():>11.4f}{auc:>14.3f}"
        )
    print()
    best = sorted(separability.items(), key=lambda kv: -kv[1])[:5]
    print("most separable FP from TP:", ", ".join(f"{k} {v:.3f}" for k, v in best))
    print("(0.5 = useless; the model already uses all of these, so a high value")
    print(" means the signal is present but under-weighted, not missing)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "true_positives": len(tp),
                "false_positives": len(fp),
                "missed": len(fn),
                "separability": separability,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
