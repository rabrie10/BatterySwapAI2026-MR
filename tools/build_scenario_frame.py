"""Cache the exact population a scenario asks about, once, so ranking is cheap.

Every ranking experiment so far has cost eight to thirteen minutes because it
re-smooths eight and a half million readings and re-runs the forecaster over 48
scenarios. That is the wrong loop for an idea that can be settled by re-scoring
the same rows.

This runs that pipeline once and writes the result: for every (scenario, device)
the forecaster would be asked about, its 64 features, the remaining observation
window, the out-of-fold probability the shipped model assigns, and whether an
EOL record actually landed inside the 42-day window.

The rows come from ``HazardForecaster`` itself rather than from
``build_training_frame``, using the forecaster's own caches and climatology
context after it has run. That matters: the strided training grid and the
scenario grid are different populations and are not interchangeable
(``HANDOVER.md`` trap 2), and rebuilding the scenario grid by hand quietly
dropped 14% of the devices the forecaster actually scores.

    python tools/build_scenario_frame.py --folds outputs/v7_folds.joblib
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

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.features import DeviceView, feature_row
from bsai.forecaster import HazardForecaster
from bsai.shape import align_to
from bsai.validation import OofHazardModel

_EPOCH = pd.Timestamp("1970-01-01")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    for model in bundle["by_building"].values():
        model.volatility_scale = args.volatility_scale
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    features: list[np.ndarray] = []
    meta: list[dict] = []

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        start = pd.Timestamp(scenario["start_time"]).normalize()
        settings = scenario["settings"]
        horizon = int(settings.planning_window_days)
        forecaster.predict(
            cut, locs,
            prediction_origin=start, horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probability = forecaster.last_probabilities
        origin_ordinal = int((start - _EPOCH) / pd.Timedelta(days=1))

        end_time = pd.to_datetime(locs["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        end_time = end_time.dt.normalize()
        end_time.index = locs["battery"].astype(str).to_numpy()
        horizon_end = start + pd.Timedelta(days=horizon)

        # Re-read the same rows the forecaster just scored, from its own caches
        # and its own climatology context, so the features are identical.
        for battery in locs["battery"].astype(str):
            series = forecaster.cache.devices.get(battery)
            if series is None:
                continue
            position = series.index_of(origin_ordinal)
            if position < 0:
                continue
            position = min(position, len(series) - 1)
            view = DeviceView(series.smooth_voltage, series.smooth_temperature)
            shape_view = align_to(
                forecaster.shape_cache.devices.get(battery), series.origin, len(series)
            )
            row = feature_row(
                view, position, series.origin + position, forecaster._context, shape_view
            )
            if row is None:
                continue
            record = not_dead.get(battery, pd.NaT)
            features.append(np.asarray(row, dtype=np.float32))
            meta.append(
                {
                    "scenario_index": index,
                    "battery": battery,
                    "building": building_of.get(battery, ""),
                    "probability": float(probability.get(battery, 0.0)),
                    "remaining": float(
                        (end_time.loc[battery] - start) / pd.Timedelta(days=1)
                    ),
                    "due": bool(pd.notna(record) and record <= horizon_end),
                    "days_to_eol": (
                        float((record - start) / pd.Timedelta(days=1))
                        if pd.notna(record) else np.inf
                    ),
                    "substitute_eol": float(
                        (end_time.loc[battery] - start) / pd.Timedelta(days=1)
                    ) + float(settings.unobserved_eol_days),
                }
            )
        print(f"  {scenario['name']:>5}  rows={len(meta):6d}", flush=True)

    table = pd.DataFrame(meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.vstack(features),
        scenario_index=table.scenario_index.to_numpy(np.int32),
        battery=table.battery.to_numpy(),
        building=table.building.to_numpy(),
        probability=table.probability.to_numpy(),
        remaining=table.remaining.to_numpy(),
        due=table.due.to_numpy(),
        days_to_eol=table.days_to_eol.to_numpy(),
        substitute_eol=table.substitute_eol.to_numpy(),
    )
    n = table.scenario_index.nunique()
    print()
    print(f"{len(table)} rows over {n} scenarios ({len(table)/n:.1f} per scenario)")
    print(f"due {int(table.due.sum())} ({table.due.sum()/n:.2f} per scenario), "
          f"predicted mass {table.probability.sum()/n:.2f} per scenario")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
