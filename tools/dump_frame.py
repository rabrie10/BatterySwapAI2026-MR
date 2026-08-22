"""Dump the full per-row scenario-cutoff diagnostic frame.

One row per (scenario, alive battery): raw out-of-fold probability, margin,
staleness, remaining observation, realized label and days-to-EOL. This is the
frame every calibration and threshold decision should be interrogated on.

    python tools/dump_frame.py --out outputs/frame_oof.parquet
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

from bsai.forecaster import HazardForecaster
from bsai.smoothing import SmoothingCache
from bsai.validation import OofHazardModel

_EPOCH = pd.Timestamp("1970-01-01")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    parser.add_argument("--raw", action="store_true", help="strip calibration")
    parser.add_argument("--out", type=Path, default=Path("outputs/frame_oof.parquet"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    for model in bundle["by_building"].values():
        model.volatility_scale = args.volatility_scale
        if args.raw:
            model.calibration = None
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        ),
        calibration=None,
    )

    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    del raw

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    frames: list[pd.DataFrame] = []
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
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
        origin = int((start - _EPOCH) / pd.Timedelta(days=1))
        battery_ids = locs["battery"].astype(str).to_numpy()
        end_time = pd.to_datetime(locs["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        remaining = (
            (end_time.dt.normalize() - start) / pd.Timedelta(days=1)
        ).to_numpy(dtype=float)

        margin = np.full(len(battery_ids), np.nan)
        staleness = np.full(len(battery_ids), np.nan)
        for row, battery in enumerate(battery_ids):
            series = cache.devices.get(battery)
            if series is None:
                continue
            i = min(origin - series.origin, len(series) - 1)
            if i < 0:
                continue
            values = series.smooth_voltage[: i + 1]
            valid = np.flatnonzero(~np.isnan(values))
            if valid.size == 0:
                continue
            margin[row] = values[valid[-1]] - 2.4
            staleness[row] = i - valid[-1]

        eol = not_dead.reindex(battery_ids)
        days_to_eol = ((eol - start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "scenario": index,
                    "battery": battery_ids,
                    "building": [building_of.get(b, "") for b in battery_ids],
                    "p": probability.reindex(battery_ids).to_numpy(),
                    "margin": margin,
                    "staleness": staleness,
                    "remaining": remaining,
                    "days_to_eol": days_to_eol,
                    "due": (pd.notna(eol) & (eol <= start + pd.Timedelta(days=horizon))).to_numpy(),
                }
            )
        )
        print(f"  s_{index}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
