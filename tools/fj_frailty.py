"""Survival surprise: what has this device already survived, by V8's own reckoning?

V8 is memoryless. At every cutoff it reads the current state and returns a
probability; nothing carries forward the fact that it assigned the same device a
high hazard last week, and the week before, and was wrong both times. If there is
device-level frailty -- persistent individual heterogeneity the population
dynamics do not capture -- then the accumulated hazard a device has *survived* is
a causal, legitimate posterior update, and it is exactly the quantity V8 throws
away.

Construction, one device at a time and strictly from its own past telemetry:

* weekly pseudo-cutoffs across the device's whole history, non-overlapping so no
  interval of risk is counted twice;
* at each past cutoff the frozen V8 fold model that never saw this device's
  building, evaluated at a **7-day** horizon;
* the device is observably active at the scenario cutoff, so every completed
  weekly interval before it is a survival observation;
* `H_surv = sum over completed weeks of -log(1 - p7(s))`, the cumulative baseline
  hazard survived.

**The probability used is the raw first-passage value, before
`RemainingCalibration`.** That correction is keyed on `end_time - cutoff`, and
`end_time` is the dataset export date -- a fact about the future at a historical
pseudo-cutoff, even though the deployed model legitimately receives it inside a
scenario. Using the uncalibrated passage probability keeps every input to
`H_surv` a function of past telemetry alone.

No device identity, no EOL label, no scenario outcome, and no overlapping
horizons enter any feature here.

    python tools/fj_frailty.py --report outputs/fj_frailty.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices  # noqa: E402

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.hazard import HORIZON_GRID, build_training_frame  # noqa: E402
from bsai.shape import ShapeCache  # noqa: E402
from bsai.smoothing import SmoothingCache  # noqa: E402
from tools.fj_frame import load_frame  # noqa: E402
from tools.fj_templates import cutoff_index  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
WEEK = 7
HIGH = (0.1, 0.2, 0.3)
RECENT_WEEKS = (4, 8, 12, 26)

NAMES = (
    "H_surv",
    "H_surv_log1p",
    "weeks_over_10",
    "weeks_over_20",
    "weeks_over_30",
    "max_prior_p7",
    "consecutive_high_weeks",
    "H_last_4",
    "H_last_8",
    "H_last_12",
    "H_last_26",
    "weeks_since_first_high",
    "weeks_observed",
)


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def weekly_hazard(dataset: Path, folds: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per device: the weekly cutoff grid and the raw out-of-fold p7 at each."""
    started = time.time()
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]

    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw

    eol_index: dict[str, int | None] = {}
    observation_index: dict[str, int] = {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = (
            None if pd.isna(moment) else _ordinal(moment) - series.origin
        )
        end = observation_end.get(device_id)
        observation_index[device_id] = (
            (series.origin + len(series) - 1)
            if pd.isna(end)
            else _ordinal(end) - series.origin
        )

    frame = build_training_frame(
        cache, eol_index, building_of, observation_index,
        shape_cache=shape_cache, stride=WEEK,
    )
    print(f"  {frame.features.shape[0]} weekly pseudo-cutoffs "
          f"({time.time() - started:.0f}s)", flush=True)

    bundle = joblib.load(folds)
    column = list(HORIZON_GRID).index(7)
    p7 = np.zeros(frame.features.shape[0])
    for building in np.unique(frame.building):
        model = bundle["by_building"].get(building)
        if model is None:
            continue
        mask = frame.building == building
        rows = int(mask.sum())
        # The raw passage probability at 7 days: no RemainingCalibration, which
        # would need end_time and so would not be causal at a past cutoff.
        p7[mask] = model.probabilities(
            frame.features[mask], np.full(rows, 7.0, dtype=np.float32)
        )
    print(f"  raw out-of-fold p7 computed, median {np.median(p7):.5f}, "
          f"max {p7.max():.4f} ({time.time() - started:.0f}s)", flush=True)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device in np.unique(frame.device):
        rows = np.flatnonzero(frame.device == device)
        order = rows[np.argsort(frame.cutoff[rows])]
        out[str(device)] = (frame.cutoff[order].astype(int), p7[order])
    return out


def survival_features(
    cutoffs: np.ndarray, p7: np.ndarray, now: int
) -> list[float]:
    """Everything the device has already survived, as of day ``now``.

    Only weeks whose seven-day interval *completed* strictly before ``now`` are
    counted, so the intervals tile the past without overlapping and no risk is
    counted twice.
    """
    done = cutoffs + WEEK <= now
    if not done.any():
        return [np.nan] * len(NAMES)
    past = np.clip(p7[done], 0.0, 1.0 - 1e-9)
    hazard = -np.log(1.0 - past)
    cumulative = float(hazard.sum())

    out = [cumulative, float(np.log1p(cumulative))]
    for level in HIGH:
        out.append(float((past > level).sum()))
    out.append(float(past.max()))

    high = past > HIGH[0]
    run = 0
    for value in high[::-1]:
        if not value:
            break
        run += 1
    out.append(float(run))

    for weeks in RECENT_WEEKS:
        out.append(float(hazard[-weeks:].sum()) if hazard.size else 0.0)

    first = np.flatnonzero(high)
    out.append(float(high.size - first[0]) if first.size else 0.0)
    out.append(float(past.size))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_frailty.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_frailty.json"))
    args = parser.parse_args()

    started = time.time()
    hazard = weekly_hazard(args.dataset, args.folds)
    frame = load_frame(args.frame)

    from tools.fj_terminality import load_series

    series = load_series(args.series)
    cutoff = cutoff_index(frame, series, args.dataset)

    out = np.full((frame.features.shape[0], len(NAMES)), np.nan)
    for row in range(frame.features.shape[0]):
        entry = hazard.get(str(frame.battery[row]))
        if entry is None or cutoff[row] < 0:
            continue
        out[row] = survival_features(entry[0], entry[1], int(cutoff[row]))

    coverage = np.isfinite(out).mean(axis=0)
    print()
    print("coverage and spread per survival feature:")
    for index, name in enumerate(NAMES):
        column = out[:, index]
        good = column[np.isfinite(column)]
        print(f"  {name:>24} {coverage[index]:6.1%}  median {np.median(good):9.3f}  "
              f"p90 {np.percentile(good, 90):9.3f}  max {good.max():10.3f}")

    np.savez_compressed(args.out, features=out.astype(np.float32),
                        names=np.asarray(NAMES))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "names": list(NAMES),
        "coverage": {n: round(float(c), 4) for n, c in zip(NAMES, coverage)},
    }, indent=1))
    print(f"\nwrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
