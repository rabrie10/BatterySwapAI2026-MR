"""Does the volatility ratio predict the *increments*, not the deaths?

The gate before any dynamic-Wiener variant is built. V8's advantage over every
classifier this project has tried is sample efficiency: drift and scatter are
learned from hundreds of thousands of observed voltage windows rather than from
82 EOL events. So if `std_ratio_30_180` is real health-state information, it
should show up in the quantity those regressors actually predict --

    drop(h)  = margin(t) - margin(t + h)          the drift target
    |resid|  = |drop - E[drop]|                   the scatter target

-- measured out of fold by *building*, on the increment population. If it does
not improve those, there is nothing for the first-passage law to convert and no
planner run is warranted.

This matters more than it sounds. `docs/task1_investigation_findings.md`
measures within-device 42-day volatility at about 0.041 V against roughly
0.021 V of drift: **crossing is a noise-driven event**, so sigma is the dominant
term in the passage formula and the scatter model is where a real dynamics
signal would pay.

Fidelity follows `tools/transfer_stress.py`: stride 8, max_iter 150. The
comparison is base against base-plus-one-column on identical rows and folds, so
the absolute level does not have to match production.

    python tools/fj_increments.py --stride 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices  # noqa: E402

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.hazard import build_training_frame  # noqa: E402
from bsai.shape import ShapeCache  # noqa: E402
from bsai.smoothing import SmoothingCache  # noqa: E402
from bsai.terminality import std_ratio  # noqa: E402
from bsai.wiener import FIT_HORIZONS, build_increment_targets  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
VOLTAGE = FEATURE_NAMES.index("voltage")


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def increment_targets(frame, cache, horizons, *, censor_aware: bool):
    """Design, target and the frame row each window came from, in one pass.

    ``censor_aware`` is V10's correction (commit 13640b7): a window may *end*
    past the crossing, and one that contains the crossing counts at least the
    full margin. Excluding those windows -- which is what V8 ships -- censors
    exactly the steepest observed drops out of the fit, so near the barrier the
    surviving training population is the batteries that did *not* cross. Any
    question about near-barrier dynamics asked on the V8 population is therefore
    conditioned on survival, which is the wrong conditioning for this gate.
    """
    from bsai.margin import EOL_THRESHOLD

    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")
    designs, drops, indices = [], [], []
    for horizon in horizons:
        rows, values = [], []
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            margin = margins.get(device_id)
            start = stop
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            if censor_aware:
                usable = (ends <= last) & (cutoffs >= 0)
            else:
                limit = last if crossing < 0 else min(last, crossing - 1)
                usable = (ends <= limit) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here, there = margin[cutoffs[usable]], margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            fall = here[finite] - there[finite]
            if censor_aware and crossing >= 0:
                crossed = (cutoffs[usable][finite] < crossing) & (
                    ends[usable][finite] >= crossing
                )
                fall = np.where(crossed, np.maximum(fall, here[finite]), fall)
            rows.append(chosen[finite])
            values.append(fall)
        if not rows:
            continue
        index = np.concatenate(rows)
        designs.append(
            np.hstack([
                frame.features[index],
                np.full((index.size, 1), horizon, dtype=np.float32),
            ])
        )
        drops.append(np.concatenate(values))
        indices.append(index)
    return (
        np.vstack(designs), np.concatenate(drops), np.concatenate(indices)
    )


def build(dataset: Path, stride: int):
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
    print(f"  smoothed {len(cache.devices)} devices, {time.time() - started:.0f}s",
          flush=True)

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
        shape_cache=shape_cache, stride=stride,
    )
    print(f"  {frame.features.shape[0]} cutoffs, {time.time() - started:.0f}s",
          flush=True)

    # The volatility ratio at every cutoff, from the same smoothed grid the
    # features come from, reading nothing past the cutoff.
    ratio = np.full(frame.features.shape[0], np.nan)
    for device_id, series in cache.devices.items():
        rows = np.flatnonzero(frame.device == device_id)
        if rows.size == 0:
            continue
        voltage = series.smooth_voltage
        for row in rows:
            ratio[row] = std_ratio(voltage, int(frame.cutoff[row]))
    print(f"  volatility ratio on {np.isfinite(ratio).mean():.1%} of cutoffs, "
          f"{time.time() - started:.0f}s", flush=True)
    return frame, cache, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--censor-aware", action="store_true",
                        help="V10's target: windows may end past the crossing and "
                             "a crossing window counts at least the full margin")
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_increments.json"))
    args = parser.parse_args()

    frame, cache, ratio = build(args.dataset, args.stride)

    started = time.time()
    design, drop, index = increment_targets(
        frame, cache, FIT_HORIZONS, censor_aware=args.censor_aware
    )
    if not args.censor_aware:
        # Cross-check against the shipped implementation while the two agree.
        reference, reference_drop = build_increment_targets(frame, cache, FIT_HORIZONS)
        assert reference.shape == design.shape
        assert np.allclose(reference_drop, drop)
    print(f"  {design.shape[0]} increment windows "
          f"({'censor-aware' if args.censor_aware else 'V8 pre-crossing only'}), "
          f"{time.time() - started:.0f}s", flush=True)
    groups = frame.building[index]
    row_ratio = ratio[index]
    margin = design[:, VOLTAGE].astype(float) - 2.4
    horizon = design[:, -1].astype(float)

    usable = np.isfinite(row_ratio)
    print(f"  {usable.mean():.1%} of windows have the ratio; "
          f"{np.unique(groups[usable]).size} buildings")
    design, drop, groups, row_ratio = (
        design[usable], drop[usable], groups[usable], row_ratio[usable]
    )
    margin, horizon = margin[usable], horizon[usable]
    extended = np.hstack([design, row_ratio[:, None].astype(np.float32)])

    params = dict(
        max_iter=args.max_iter, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.0, random_state=20260823,
    )

    def constraints(width: int, horizon_column: int) -> np.ndarray:
        out = np.zeros(width, dtype=int)
        out[horizon_column] = 1
        return out

    splitter = GroupKFold(n_splits=args.folds)
    results = {"drift": {"base": [], "plus": []}, "scatter": {"base": [], "plus": []}}
    band_gain: list[np.ndarray] = []
    bands = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9.0)]

    for fold, (train, test) in enumerate(splitter.split(design, drop, groups)):
        for label, matrix, horizon_column in (
            ("base", design, design.shape[1] - 1),
            ("plus", extended, design.shape[1] - 1),
        ):
            drift = HistGradientBoostingRegressor(
                monotonic_cst=constraints(matrix.shape[1], horizon_column), **params
            )
            drift.fit(matrix[train], drop[train])
            predicted = drift.predict(matrix[test])
            results["drift"][label].append(
                float(np.abs(drop[test] - predicted).mean())
            )
            residual = np.abs(drop[train] - drift.predict(matrix[train]))
            scatter = HistGradientBoostingRegressor(
                monotonic_cst=constraints(matrix.shape[1], horizon_column), **params
            )
            scatter.fit(matrix[train], residual)
            held = np.abs(drop[test] - predicted)
            results["scatter"][label].append(
                float(np.abs(held - scatter.predict(matrix[test])).mean())
            )
            if label == "plus":
                gain = np.abs(drop[test] - base_prediction) - np.abs(drop[test] - predicted)
                band_gain.append(
                    np.asarray([
                        float(gain[(margin[test] >= lo) & (margin[test] < hi)].mean())
                        if ((margin[test] >= lo) & (margin[test] < hi)).any() else np.nan
                        for lo, hi in bands
                    ])
                )
            else:
                base_prediction = predicted
        print(f"  fold {fold}: drift MAE {results['drift']['base'][-1]:.6f} -> "
              f"{results['drift']['plus'][-1]:.6f}   scatter MAE "
              f"{results['scatter']['base'][-1]:.6f} -> "
              f"{results['scatter']['plus'][-1]:.6f}   "
              f"({time.time() - started:.0f}s)", flush=True)

    print()
    summary = {}
    for target in ("drift", "scatter"):
        base = np.asarray(results[target]["base"])
        plus = np.asarray(results[target]["plus"])
        relative = (base - plus) / base
        summary[target] = {
            "base_mae": [round(v, 6) for v in base.tolist()],
            "plus_mae": [round(v, 6) for v in plus.tolist()],
            "mean_relative_gain": round(float(relative.mean()), 5),
            "folds_improved": int((plus < base).sum()),
        }
        print(f"{target:>8}: mean relative MAE gain {relative.mean():+.3%}, "
              f"improved in {int((plus < base).sum())}/{args.folds} building folds")
        print(f"          per fold: " + "  ".join(f"{v:+.2%}" for v in relative))

    stacked = np.vstack(band_gain)
    print()
    print("drift-MAE gain by margin band (V, positive = the ratio helps):")
    for column, (lo, hi) in enumerate(bands):
        values = stacked[:, column]
        print(f"  margin [{lo:.2f}, {hi:.2f}): mean {np.nanmean(values):+.3e}   "
              f"folds improved {int(np.nansum(values > 0))}/{args.folds}")
    summary["band_gain"] = {
        f"{lo}-{hi}": round(float(np.nanmean(stacked[:, i])), 8)
        for i, (lo, hi) in enumerate(bands)
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=1))


def _row_index(frame, cache, horizons) -> np.ndarray:
    """Which frame row each stacked increment window came from.

    Mirrors ``bsai.wiener.build_increment_targets`` exactly; if that changes,
    the assertion in ``main`` fires rather than silently misaligning.
    """
    from bsai.margin import EOL_THRESHOLD

    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")
    out: list[np.ndarray] = []
    for horizon in horizons:
        rows: list[np.ndarray] = []
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            margin = margins.get(device_id)
            start = stop
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            limit = last if crossing < 0 else min(last, crossing - 1)
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= limit) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            rows.append(chosen[finite])
        if rows:
            out.append(np.concatenate(rows))
    return np.concatenate(out)


if __name__ == "__main__":
    main()
