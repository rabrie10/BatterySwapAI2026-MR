"""Train the Wiener first-passage model.

Comparisons that decide whether this ships, all out-of-fold by building on the
42-day decision:

    physics control (margin / -slope)   precision@12 0.309, best timing 1823.4
    V6 hazard classifier                precision@12 0.300, best timing 1813.5

The control is the important one. A two-line rule matched the fifty-one-feature
classifier, so beating the classifier means nothing on its own.

    python tools/train_wiener.py --stride 4
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices

from bsai import features as feature_lib
from bsai.features import fleet_climatology
from bsai.hazard import build_training_frame
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache
from bsai.wiener import FIT_HORIZONS, WienerModel, build_increment_targets

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def report(probability: np.ndarray, truth: np.ndarray) -> dict:
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "base_rate": round(float(truth.mean()), 5),
        "auc": round(float(roc_auc_score(truth, probability)), 4),
        "pr_auc": round(float(average_precision_score(truth, probability)), 4),
        "predicted_over_actual": round(
            float(probability.sum() / max(truth.sum(), 1)), 3
        ),
    }
    order = np.argsort(-probability)
    for k in (50, 100, 200, 500, 1000):
        if k <= truth.size:
            out[f"precision_at_{k}"] = round(float(truth[order[:k]].mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/v7_training_report.json"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--no-shape", action="store_true", help="ablate the within-day features")
    parser.add_argument(
        "--feature-variant",
        choices=sorted(feature_lib.FEATURE_VARIANTS),
        default="base",
        help="feature set to train on; 'invariant' is the V12 building-"
        "invariant variant (see docs/TRANSFER_STRESS.md)",
    )
    parser.add_argument(
        "--knee-weight",
        type=float,
        default=0.0,
        help="extra weight on increment windows ending within 21 d of the "
        "device's crossing: weight = 1 + w. Counters the 2.3-2.4x knee "
        "under-representation and drift magnitude shrinkage measured in "
        "outputs/roadblock_report.md (L2 ii-iv). 0 disables.",
    )
    args = parser.parse_args()

    feature_lib.set_feature_variant(args.feature_variant)
    variant_names = tuple(feature_lib.active_feature_names())

    started = time.time()
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]

    print("smoothing and within-day shape...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = None if args.no_shape else ShapeCache()
    if shape_cache is not None:
        shape_cache.update(raw)
    raw_cache = None
    raw_any_cache = None
    if feature_lib.variant_needs_raw(args.feature_variant):
        from bsai.rawdaily import RawDailyCache
        from bsai.v12_rawany import RawAnyCache

        raw_cache = RawDailyCache()
        raw_cache.update(raw)
        raw_any_cache = RawAnyCache()
        raw_any_cache.update(raw)
    del raw
    print(f"  {len(cache.devices)} devices, {time.time() - started:.0f}s", flush=True)

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

    print("building cutoffs...", flush=True)
    if args.feature_variant == "base":
        frame = build_training_frame(
            cache,
            eol_index,
            building_of,
            observation_index,
            shape_cache=shape_cache,
            stride=args.stride,
        )
    else:
        # Variant rows need the raw-daily adapter, which the frozen
        # hazard.build_training_frame cannot pass; the tools-side twin
        # reproduces its cutoff population exactly.
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from v12_frame import build_variant_training_frame

        frame = build_variant_training_frame(
            cache,
            eol_index,
            building_of,
            observation_index,
            shape_cache=shape_cache,
            raw_cache=raw_cache,
            raw_any_cache=raw_any_cache,
            variant=args.feature_variant,
            stride=args.stride,
        )
    truth = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & ((frame.crossing - frame.cutoff) <= DECISION_HORIZON)
        & (frame.crossing <= frame.observation_end)
    ).astype(np.int8)
    decision_horizon = np.clip(
        np.minimum(DECISION_HORIZON, frame.observation_end - frame.cutoff), 0.0, None
    ).astype(np.float32)
    print(
        f"  {len(frame)} cutoffs x {frame.features.shape[1]} features, "
        f"{int(truth.sum())} due ({truth.mean():.4f}), {time.time() - started:.0f}s",
        flush=True,
    )

    print("building observed increments...", flush=True)
    window_weights = None
    if args.knee_weight > 0:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        from v12_fit import build_increment_targets_with_meta, knee_weights

        design, drop, end_to_crossing = build_increment_targets_with_meta(
            frame, cache, FIT_HORIZONS
        )
        window_weights = knee_weights(end_to_crossing, args.knee_weight)
        print(
            f"  knee-regime share {(window_weights > 1.0).mean():.4f}, "
            f"weight 1+{args.knee_weight:g}",
            flush=True,
        )
    else:
        design, drop = build_increment_targets(frame, cache, FIT_HORIZONS)
    print(
        f"  {design.shape[0]} windows, mean fall {drop.mean():.5f} V, "
        f"{time.time() - started:.0f}s",
        flush=True,
    )

    # Groups for the increment rows: rebuild by matching cutoffs is expensive, so
    # tag each window with its device's building as the frame was walked.
    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # A window inherits the building of the cutoff it came from. Rebuild that by
    # regenerating the same row order the target builder used.
    increment_groups = _increment_groups(frame, cache, FIT_HORIZONS)
    assert increment_groups.size == design.shape[0]

    def fit_model(rows=None) -> WienerModel:
        block = slice(None) if rows is None else rows
        if window_weights is None:
            return WienerModel.fit(
                design[block],
                drop[block],
                climatology,
                params={"max_iter": args.max_iter},
            )
        from v12_fit import fit_wiener_weighted

        return fit_wiener_weighted(
            design[block],
            drop[block],
            climatology,
            sample_weight=window_weights[block],
            params={"max_iter": args.max_iter},
        )

    print(f"fitting {args.folds} grouped folds...", flush=True)
    oof = np.zeros(len(frame), dtype=float)
    fold_models: dict[str, WienerModel] = {}
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_rows, _) in enumerate(
        splitter.split(design, drop, increment_groups)
    ):
        held_out = set(np.unique(increment_groups)) - set(
            np.unique(increment_groups[train_rows])
        )
        model = fit_model(train_rows)
        model.feature_names = variant_names
        mask = np.isin(frame.building, list(held_out))
        if mask.any():
            oof[mask] = model.probabilities(
                frame.features[mask], decision_horizon[mask]
            )
        for building in held_out:
            fold_models[str(building)] = model
        print(f"  fold {fold} done, {time.time() - started:.0f}s", flush=True)

    metrics = {"scale_1.0": report(oof, truth)}

    # One scalar for the smoothing: a seven-day trailing median suppresses the
    # excursions a Brownian path would make, so the observed running minimum is
    # less extreme than the increment variance implies.
    best_scale, best_gap = 1.0, abs(oof.sum() - truth.sum())
    for scale in np.arange(0.4, 2.61, 0.1):
        scaled = _rescale(fold_models, frame, decision_horizon, float(scale))
        gap = abs(scaled.sum() - truth.sum())
        if gap < best_gap:
            best_scale, best_gap = float(scale), gap
    oof_scaled = _rescale(fold_models, frame, decision_horizon, best_scale)
    metrics[f"scale_{best_scale:.1f}"] = report(oof_scaled, truth)
    print(f"volatility_scale = {best_scale:.1f}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    for model in fold_models.values():
        model.volatility_scale = best_scale

    print("fitting production model on all buildings...", flush=True)
    production = fit_model()
    production.feature_names = variant_names
    if args.feature_variant != "base":
        production.model_version = f"bsai-wiener/v1+{args.feature_variant}"
    if args.knee_weight > 0:
        production.model_version += f"+k{args.knee_weight:g}"
    production.volatility_scale = best_scale
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, args.out)
    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"by_building": fold_models, "climatology": climatology}, args.folds_out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_version": production.model_version,
                "within_day_features": not args.no_shape,
                "feature_variant": args.feature_variant,
                "knee_weight": args.knee_weight,
                "n_features": int(frame.features.shape[1]),
                "stride_days": int(args.stride),
                "fit_horizons": list(FIT_HORIZONS),
                "volatility_scale": best_scale,
                "n_cutoffs": int(len(frame)),
                "n_windows": int(design.shape[0]),
                "metrics": metrics,
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


def _rescale(fold_models, frame, decision_horizon, scale: float) -> np.ndarray:
    out = np.zeros(len(frame), dtype=float)
    for building in np.unique(frame.building):
        model = fold_models.get(str(building))
        if model is None:
            continue
        mask = frame.building == building
        previous = model.volatility_scale
        model.volatility_scale = scale
        out[mask] = model.probabilities(
            frame.features[mask], decision_horizon[mask]
        )
        model.volatility_scale = previous
    return out


def _increment_groups(frame, cache, horizons) -> np.ndarray:
    """Building label per increment window, in the builder's row order."""
    parts: list[np.ndarray] = []
    order = np.argsort(frame.device, kind="stable")
    for horizon in horizons:
        rows: list[np.ndarray] = []
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            start = stop
            series = cache.devices.get(device_id)
            if series is None:
                continue
            margin = series.smooth_voltage
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            # Mirrors build_increment_targets: windows may end past the crossing.
            usable = (ends <= last) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            finite = np.isfinite(margin[cutoffs[usable]]) & np.isfinite(margin[ends[usable]])
            if not finite.any():
                continue
            rows.append(frame.building[chosen[finite]])
        if rows:
            parts.append(np.concatenate(rows))
    return np.concatenate(parts)


if __name__ == "__main__":
    main()
