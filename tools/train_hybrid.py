"""Train the scenario-landmark GBDT discrete-hazard + Weibull-AFT hybrid.

The near-term model is validated out of fold by building.  Its calibration is
fit only on those out-of-building predictions, then frozen into both the fold
bundle and the production artifact.
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

from batteryswap_public.utils import load_dataset, load_devices

from bsai.features import fleet_climatology
from bsai.hazard import build_training_frame
from bsai.hybrid import (
    HAZARD_BINS,
    HybridHazardAFTModel,
    fit_horizon_calibrators,
    horizon_labels,
    take_frame,
)
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def _metric(probability: np.ndarray, truth: np.ndarray) -> dict:
    order = np.argsort(-probability)
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "predicted_over_actual": round(float(probability.sum() / max(truth.sum(), 1)), 3),
        "auc": round(float(roc_auc_score(truth, probability)), 4),
        "pr_auc": round(float(average_precision_score(truth, probability)), 4),
    }
    for k in (50, 100, 200, 500):
        if k <= truth.size:
            out[f"precision_at_{k}"] = round(float(truth[order[:k]].mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--out", type=Path, default=Path("models/v8_hybrid.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v8_hybrid_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/v8_hybrid_training_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--aft-penalizer", type=float, default=0.5)
    parser.add_argument("--rank-blend", type=float, default=0.65,
                        help="weight of the direct 42-day ranking head")
    parser.add_argument("--scenario-step", type=int, default=1,
                        help="use every Nth official scenario cutoff for quick experiments")
    args = parser.parse_args()

    started = time.time()
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"])
    observation_end = devices.set_index("device_id")["end_time"]
    _, _, _, scenarios = load_dataset(args.dataset)
    cutoff_days = np.asarray(
        [_ordinal(scenario["start_time"]) for scenario in scenarios[:: max(args.scenario_step, 1)]],
        dtype=np.int64,
    )

    print("smoothing and within-day shape...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw

    eol_index: dict[str, int | None] = {}
    observation_index: dict[str, int] = {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = None if pd.isna(moment) else _ordinal(moment) - series.origin
        end = observation_end.get(device_id)
        observation_index[device_id] = (
            series.origin + len(series) - 1 if pd.isna(end) else _ordinal(end) - series.origin
        )

    frame = build_training_frame(
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
        cutoff_days=cutoff_days,
    )
    remaining = np.maximum(frame.observation_end - frame.cutoff, 0).astype(float)
    labels = horizon_labels(frame)
    climatology = fleet_climatology(
        {device: (series.origin, series.smooth_temperature) for device, series in cache.devices.items()}
    )
    print(
        f"{len(frame)} scenario landmarks, {frame.features.shape[1]} features, "
        f"{int(labels[:, -1].sum())} due-within-42 views",
        flush=True,
    )

    raw_oof = np.zeros((len(frame), len(HAZARD_BINS)), dtype=float)
    fold_models: dict[str, HybridHazardAFTModel] = {}
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train, test) in enumerate(splitter.split(frame.features, labels[:, -1], frame.building)):
        model = HybridHazardAFTModel.fit(
            take_frame(frame, train),
            climatology,
            hazard_params={"max_iter": args.max_iter, "random_state": 20260821 + fold},
            aft_penalizer=args.aft_penalizer,
        )
        model.rank_blend = float(args.rank_blend)
        grid = model.predict_grid(frame.features[test], remaining[test])
        columns = [list(model.horizons).index(horizon) for horizon in HAZARD_BINS]
        raw_oof[test] = grid[:, columns]
        for building in np.unique(frame.building[test]):
            fold_models[str(building)] = model
        print(f"fold {fold + 1}/{args.folds} done in {time.time() - started:.0f}s", flush=True)

    calibrators = fit_horizon_calibrators(raw_oof, labels)
    for model in {id(value): value for value in fold_models.values()}.values():
        model.calibrators = dict(calibrators)

    calibrated_oof = np.zeros_like(raw_oof)
    for column, horizon in enumerate(HAZARD_BINS):
        calibrated_oof[:, column] = calibrators[horizon].apply(raw_oof[:, column])
    calibrated_oof = np.maximum.accumulate(calibrated_oof, axis=1)

    print("fitting production model...", flush=True)
    production = HybridHazardAFTModel.fit(
        frame,
        climatology,
        hazard_params={"max_iter": args.max_iter},
        aft_penalizer=args.aft_penalizer,
    )
    production.rank_blend = float(args.rank_blend)
    production.calibrators = dict(calibrators)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, args.out)
    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"by_building": fold_models, "climatology": climatology}, args.folds_out)

    metrics = {
        str(horizon): _metric(calibrated_oof[:, column], labels[:, column])
        for column, horizon in enumerate(HAZARD_BINS)
    }
    report = {
        "model_version": production.model_version,
        "n_landmarks": int(len(frame)),
        "n_devices": int(np.unique(frame.device).size),
        "n_buildings": int(np.unique(frame.building).size),
        "n_features": int(frame.features.shape[1]),
        "scenario_step": int(args.scenario_step),
        "hazard_bins": list(HAZARD_BINS),
        "aft_penalizer": float(args.aft_penalizer),
        "rank_blend": float(args.rank_blend),
        "metrics": metrics,
        "seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {args.out} and {args.folds_out}", flush=True)


if __name__ == "__main__":
    main()
