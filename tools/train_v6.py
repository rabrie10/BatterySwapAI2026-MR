"""Train the V6 first-passage model.

Two artifacts come out of one pass so that the calibration attached to the
shipped model is never fitted on data the model itself saw:

* ``models/v6_hazard.joblib`` -- the production model, fitted on every building,
  carrying isotonic calibrators fitted on out-of-fold predictions only.
* ``outputs/v6_folds.joblib`` -- the five fold models, used by
  ``tools/validate_v6.py`` to score whole scenarios with predictions that never
  saw the device's own building. Not needed at submission time.

Folds are grouped by building because the public and private splits contain
different buildings, and the observed EOL rate per building spans 0.043 to
0.833. A random split would leak that structure and flatter every number.

    python tools/train_v6.py
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices

from bsai.features import fleet_climatology
from bsai.hazard import HORIZON_GRID, HazardModel, build_training_frame, stack_horizons
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")


def _ordinal(value: pd.Timestamp) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("models/v6_hazard.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v6_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/v6_training_report.json"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    args = parser.parse_args()

    started = time.time()
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))

    eol = pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"]
    eol = pd.to_datetime(eol)

    print("smoothing...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    del raw
    print(f"  {len(cache.devices)} devices, {time.time() - started:.0f}s", flush=True)

    observation_end = devices.set_index("device_id")["end_time"]
    observation_end_index: dict[str, int] = {}
    eol_index: dict[str, int | None] = {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = (
            None if pd.isna(moment) else _ordinal(moment) - series.origin
        )
        end = observation_end.get(device_id)
        observation_end_index[device_id] = (
            (series.origin + len(series) - 1)
            if pd.isna(end)
            else _ordinal(end) - series.origin
        )

    print("building cutoffs...", flush=True)
    frame = build_training_frame(
        cache, eol_index, building_of, observation_end_index, stride=args.stride
    )
    design, labels, row_index, horizons = stack_horizons(frame)
    groups = frame.building[row_index]
    print(
        f"  {len(frame)} cutoffs -> {len(labels)} stacked rows, "
        f"{int(labels.sum())} positive ({labels.mean():.4f}), {time.time() - started:.0f}s",
        flush=True,
    )

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # A second cutoff population, sampled the way a scenario samples: every
    # alive device on a shared grid of dates. Calibrating on the stride
    # population left the top probability bucket predicting 0.87 against a
    # realised 0.39 at scenario cutoffs, and the planner acts on exactly those
    # high-probability batteries.
    span = max(
        series.origin + len(series) for series in cache.devices.values()
    )
    floor = min(series.origin for series in cache.devices.values())
    monday = floor + ((3 - floor % 7) % 7)  # 1970-01-01 was a Thursday
    calibration_days = np.arange(monday, span, 7, dtype=np.int64)
    calibration_frame = build_training_frame(
        cache,
        eol_index,
        building_of,
        observation_end_index,
        cutoff_days=calibration_days,
    )
    calibration_design, calibration_labels, calibration_index, calibration_horizons = (
        stack_horizons(calibration_frame)
    )
    calibration_groups = calibration_frame.building[calibration_index]
    print(
        f"  calibration population: {len(calibration_frame)} cutoffs over "
        f"{len(calibration_days)} dates -> {len(calibration_labels)} rows, "
        f"{int(calibration_labels.sum())} positive ({calibration_labels.mean():.4f})",
        flush=True,
    )

    print(f"fitting {args.folds} grouped folds...", flush=True)
    oof = np.zeros(len(labels))
    calibration_oof = np.zeros(len(calibration_labels))
    fold_models: dict[str, HazardModel] = {}
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(design, labels, groups)
    ):
        model = HazardModel.fit(
            design[train_index],
            labels[train_index],
            climatology,
            params={"max_iter": args.max_iter},
        )
        oof[test_index] = model.classifier.predict_proba(design[test_index])[:, 1]
        held_out = set(np.unique(groups[test_index]))
        for building in held_out:
            fold_models[str(building)] = model
        mask = np.isin(calibration_groups, list(held_out))
        if mask.any():
            calibration_oof[mask] = model.classifier.predict_proba(
                calibration_design[mask]
            )[:, 1]
        print(f"  fold {fold} done, {time.time() - started:.0f}s", flush=True)

    metrics = _report(oof, labels, horizons)
    print(json.dumps(metrics, indent=2))

    print("fitting production model on all buildings...", flush=True)
    production = HazardModel.fit(
        design, labels, climatology, params={"max_iter": args.max_iter}
    )
    production.fit_calibration(
        calibration_oof, calibration_labels, calibration_horizons
    )
    # Validation must not be scored through a calibrator fitted on its own
    # buildings, so each fold gets one fitted on the other four.
    for fold, (_, test_index) in enumerate(splitter.split(design, labels, groups)):
        held_out = set(np.unique(groups[test_index]))
        others = ~np.isin(calibration_groups, list(held_out))
        fold_calibrated = HazardModel(
            classifier=fold_models[str(next(iter(held_out)))].classifier,
            climatology=climatology,
        )
        fold_calibrated.fit_calibration(
            calibration_oof[others],
            calibration_labels[others],
            calibration_horizons[others],
        )
        for building in held_out:
            fold_models[str(building)] = fold_calibrated

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, args.out)
    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"by_building": fold_models, "climatology": climatology}, args.folds_out)

    report = {
        "model_version": production.model_version,
        "horizons": list(HORIZON_GRID),
        "n_cutoffs": int(len(frame)),
        "n_stacked_rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "stride_days": int(args.stride),
        "folds": int(args.folds),
        "n_features": int(design.shape[1] - 1),
        "calibrated_bands": sorted(production.calibrators),
        "calibration_rows": int(len(calibration_labels)),
        "calibration_positive_rate": float(calibration_labels.mean()),
        "metrics": metrics,
        "seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out} and {args.report} in {time.time() - started:.0f}s")


def _report(oof: np.ndarray, labels: np.ndarray, horizons: np.ndarray) -> dict:
    """Out-of-fold quality, reported the way the plan requires.

    PR-AUC and precision@k rather than AUC alone: at a 1% base rate a high AUC
    coexists comfortably with a planner that services the wrong batteries.
    """
    out: dict = {
        "stacked": {
            "auc": round(float(roc_auc_score(labels, oof)), 4),
            "pr_auc": round(float(average_precision_score(labels, oof)), 4),
            "brier": round(float(brier_score_loss(labels, oof)), 5),
        }
    }
    for horizon in (14, 42):
        mask = horizons == horizon
        if mask.sum() < 100 or labels[mask].sum() < 5:
            continue
        probabilities, actual = oof[mask], labels[mask]
        order = np.argsort(-probabilities)
        entry = {
            "n": int(mask.sum()),
            "positives": int(actual.sum()),
            "base_rate": round(float(actual.mean()), 4),
            "auc": round(float(roc_auc_score(actual, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(actual, probabilities)), 4),
        }
        for k in (10, 20, 50, 100):
            entry[f"precision_at_{k}"] = round(float(actual[order[:k]].mean()), 3)
        out[f"horizon_{horizon}"] = entry
    return out


if __name__ == "__main__":
    main()
