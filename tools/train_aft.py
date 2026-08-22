"""Fit the censored days-to-EOL model on the folds the Wiener model uses.

The comparison only means something if nothing else changes, so the fold
partition is *read back* from the existing Wiener bundle rather than recomputed:
``GroupKFold`` assigns buildings by their row counts, and this model is fitted on
cutoff rows where that one was fitted on increment windows, so recomputing would
quietly produce a different split and a different number.

    python tools/train_aft.py --stride 4 --max-iter 250
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices

from bsai.aft import AFTModel
from bsai.features import fleet_climatology
from bsai.hazard import build_training_frame
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def report(probability: np.ndarray, truth: np.ndarray) -> dict:
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
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
    parser.add_argument("--wiener-folds", type=Path,
                        default=Path("outputs/v7_folds.joblib"),
                        help="source of the building-to-fold partition")
    parser.add_argument("--folds-out", type=Path,
                        default=Path("outputs/v8_aft_folds.joblib"))
    parser.add_argument("--report", type=Path,
                        default=Path("docs/v8_aft_training_report.json"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--em-iterations", type=int, default=4)
    args = parser.parse_args()

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
    shape_cache = ShapeCache()
    shape_cache.update(raw)
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
    frame = build_training_frame(
        cache, eol_index, building_of, observation_index,
        shape_cache=shape_cache, stride=args.stride,
    )

    # Event time is the crossing; censoring is the end of the observation window,
    # because that is the last day on which a record could be filed.
    event = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & (frame.crossing <= frame.observation_end)
    )
    duration = np.where(
        event,
        frame.crossing - frame.cutoff,
        np.maximum(frame.observation_end - frame.cutoff, 1),
    ).astype(float)
    truth = (event & ((frame.crossing - frame.cutoff) <= DECISION_HORIZON)).astype(np.int8)
    remaining = np.maximum(frame.observation_end - frame.cutoff, 0).astype(float)
    print(
        f"  {len(frame)} cutoffs x {frame.features.shape[1]} features, "
        f"{int(event.sum())} events ({event.mean():.4f}), "
        f"{int(truth.sum())} due within {DECISION_HORIZON}d, "
        f"median event time {np.median(duration[event]):.0f}d, "
        f"{time.time() - started:.0f}s",
        flush=True,
    )

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # Reuse the Wiener partition exactly: buildings that share a model object
    # shared a fold.
    wiener = joblib.load(args.wiener_folds)
    fold_of_building: dict[str, int] = {}
    seen: dict[int, int] = {}
    for building, model in wiener["by_building"].items():
        key = id(model)
        if key not in seen:
            seen[key] = len(seen)
        fold_of_building[str(building)] = seen[key]
    print(f"reusing {len(seen)} folds from {args.wiener_folds}", flush=True)

    row_fold = np.array(
        [fold_of_building.get(str(b), -1) for b in frame.building], dtype=int
    )
    oof = np.zeros(len(frame), dtype=float)
    fold_models: dict[str, AFTModel] = {}
    for fold in sorted(seen.values()):
        held_out = row_fold == fold
        if not held_out.any():
            continue
        train_rows = ~held_out
        model = AFTModel.fit(
            frame.features[train_rows],
            duration[train_rows],
            event[train_rows],
            climatology,
            params={"max_iter": args.max_iter},
            iterations=args.em_iterations,
        )
        grid = model.predict_grid(frame.features[held_out], remaining[held_out])
        # Column of the 42-day decision horizon, clipped by remaining observation
        # exactly as the planner will see it.
        column = list(model.horizons).index(DECISION_HORIZON)
        oof[held_out] = grid[:, column]
        for building, assigned in fold_of_building.items():
            if assigned == fold:
                fold_models[building] = model
        print(
            f"  fold {fold} done, sigma={model.sigma:.3f}, "
            f"{time.time() - started:.0f}s",
            flush=True,
        )

    metrics = {"oof": report(oof, truth.astype(int))}
    print(json.dumps(metrics, indent=2), flush=True)

    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"by_building": fold_models, "climatology": climatology}, args.folds_out
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "stride": args.stride,
                "max_iter": args.max_iter,
                "em_iterations": args.em_iterations,
                "n_events": int(event.sum()),
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"wrote {args.folds_out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
