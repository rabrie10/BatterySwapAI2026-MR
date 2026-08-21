"""Train the V7 margin model.

The decisive comparison is against V6's out-of-fold horizon-42 numbers on the
same folds: PR-AUC 0.4052, and precision at the swap counts we actually operate
at. The leaderboard charges us 56.6 of early cost per planned swap against first
place's 24.0, so precision at k in the 10-25 range is the number that matters,
not AUC.

    python tools/train_v7.py --stride 17     # fast feasibility check
    python tools/train_v7.py                 # full run
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

from bsai.features import fleet_climatology
from bsai.hazard import build_training_frame
from bsai.margin import (
    QUANTILES,
    TRAIN_HORIZONS,
    MarginModel,
    build_margin_targets,
)
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def load_everything(dataset: Path):
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]

    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
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
    return cache, building_of, eol_index, observation_index


def decision_labels(frame) -> tuple[np.ndarray, np.ndarray]:
    """Truth and effective horizon for the 42-day decision the planner makes."""
    effective = np.clip(
        np.minimum(DECISION_HORIZON, frame.observation_end - frame.cutoff), 0.0, None
    ).astype(np.float32)
    truth = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & ((frame.crossing - frame.cutoff) <= DECISION_HORIZON)
        & (frame.crossing <= frame.observation_end)
    )
    return truth.astype(np.int8), effective


def report(probability: np.ndarray, truth: np.ndarray) -> dict:
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "base_rate": round(float(truth.mean()), 5),
        "auc": round(float(roc_auc_score(truth, probability)), 4),
        "pr_auc": round(float(average_precision_score(truth, probability)), 4),
    }
    order = np.argsort(-probability)
    for k in (10, 20, 50, 100, 200, 500):
        if k <= truth.size:
            out[f"precision_at_{k}"] = round(float(truth[order[:k]].mean()), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("models/v7_margin.joblib"))
    parser.add_argument("--folds-out", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("docs/v7_training_report.json"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()

    started = time.time()
    print("smoothing...", flush=True)
    cache, building_of, eol_index, observation_index = load_everything(args.dataset)
    print(f"  {len(cache.devices)} devices, {time.time() - started:.0f}s", flush=True)

    print("building cutoffs...", flush=True)
    frame = build_training_frame(
        cache, eol_index, building_of, observation_index, stride=args.stride
    )
    truth, decision_horizon = decision_labels(frame)
    print(
        f"  {len(frame)} cutoffs, {int(truth.sum())} due within {DECISION_HORIZON}d "
        f"({truth.mean():.4f}), {time.time() - started:.0f}s",
        flush=True,
    )

    print("building margin targets...", flush=True)
    design, y, row_index, _ = build_margin_targets(frame, cache, TRAIN_HORIZONS)
    groups = frame.building[row_index]
    print(
        f"  {design.shape[0]} rows x {design.shape[1]} cols, "
        f"{int((y < 0).sum())} crossing ({(y < 0).mean():.4f}), "
        f"{time.time() - started:.0f}s",
        flush=True,
    )

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # The evaluation design is the 42-day decision, one row per cutoff.
    decision_design = np.hstack(
        [frame.features, decision_horizon[:, None].astype(np.float32)]
    )

    print(f"fitting {args.folds} grouped folds x {len(QUANTILES)} quantiles...", flush=True)
    oof = np.zeros(len(frame), dtype=float)
    fold_models: dict[str, MarginModel] = {}
    splitter = GroupKFold(n_splits=args.folds)
    cutoff_groups = frame.building

    for fold, (train_rows, _) in enumerate(
        splitter.split(design, y, groups)
    ):
        held_out = set(np.unique(groups)) - set(np.unique(groups[train_rows]))
        model = MarginModel.fit(
            design[train_rows],
            y[train_rows],
            climatology,
            params={"max_iter": args.max_iter},
        )
        mask = np.isin(cutoff_groups, list(held_out))
        if mask.any():
            grid = model.predict_grid(
                frame.features[mask], decision_horizon[mask].astype(float)
            )
            # predict_grid returns the whole horizon grid; take the 42-day column.
            column = list(model.horizons).index(DECISION_HORIZON)
            oof[mask] = grid[:, column]
        for building in held_out:
            fold_models[str(building)] = model
        print(f"  fold {fold} done, {time.time() - started:.0f}s", flush=True)

    metrics = {"decision_horizon_42_oof": report(oof, truth)}
    print(json.dumps(metrics, indent=2), flush=True)

    if not args.skip_production:
        print("fitting production model on all buildings...", flush=True)
        production = MarginModel.fit(
            design, y, climatology, params={"max_iter": args.max_iter}
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(production, args.out)
        args.folds_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"by_building": fold_models, "climatology": climatology}, args.folds_out
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_version": "bsai-margin/v1",
                "stride_days": int(args.stride),
                "train_horizons": list(TRAIN_HORIZONS),
                "quantiles": list(QUANTILES),
                "max_iter": int(args.max_iter),
                "n_cutoffs": int(len(frame)),
                "n_rows": int(design.shape[0]),
                "metrics": metrics,
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {args.report} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
