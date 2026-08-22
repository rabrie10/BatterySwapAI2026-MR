"""Iteration 4: attach fixed-horizon 14/28 aux quantile sets to the fitted head.

The base head failed gate (a) by 0.0011 (<0.02), which triggers the one
allowed iteration: per fold, fit two extra quantile sets on that fold's
horizon-14 and horizon-28 windows only (features WITHOUT the horizon column)
and let ``QuantileHeadModel.probabilities`` take the pointwise max of the
three crossing probabilities -- a plunge that completes in 14 days is certain
by 42, and the dedicated short-horizon sets see those plunges undiluted by
the pooled monotone-in-horizon surface.

The main regressors are NOT refit: the training frame, increment targets and
fold split are deterministic (same stride, same seed), so this reproduces the
training run's fold membership exactly and only adds the aux fits, then
re-runs gates (a) and (b).

    python tools/qhead_aux_patch.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from batteryswap_public.utils import load_devices

from bsai.hazard import build_training_frame
from bsai.quantile_head import QuantileHeadModel
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache
from bsai.wiener import FIT_HORIZONS, build_increment_targets

from train_qhead import (
    INCUMBENT_PR_AUC,
    cell_report,
    incumbent_oof,
    scenario_check,
)
from train_wiener import DECISION_HORIZON, _increment_groups, _ordinal, report

AUX_HORIZONS = (14, 28)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/qhead_folds.joblib"))
    parser.add_argument("--model", type=Path, default=Path("outputs/qhead_model.joblib"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/qhead_aux_report.json")
    )
    args = parser.parse_args()

    started = time.time()
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]

    print("rebuilding caches and frame (deterministic)...", flush=True)
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
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
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
    design, drop = build_increment_targets(frame, cache, FIT_HORIZONS)
    increment_groups = _increment_groups(frame, cache, FIT_HORIZONS)
    assert increment_groups.size == design.shape[0]
    print(
        f"  {len(frame)} cutoffs / {design.shape[0]} windows, "
        f"{time.time() - started:.0f}s",
        flush=True,
    )

    bundle = joblib.load(args.folds)
    fold_models = bundle["by_building"]
    params = {"max_iter": args.max_iter}
    horizon_column = design[:, -1]

    # One aux fit per DISTINCT fold model, on that fold's training windows.
    distinct: dict[int, QuantileHeadModel] = {}
    held_of: dict[int, list[str]] = {}
    for building, model in fold_models.items():
        distinct[id(model)] = model
        held_of.setdefault(id(model), []).append(building)

    for count, (key, model) in enumerate(distinct.items()):
        train = ~np.isin(increment_groups, held_of[key])
        aux = {}
        for aux_horizon in AUX_HORIZONS:
            subset = train & (horizon_column == aux_horizon)
            aux[aux_horizon] = QuantileHeadModel.fit_aux(
                design[subset, :-1], drop[subset], params=params
            )
        model.aux = aux
        print(
            f"  fold {count} aux fitted (held out {sorted(held_of[key])[:2]}...), "
            f"{time.time() - started:.0f}s",
            flush=True,
        )

    print("recomputing OOF at the 42-day decision...", flush=True)
    oof = np.zeros(len(frame), dtype=float)
    for building in np.unique(frame.building):
        model = fold_models.get(str(building))
        if model is None:
            continue
        mask = frame.building == building
        oof[mask] = model.probabilities(frame.features[mask], decision_horizon[mask])

    metrics = {"qhead_aux": report(oof, truth)}
    oof_cens = incumbent_oof(frame, decision_horizon)
    if oof_cens is not None:
        metrics["incumbent_cens_same_rows"] = report(oof_cens, truth)

    predictions = {"qhead_aux": oof}
    if oof_cens is not None:
        predictions["cens"] = oof_cens
    stride_cells = cell_report(frame.features, truth.astype(float), predictions)

    gate_a = {
        "qhead_pr_auc": metrics["qhead_aux"]["pr_auc"],
        "incumbent_reference_pr_auc": INCUMBENT_PR_AUC,
        "qhead_precision_at_100": metrics["qhead_aux"].get("precision_at_100"),
        "qhead_precision_at_500": metrics["qhead_aux"].get("precision_at_500"),
        "pass": bool(metrics["qhead_aux"]["pr_auc"] > INCUMBENT_PR_AUC),
    }
    print(json.dumps({"metrics": metrics, "gate_a": gate_a}, indent=2), flush=True)
    print(json.dumps({"gate_b_stride": stride_cells}, indent=2), flush=True)

    scen_check = scenario_check(fold_models, None)
    if scen_check is not None:
        print(json.dumps({"gate_b_scenario": scen_check}, indent=2), flush=True)

    print("aux for the production model (all windows)...", flush=True)
    production = joblib.load(args.model)
    aux = {}
    for aux_horizon in AUX_HORIZONS:
        subset = horizon_column == aux_horizon
        aux[aux_horizon] = QuantileHeadModel.fit_aux(
            design[subset, :-1], drop[subset], params=params
        )
    production.aux = aux

    joblib.dump(bundle, args.folds)
    joblib.dump(production, args.model)
    print(f"patched {args.folds} and {args.model}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_version": "bsai-qhead/v1",
                "aux_horizons": list(AUX_HORIZONS),
                "stride_days": int(args.stride),
                "max_iter": int(args.max_iter),
                "metrics": metrics,
                "gate_a": gate_a,
                "gate_b_stride": stride_cells,
                "gate_b_scenario": scen_check,
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"wrote {args.report} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
