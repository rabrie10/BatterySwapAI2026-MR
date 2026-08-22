"""Train the multi-quantile increment head and run gates (a) and (b).

Gate (a): stride-sample OOF PR-AUC on the 42-day decision must beat the
incumbent censored drift model's 0.4706 (models/v8_cens.joblib, stride 4).
The incumbent is ALSO re-scored on this run's exact rows via its shipped fold
models (outputs/v8_folds_cens.joblib, volatility_scale forced to 1.0, raw
probabilities) so the comparison never rides on a population difference.

Gate (b): on the knee cell (margin 0.05-0.10 x beta_30 >= 0.014) the mean
predicted p must land 0.20-0.35 (empirical 0.275 at scenario cutoffs) without
the plateau cell (margin > 0.15 x beta_30 < 0.008) inflating past 2x its
empirical rate. Checked on the stride OOF rows and, when the transfer-stress
prep checkpoint is available, on the true scenario-cutoff frame.

Everything upstream of the quantile regressors -- the censor-aware increment
targets and the building-grouped folds -- is imported from bsai.wiener /
tools/train_wiener.py unchanged.

    python tools/train_qhead.py --stride 4
    python tools/train_qhead.py --stride 4 --aux   # + fixed-horizon 14/28 sets
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
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from batteryswap_public.utils import load_devices

from bsai.features import FEATURE_NAMES, fleet_climatology
from bsai.hazard import build_training_frame
from bsai.margin import EOL_THRESHOLD
from bsai.quantile_head import QUANTILES, QuantileHeadModel
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache
from bsai.wiener import FIT_HORIZONS, build_increment_targets

# Reuse the incumbent's mechanics verbatim: same censor-aware targets, same
# fold grouping by building, same OOF report.
from train_wiener import DECISION_HORIZON, _increment_groups, _ordinal, report

VOLTAGE_COL = FEATURE_NAMES.index("voltage")
BETA30_COL = FEATURE_NAMES.index("beta_30")

INCUMBENT_PR_AUC = 0.4706  # models/v8_cens.joblib stride-4 OOF reference

TRANSFER_PREP = Path(
    os.environ.get(
        "TRANSFER_STRESS_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\transfer_stress",
    )
) / "prep.joblib"


def cell_masks(features: np.ndarray) -> dict[str, np.ndarray]:
    margin = features[:, VOLTAGE_COL].astype(float) - EOL_THRESHOLD
    beta30 = features[:, BETA30_COL].astype(float)
    return {
        "knee": (margin >= 0.05) & (margin < 0.10) & (beta30 >= 0.014),
        "plateau": (margin > 0.15) & (beta30 < 0.008),
    }


def cell_report(
    features: np.ndarray, truth: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict:
    out: dict = {}
    for name, mask in cell_masks(features).items():
        entry = {
            "n": int(mask.sum()),
            "empirical": round(float(truth[mask].mean()), 4) if mask.any() else None,
        }
        for label, p in predictions.items():
            entry[f"mean_p_{label}"] = (
                round(float(p[mask].mean()), 4) if mask.any() else None
            )
        out[name] = entry
    return out


def fit_fold_aux(model: QuantileHeadModel, design, drop, rows, params) -> None:
    """Attach fixed-horizon 14/28 quantile sets fitted on ``rows`` only."""
    aux: dict = {}
    horizon_column = design[:, -1]
    for aux_horizon in (14, 28):
        subset = rows & (horizon_column == aux_horizon)
        aux[aux_horizon] = QuantileHeadModel.fit_aux(
            design[subset, :-1], drop[subset], params=params
        )
    model.aux = aux


def incumbent_oof(frame, decision_horizon: np.ndarray) -> np.ndarray | None:
    """Shipped cens fold models scored raw at scale 1.0 on this run's rows."""
    path = REPO_ROOT / "outputs/v8_folds_cens.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    out = np.zeros(len(frame), dtype=float)
    seen = np.zeros(len(frame), dtype=bool)
    for building in np.unique(frame.building):
        model = bundle["by_building"].get(str(building))
        if model is None:
            continue
        mask = frame.building == building
        previous = model.volatility_scale
        model.volatility_scale = 1.0
        out[mask] = model.probabilities(frame.features[mask], decision_horizon[mask])
        model.volatility_scale = previous
        seen[mask] = True
    if not seen.all():
        print(f"  incumbent OOF missing {int((~seen).sum())} rows", flush=True)
    return out


def scenario_check(fold_models: dict, args) -> dict | None:
    """Gate (b) on the true scenario-cutoff population (19,890 rows)."""
    if not TRANSFER_PREP.exists():
        return None
    scen = joblib.load(TRANSFER_PREP)["scen"]
    has = scen["has_row"]
    h_eff = np.clip(np.minimum(42.0, scen["remaining"]), 0.0, None)

    def dispatch(models: dict, scale: float | None = None) -> np.ndarray:
        p = np.zeros(len(scen["due"]), dtype=float)
        for building in np.unique(scen["building"]):
            model = models.get(str(building))
            if model is None:
                continue
            mask = (scen["building"] == building) & has
            if not mask.any():
                continue
            previous = model.volatility_scale
            if scale is not None:
                model.volatility_scale = scale
            raw = model.probabilities(
                scen["features"][mask].astype(np.float32), h_eff[mask]
            )
            model.volatility_scale = previous
            p[mask] = np.where(h_eff[mask] <= 0.0, 0.0, raw)
        return p

    predictions = {"qhead": dispatch(fold_models)}
    incumbent_path = REPO_ROOT / "outputs/v8_folds_cens.joblib"
    if incumbent_path.exists():
        predictions["cens"] = dispatch(
            joblib.load(incumbent_path)["by_building"], scale=1.0
        )

    due = scen["due"].astype(float)
    out = {
        "n_rows": int(len(due)),
        "n_due": int(due.sum()),
        "sum_p_per_scenario": {
            label: round(float(p.sum() / scen["n_scenarios"]), 3)
            for label, p in predictions.items()
        },
        "realized_per_scenario": round(float(due.sum() / scen["n_scenarios"]), 3),
        "cells": cell_report(scen["features"], due, predictions),
    }
    try:
        from sklearn.metrics import average_precision_score

        for label, p in predictions.items():
            out[f"pr_auc_{label}"] = round(
                float(average_precision_score(scen["due"].astype(int), p)), 4
            )
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qhead_model.joblib"))
    parser.add_argument(
        "--folds-out", type=Path, default=Path("outputs/qhead_folds.joblib")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/qhead_training_report.json")
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument(
        "--aux",
        action="store_true",
        help="add fixed-horizon 14/28 quantile sets, pointwise max (iteration 4)",
    )
    parser.add_argument(
        "--skip-production",
        action="store_true",
        help="OOF gates only; do not fit or write the production artifacts",
    )
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
    print(
        f"  {len(frame)} cutoffs, {int(truth.sum())} due ({truth.mean():.4f}), "
        f"{time.time() - started:.0f}s",
        flush=True,
    )

    print("building observed increments (censor-aware)...", flush=True)
    design, drop = build_increment_targets(frame, cache, FIT_HORIZONS)
    increment_groups = _increment_groups(frame, cache, FIT_HORIZONS)
    assert increment_groups.size == design.shape[0]
    print(f"  {design.shape[0]} windows, {time.time() - started:.0f}s", flush=True)

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    params = {"max_iter": args.max_iter}
    print(
        f"fitting {args.folds} grouped folds x {len(QUANTILES)} quantiles"
        f"{' + aux 14/28' if args.aux else ''}...",
        flush=True,
    )
    oof = np.zeros(len(frame), dtype=float)
    fold_models: dict[str, QuantileHeadModel] = {}
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train_rows, _) in enumerate(
        splitter.split(design, drop, increment_groups)
    ):
        held_out = set(np.unique(increment_groups)) - set(
            np.unique(increment_groups[train_rows])
        )
        model = QuantileHeadModel.fit(
            design[train_rows], drop[train_rows], climatology, params=params
        )
        if args.aux:
            train_mask = np.zeros(design.shape[0], dtype=bool)
            train_mask[train_rows] = True
            fit_fold_aux(model, design, drop, train_mask, params)
        mask = np.isin(frame.building, list(held_out))
        if mask.any():
            oof[mask] = model.probabilities(
                frame.features[mask], decision_horizon[mask]
            )
        for building in held_out:
            fold_models[str(building)] = model
        print(f"  fold {fold} done, {time.time() - started:.0f}s", flush=True)

    metrics = {"qhead": report(oof, truth)}

    print("re-scoring incumbent cens fold models on the same rows...", flush=True)
    oof_cens = incumbent_oof(frame, decision_horizon)
    if oof_cens is not None:
        metrics["incumbent_cens_same_rows"] = report(oof_cens, truth)

    predictions = {"qhead": oof}
    if oof_cens is not None:
        predictions["cens"] = oof_cens
    stride_cells = cell_report(frame.features, truth.astype(float), predictions)

    gate_a = {
        "qhead_pr_auc": metrics["qhead"]["pr_auc"],
        "incumbent_reference_pr_auc": INCUMBENT_PR_AUC,
        "incumbent_same_rows_pr_auc": metrics.get(
            "incumbent_cens_same_rows", {}
        ).get("pr_auc"),
        "qhead_precision_at_100": metrics["qhead"].get("precision_at_100"),
        "qhead_precision_at_500": metrics["qhead"].get("precision_at_500"),
        "pass": bool(metrics["qhead"]["pr_auc"] > INCUMBENT_PR_AUC),
    }
    print(json.dumps({"metrics": metrics, "gate_a": gate_a}, indent=2), flush=True)
    print(json.dumps({"gate_b_stride": stride_cells}, indent=2), flush=True)

    print("scenario-cutoff check (transfer-stress prep)...", flush=True)
    scen_check = scenario_check(fold_models, args)
    if scen_check is not None:
        print(json.dumps({"gate_b_scenario": scen_check}, indent=2), flush=True)

    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"by_building": fold_models, "climatology": climatology}, args.folds_out
    )
    print(f"wrote {args.folds_out}", flush=True)

    if not args.skip_production:
        print("fitting production model on all buildings...", flush=True)
        production = QuantileHeadModel.fit(design, drop, climatology, params=params)
        if args.aux:
            fit_fold_aux(
                production, design, drop, np.ones(design.shape[0], dtype=bool), params
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(production, args.out)
        print(f"wrote {args.out}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "model_version": "bsai-qhead/v1",
                "quantiles": list(QUANTILES),
                "aux": bool(args.aux),
                "stride_days": int(args.stride),
                "max_iter": int(args.max_iter),
                "fit_horizons": list(FIT_HORIZONS),
                "n_cutoffs": int(len(frame)),
                "n_windows": int(design.shape[0]),
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
