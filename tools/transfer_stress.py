"""Transfer-stress harness: does the Task-1 model survive fresh buildings?

Why this exists. Local 5-fold-by-building validation said the censor-aware
target ("cens", models/v8_cens.joblib) was -84 cost against the incumbent
("v7", models/v7_wiener.joblib); the public leaderboard said +179 worse
(early_swap +104, late_swap +75), and its deployed sum-of-p ran >= 11.4 dues
per scenario against a realized ~9.5. Five folds over 24 buildings leave
near-twin buildings inside every training fold; public buildings are fresh
draws. This harness measures that gap directly with leave-one-building-out
(24 folds) and grouped "hard" holdouts chosen to be maximally dissimilar
from the training remainder, for BOTH target variants:

  v7-style   increment windows must end strictly before the crossing
             (limit = min(last, crossing-1)); no bump.
  cens-style current bsai/wiener.py: windows may end past the crossing and a
             window containing the crossing counts at least the full margin
             (drop = max(drop, margin_at_start)).

Everything is measured at the 48 scenario cutoffs on held-out buildings only:
sum-of-p vs realized (raw and with a RemainingCalibration fitted on the
training buildings, exactly the production procedure), PR-AUC, precision@k,
top-bucket reliability, rank-vs-level policies, and feature fragility
(KS shift + per-building dispersion of beta_30 vs beta_rise).

Fidelity caveats (relative comparisons are the point, not absolute costs):
  * stride 8 / max_iter 150 against production's stride 4 / max_iter 250;
  * per-fold RemainingCalibration is fitted on the fold's own training-building
    scenario rows through the fold model (production fitted on 5-fold OOF
    predictions); slightly optimistic for the calibrated numbers on both
    variants equally;
  * the scenario frame is rebuilt from the full-history caches (smoothing is
    causal, so this is exact for features); the reconstruction is validated in
    the prep phase against the shipped artifacts' known sums (8.45 / 10.01
    dues per scenario).

Usage (phases checkpoint into --work, safe to re-run):
    python tools/transfer_stress.py --phase prep
    python tools/transfer_stress.py --phase fit --scope hard
    python tools/transfer_stress.py --phase fit --scope loo
    python tools/transfer_stress.py --phase report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import average_precision_score

from batteryswap_public.utils import load_devices

from bsai.calibrate import RemainingCalibration
from bsai.features import (
    FEATURE_NAMES,
    DeviceView,
    FeatureContext,
    feature_row,
    fleet_climatology,
)
from bsai.hazard import build_training_frame
from bsai.margin import EOL_THRESHOLD
from bsai.shape import ShapeCache, align_to
from bsai.smoothing import SmoothingCache
from bsai.wiener import FIT_HORIZONS, WienerModel

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42
VARIANTS = ("v7", "cens")
DEFAULT_WORK = Path(
    os.environ.get(
        "TRANSFER_STRESS_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\transfer_stress",
    )
)

VOLTAGE_COL = FEATURE_NAMES.index("voltage")
BETA30_COL = FEATURE_NAMES.index("beta_30")
BETARISE_COL = FEATURE_NAMES.index("beta_rise")
DWELL_COL = FEATURE_NAMES.index("days_below_2.45")


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


# --------------------------------------------------------------------------
# prep: caches -> stride frame + scenario frame + window bank, checkpointed
# --------------------------------------------------------------------------

def build_scenario_frame(cache, shape_cache, devices, eol, scenarios) -> dict:
    """One row per (scenario, alive device), mirroring HazardForecaster.predict.

    Alive = no EOL recorded at or before the scenario start (iterate_scenarios'
    filter). Devices whose smoothed grid ended earlier still get a row at the
    clamped last index -- including the forecaster's understated staleness --
    because that is exactly what deployment does. Devices with no usable
    feature row keep probability zero but stay in the population, so sums and
    realized counts match the deployed accounting.
    """
    context = FeatureContext(
        climatology=fleet_climatology(
            {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
        )
    )
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    end_time = devices.set_index("device_id")["end_time"]

    views: dict[str, tuple] = {}
    for device_id, series in cache.devices.items():
        view = DeviceView(series.smooth_voltage, series.smooth_temperature)
        shape_view = align_to(
            None if shape_cache is None else shape_cache.devices.get(device_id),
            series.origin,
            len(series),
        )
        views[device_id] = (series, view, shape_view)

    columns: dict[str, list] = {
        k: []
        for k in (
            "scenario_index",
            "device",
            "building",
            "remaining",
            "due",
            "has_row",
        )
    }
    feature_rows: list[list[float] | None] = []

    for s_index, scenario in enumerate(scenarios):
        start = pd.Timestamp(scenario["start_time"])
        start_ordinal = _ordinal(start)
        horizon_end = start + pd.Timedelta(days=DECISION_HORIZON)
        for device_id in devices["device_id"]:
            moment = eol.get(device_id)
            if not pd.isna(moment) and moment <= start:
                continue  # already dead: the evaluator removed it from locations
            end = end_time.get(device_id)
            remaining = float(
                (pd.Timestamp(end).normalize() - start.normalize())
                / pd.Timedelta(days=1)
            )
            due = int((not pd.isna(moment)) and moment <= horizon_end)
            row = None
            bundle = views.get(device_id)
            if bundle is not None:
                series, view, shape_view = bundle
                index = start_ordinal - series.origin
                if index >= 0:
                    index = min(index, len(series) - 1)  # forecaster's clamp
                    row = feature_row(
                        view, index, series.origin + index, context, shape_view
                    )
            columns["scenario_index"].append(s_index)
            columns["device"].append(device_id)
            columns["building"].append(building_of.get(device_id, ""))
            columns["remaining"].append(remaining)
            columns["due"].append(due)
            columns["has_row"].append(row is not None)
            feature_rows.append(row)

    n = len(feature_rows)
    features = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i, row in enumerate(feature_rows):
        if row is not None:
            features[i] = row
    return {
        "scenario_index": np.asarray(columns["scenario_index"], dtype=np.int64),
        "device": np.asarray(columns["device"]),
        "building": np.asarray(columns["building"]),
        "remaining": np.asarray(columns["remaining"], dtype=float),
        "due": np.asarray(columns["due"], dtype=np.int8),
        "has_row": np.asarray(columns["has_row"], dtype=bool),
        "features": features,
        "n_scenarios": len(scenarios),
    }


def build_window_bank(frame, cache, horizons=FIT_HORIZONS) -> dict:
    """Increment windows with enough sidecar columns to derive BOTH variants.

    Mirrors bsai.wiener.build_increment_targets (cens rule: windows may end
    past the crossing). ``crossed`` marks windows containing/after the
    crossing; the v7-style variant is the ~crossed subset with the raw drop,
    the cens-style variant is every row with drop bumped to at least the
    starting margin on crossed windows.
    """
    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")

    designs, drops, starts, crossings, buildings = [], [], [], [], []
    for horizon in horizons:
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            start = stop
            margin = margins.get(device_id)
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= last) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            chosen = chosen[finite]
            designs.append(
                np.hstack(
                    [
                        frame.features[chosen],
                        np.full((chosen.size, 1), horizon, dtype=np.float32),
                    ]
                )
            )
            drops.append((here[finite] - there[finite]).astype(np.float32))
            starts.append(here[finite].astype(np.float32))
            crossed = (
                np.zeros(chosen.size, dtype=bool)
                if crossing < 0
                else (ends[usable][finite] >= crossing)
            )
            crossings.append(crossed)
            buildings.append(frame.building[chosen])

    return {
        "design": np.vstack(designs).astype(np.float32),
        "drop_raw": np.concatenate(drops),
        "margin_start": np.concatenate(starts),
        "crossed": np.concatenate(crossings),
        "building": np.concatenate(buildings),
    }


def variant_rows(bank: dict, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(row mask, target) for one target variant over the window bank."""
    if variant == "v7":
        return ~bank["crossed"], bank["drop_raw"]
    if variant == "cens":
        target = np.where(
            bank["crossed"],
            np.maximum(bank["drop_raw"], bank["margin_start"]),
            bank["drop_raw"],
        )
        return np.ones(len(bank["drop_raw"]), dtype=bool), target
    raise ValueError(variant)


def predict_scenario(model: WienerModel, scen: dict) -> np.ndarray:
    """42-day probability per scenario row: min(42, remaining) effective horizon."""
    p = np.zeros(len(scen["due"]), dtype=float)
    mask = scen["has_row"]
    h_eff = np.clip(np.minimum(42.0, scen["remaining"][mask]), 0.0, None)
    raw = model.probabilities(scen["features"][mask].astype(np.float32), h_eff)
    p[mask] = np.where(h_eff <= 0.0, 0.0, raw)
    return p


def phase_prep(args, work: Path) -> dict:
    out = work / "prep.joblib"
    if out.exists() and not args.force:
        print(f"prep checkpoint exists: {out}", flush=True)
        return joblib.load(out)

    started = time.time()
    dataset = args.dataset
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]
    scenarios = json.loads((dataset / "scenarios.json").read_text())

    print("smoothing + within-day shape...", flush=True)
    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw
    print(f"  {len(cache.devices)} devices, {time.time()-started:.0f}s", flush=True)

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

    print(f"stride frame (stride={args.stride})...", flush=True)
    frame = build_training_frame(
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
        stride=args.stride,
    )
    print(f"  {len(frame)} cutoffs, {time.time()-started:.0f}s", flush=True)

    print("scenario frame (48 cutoffs, deployment population)...", flush=True)
    scen = build_scenario_frame(cache, shape_cache, devices, eol, scenarios)
    print(
        f"  {len(scen['due'])} rows, {int(scen['due'].sum())} due, "
        f"{int(scen['has_row'].sum())} with feature rows, "
        f"{time.time()-started:.0f}s",
        flush=True,
    )

    print("window bank...", flush=True)
    bank = build_window_bank(frame, cache, FIT_HORIZONS)
    n_cens = len(bank["drop_raw"])
    n_v7 = int((~bank["crossed"]).sum())
    print(
        f"  {n_cens} cens windows / {n_v7} v7 windows "
        f"({n_cens - n_v7} crossing windows), {time.time()-started:.0f}s",
        flush=True,
    )

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # Reconstruction check: shipped artifacts over this frame must land on the
    # known production sums (v7 ~8.45, cens ~10.01 dues/scenario calibrated).
    check = {}
    for name, path in (
        ("v7", REPO_ROOT / "models/v7_wiener.joblib"),
        ("cens", REPO_ROOT / "models/v8_cens.joblib"),
    ):
        if not path.exists():
            continue
        model = joblib.load(path)
        p_raw = predict_scenario(model, scen)
        p_cal = p_raw
        if model.calibration is not None:
            p_cal = np.clip(
                p_raw * model.calibration.factor_for(scen["remaining"]), 0.0, 1.0
            )
        check[name] = {
            "rows": int(len(p_raw)),
            "sum_p_raw_per_scenario": round(float(p_raw.sum() / scen["n_scenarios"]), 3),
            "sum_p_cal_per_scenario": round(float(p_cal.sum() / scen["n_scenarios"]), 3),
            "realized_per_scenario": round(
                float(scen["due"].sum() / scen["n_scenarios"]), 3
            ),
        }
        print(f"  reconstruction {name}: {check[name]}", flush=True)

    prep = {
        "frame": frame,
        "scen": scen,
        "bank": bank,
        "climatology": climatology,
        "reconstruction_check": check,
        "stride": args.stride,
        "building_sizes": devices.groupby("building_id")["device_id"].count().to_dict(),
        "building_eol": (
            devices.assign(has=devices["device_id"].map(lambda d: pd.notna(eol.get(d))))
            .groupby("building_id")["has"]
            .sum()
            .astype(int)
            .to_dict()
        ),
    }
    work.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, out, compress=0)
    print(f"wrote {out} in {time.time()-started:.0f}s", flush=True)
    return prep


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def make_hard_folds(prep: dict) -> dict[str, tuple[str, ...]]:
    sizes = pd.Series(prep["building_sizes"]).sort_values(ascending=False)
    eol = pd.Series(prep["building_eol"]).reindex(sizes.index).fillna(0)
    rate = (eol / sizes).sort_values(ascending=False)

    folds: dict[str, tuple[str, ...]] = {}
    folds["hard_large5"] = tuple(sizes.index[:5])
    folds["hard_small10"] = tuple(sizes.sort_values(kind="stable").index[:10])
    folds["hard_mosteol5"] = tuple(eol.sort_values(ascending=False, kind="stable").index[:5])
    folds["hard_hirate6"] = tuple(rate.index[:6])

    # Buildings whose within-day dV/dT SCALE sits farthest from the fleet:
    # the HVAC-dependence stress. Only buildings with enough stride rows count.
    frame = prep["frame"]
    beta = frame.features[:, BETA30_COL].astype(float)
    shift = {}
    global_median = np.nanmedian(beta)
    for building in np.unique(frame.building):
        rows = frame.building == building
        values = beta[rows]
        values = values[np.isfinite(values)]
        if values.size < 40:
            continue
        shift[building] = abs(np.log(max(np.median(values), 1e-6) / global_median))
    ranked = sorted(shift, key=lambda b: -shift[b])
    folds["hard_betashift5"] = tuple(ranked[:5])
    return folds


def make_loo_folds(prep: dict) -> dict[str, tuple[str, ...]]:
    return {
        f"loo_{building}": (building,)
        for building in sorted(prep["building_sizes"])
    }


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

def phase_fit(args, work: Path, prep: dict) -> None:
    folds: dict[str, tuple[str, ...]] = {}
    if args.scope in ("hard", "all"):
        folds.update(make_hard_folds(prep))
    if args.scope in ("loo", "all"):
        folds.update(make_loo_folds(prep))
    if args.only:
        folds = {k: v for k, v in folds.items() if k in set(args.only)}

    bank, scen = prep["bank"], prep["scen"]
    fits_dir = work / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        (fold, variant)
        for fold in folds
        for variant in VARIANTS
        if not (fits_dir / f"{fold}__{variant}.joblib").exists() or args.force
    ]
    print(f"{len(todo)} fits to run ({len(folds)} folds x {len(VARIANTS)} variants)", flush=True)

    for count, (fold, variant) in enumerate(todo):
        heldout = folds[fold]
        rows, target = variant_rows(bank, variant)
        train = rows & ~np.isin(bank["building"], list(heldout))
        started = time.time()
        model = WienerModel.fit(
            bank["design"][train],
            target[train],
            prep["climatology"],
            params={"max_iter": args.max_iter},
        )
        p_raw = predict_scenario(model, scen)
        seconds = time.time() - started
        joblib.dump(
            {
                "fold": fold,
                "variant": variant,
                "heldout": list(heldout),
                "p_raw": p_raw.astype(np.float32),
                "n_train_windows": int(train.sum()),
                "seconds": round(seconds, 1),
                "max_iter": args.max_iter,
            },
            fits_dir / f"{fold}__{variant}.joblib",
        )
        print(
            f"  [{count+1}/{len(todo)}] {fold} {variant}: "
            f"{int(train.sum())} windows, {seconds:.0f}s",
            flush=True,
        )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _precision_at(order: np.ndarray, due: np.ndarray, k: int) -> float | None:
    if due.size < k:
        return None
    return round(float(due[order[:k]].mean()), 4)


def _within_scenario_rank(scen_index: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Percentile of p within its own scenario (1.0 = riskiest row that week)."""
    out = np.zeros_like(p, dtype=float)
    for s in np.unique(scen_index):
        mask = scen_index == s
        out[mask] = rankdata(p[mask], method="average") / mask.sum()
    return out


def _blocks(scen_index: np.ndarray, p: np.ndarray, due: np.ndarray) -> dict:
    out = {}
    for lo, hi, label in ((0, 16, "early"), (16, 32, "mid"), (32, 48, "late")):
        mask = (scen_index >= lo) & (scen_index < hi)
        n = hi - lo
        predicted, actual = float(p[mask].sum()), float(due[mask].sum())
        out[label] = {
            "predicted_per_scenario": round(predicted / n, 2),
            "realized_per_scenario": round(actual / n, 2),
            "ratio": round(predicted / max(actual, 1.0), 3),
        }
    return out


def fold_metrics(scen: dict, p_raw: np.ndarray, heldout: list[str]) -> dict:
    held = np.isin(scen["building"], heldout)
    train = ~held
    n_scen = scen["n_scenarios"]

    # Production procedure: RemainingCalibration fitted on training buildings'
    # scenario rows, then applied everywhere by remaining observation days.
    calibration = RemainingCalibration.fit(
        scen["remaining"][train], p_raw[train], scen["due"][train].astype(float)
    )
    p_cal = np.clip(p_raw * calibration.factor_for(scen["remaining"]), 0.0, 1.0)

    due_h = scen["due"][held].astype(int)
    p_raw_h, p_cal_h = p_raw[held], p_cal[held]
    scen_h = scen["scenario_index"][held]
    realized = float(due_h.sum())

    out: dict = {
        "n_rows": int(held.sum()),
        "n_due": int(realized),
        "sum_p_raw_per_scenario": round(float(p_raw_h.sum() / n_scen), 3),
        "sum_p_cal_per_scenario": round(float(p_cal_h.sum() / n_scen), 3),
        "realized_per_scenario": round(realized / n_scen, 3),
        "inflation_raw": round(float(p_raw_h.sum() / max(realized, 1.0)), 3),
        "inflation_cal": round(float(p_cal_h.sum() / max(realized, 1.0)), 3),
        "blocks_cal": _blocks(scen_h, p_cal_h, due_h),
        "calibration_factors": [round(f, 3) for f in calibration.factors],
    }

    if realized > 0:
        out["pr_auc_level"] = round(float(average_precision_score(due_h, p_raw_h)), 4)
        rank_h = _within_scenario_rank(scen_h, p_raw_h)
        out["pr_auc_rank"] = round(float(average_precision_score(due_h, rank_h)), 4)
        order_level = np.lexsort((-rank_h, -p_raw_h))
        order_rank = np.lexsort((-p_raw_h, -rank_h))
        out["precision_at_k_level"] = {
            k: _precision_at(order_level, due_h, k) for k in (5, 10, 15)
        }
        out["precision_at_k_rank"] = {
            k: _precision_at(order_rank, due_h, k) for k in (5, 10, 15)
        }
        out["base_rate"] = round(float(due_h.mean()), 5)

    for label, values in (("raw", p_raw_h), ("cal", p_cal_h)):
        bucket = values >= 0.7
        out[f"top_bucket_{label}"] = {
            "n": int(bucket.sum()),
            "predicted_mean": round(float(values[bucket].mean()), 3)
            if bucket.any()
            else None,
            "realized_mean": round(float(due_h[bucket].mean()), 3)
            if bucket.any()
            else None,
        }

    # ---- policies: absolute level vs within-scenario rank + quota ----------
    due_t = scen["due"][train].astype(int)
    p_cal_t = p_cal[train]

    policies = {}
    for tau in (0.35, 0.5):
        sel = p_cal_h >= tau
        policies[f"level_tau_{tau}"] = _policy(sel, due_h, n_scen, realized)
    # tau tuned on TRAINING buildings so that selected count == realized there:
    # the level policy a team would actually deploy.
    if due_t.sum() > 0 and (p_cal_t > 0).any():
        quantile = 1.0 - float(due_t.sum()) / len(due_t)
        tau_star = float(np.quantile(p_cal_t, quantile))
        sel = p_cal_h >= tau_star
        policies["level_tau_train_matched"] = _policy(sel, due_h, n_scen, realized)
        policies["level_tau_train_matched"]["tau"] = round(tau_star, 4)

    # rank + quota: per-scenario quota from the TRAINING base rate only.
    base_rate = float(due_t.sum()) / max(len(due_t), 1)
    sel = np.zeros(len(due_h), dtype=bool)
    for s in np.unique(scen_h):
        mask = scen_h == s
        quota = int(round(base_rate * mask.sum()))
        if quota <= 0:
            continue
        rows = np.flatnonzero(mask)
        top = rows[np.argsort(-p_raw_h[rows], kind="stable")[:quota]]
        sel[top] = True
    policies["rank_quota_train_rate"] = _policy(sel, due_h, n_scen, realized)
    policies["rank_quota_train_rate"]["train_base_rate"] = round(base_rate, 5)
    out["policies"] = policies

    per_scenario = {
        "predicted_cal": [
            round(float(p_cal_h[scen_h == s].sum()), 3) for s in range(n_scen)
        ],
        "realized": [int(due_h[scen_h == s].sum()) for s in range(n_scen)],
    }
    out["per_scenario"] = per_scenario
    return out


def _policy(selected: np.ndarray, due: np.ndarray, n_scen: int, realized: float) -> dict:
    chosen = int(selected.sum())
    hits = int(due[selected].sum())
    return {
        "swaps_per_scenario": round(chosen / n_scen, 3),
        "precision": round(hits / chosen, 4) if chosen else None,
        "recall": round(hits / realized, 4) if realized else None,
    }


def pooled_loo(scen: dict, fits: dict, variant: str, loo_folds: dict) -> dict | None:
    """Stitch every building's held-out prediction into one OOF vector."""
    p_raw = np.full(len(scen["due"]), np.nan)
    p_cal = np.full(len(scen["due"]), np.nan)
    for fold, heldout in loo_folds.items():
        record = fits.get((fold, variant))
        if record is None:
            return None
        held = np.isin(scen["building"], heldout)
        train = ~held
        calibration = RemainingCalibration.fit(
            scen["remaining"][train],
            record["p_raw"][train],
            scen["due"][train].astype(float),
        )
        p_raw[held] = record["p_raw"][held]
        p_cal[held] = np.clip(
            record["p_raw"][held] * calibration.factor_for(scen["remaining"][held]),
            0.0,
            1.0,
        )
    if np.isnan(p_raw).any():
        return None

    due = scen["due"].astype(int)
    n_scen = scen["n_scenarios"]
    realized = float(due.sum())
    rank = _within_scenario_rank(scen["scenario_index"], p_raw)
    order_level = np.lexsort((-rank, -p_raw))
    order_rank = np.lexsort((-p_raw, -rank))
    out = {
        "n_rows": int(len(due)),
        "n_due": int(realized),
        "sum_p_raw_per_scenario": round(float(p_raw.sum() / n_scen), 3),
        "sum_p_cal_per_scenario": round(float(p_cal.sum() / n_scen), 3),
        "realized_per_scenario": round(realized / n_scen, 3),
        "inflation_raw": round(float(p_raw.sum() / realized), 3),
        "inflation_cal": round(float(p_cal.sum() / realized), 3),
        "pr_auc_level": round(float(average_precision_score(due, p_raw)), 4),
        "pr_auc_rank": round(float(average_precision_score(due, rank)), 4),
        "precision_at_k_level": {
            k: _precision_at(order_level, due, k) for k in (25, 50, 100)
        },
        "precision_at_k_rank": {
            k: _precision_at(order_rank, due, k) for k in (25, 50, 100)
        },
        "blocks_cal": _blocks(scen["scenario_index"], p_cal, due),
    }
    for label, values in (("raw", p_raw), ("cal", p_cal)):
        bucket = values >= 0.7
        out[f"top_bucket_{label}"] = {
            "n": int(bucket.sum()),
            "predicted_mean": round(float(values[bucket].mean()), 3)
            if bucket.any()
            else None,
            "realized_mean": round(float(due[bucket].mean()), 3)
            if bucket.any()
            else None,
        }

    per_building = {}
    for building in np.unique(scen["building"]):
        rows = scen["building"] == building
        realized_b = float(due[rows].sum())
        per_building[building] = {
            "rows": int(rows.sum()),
            "due": int(realized_b),
            "sum_p_cal": round(float(p_cal[rows].sum()), 2),
            "inflation_cal": round(float(p_cal[rows].sum() / realized_b), 2)
            if realized_b
            else None,
        }
    out["per_building"] = per_building

    # Does the probability LEVEL allocate volume correctly across buildings?
    sums = np.asarray([v["sum_p_cal"] for v in per_building.values()], dtype=float)
    dues = np.asarray([v["due"] for v in per_building.values()], dtype=float)
    out["building_volume_spearman"] = round(
        float(
            np.corrcoef(rankdata(sums), rankdata(dues))[0, 1]
        ),
        3,
    )

    # Equal-volume selectors: same total swap count (= realized), allocated by
    # absolute level vs by per-scenario quota. Isolates allocation quality from
    # volume errors.
    k = int(realized)
    sel_level = np.zeros(len(due), dtype=bool)
    sel_level[np.argsort(-p_cal, kind="stable")[:k]] = True
    quota, extra = divmod(k, n_scen)
    sel_quota = np.zeros(len(due), dtype=bool)
    leftovers = []
    for s in range(n_scen):
        rows_s = np.flatnonzero(scen["scenario_index"] == s)
        ordered = rows_s[np.argsort(-p_cal[rows_s], kind="stable")]
        sel_quota[ordered[:quota]] = True
        leftovers.extend(ordered[quota:].tolist())
    if extra:
        leftovers = np.asarray(leftovers)
        sel_quota[leftovers[np.argsort(-p_cal[leftovers], kind="stable")[:extra]]] = True
    out["equal_volume_selectors"] = {
        "k_total": k,
        "global_by_level": {
            "precision": round(float(due[sel_level].mean()), 4),
            "recall": round(float(due[sel_level].sum() / realized), 4),
        },
        "per_scenario_quota": {
            "quota": quota,
            "precision": round(float(due[sel_quota].mean()), 4),
            "recall": round(float(due[sel_quota].sum() / realized), 4),
        },
    }
    return out


# --------------------------------------------------------------------------
# feature fragility
# --------------------------------------------------------------------------

def permutation_importance(prep: dict, sample: int = 30000, seed: int = 0) -> list[dict]:
    """Which columns the production cens drift model actually leans on."""
    path = REPO_ROOT / "models/v8_cens.joblib"
    if not path.exists():
        return []
    model = joblib.load(path)
    bank = prep["bank"]
    _, target = variant_rows(bank, "cens")
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(target), size=min(sample, len(target)), replace=False)
    design = bank["design"][rows].astype(np.float64)
    truth = target[rows]
    base = float(np.mean(np.abs(model.drift.predict(design) - truth)))
    names = list(FEATURE_NAMES) + ["horizon"]
    out = []
    for column in range(design.shape[1]):
        shuffled = design.copy()
        shuffled[:, column] = shuffled[rng.permutation(len(rows)), column]
        mae = float(np.mean(np.abs(model.drift.predict(shuffled) - truth)))
        out.append({"feature": names[column], "delta_mae": round(mae - base, 6)})
    out.sort(key=lambda r: -r["delta_mae"])
    return out


def ks_shift(scen: dict, heldout: list[str], features: list[str]) -> dict:
    held = np.isin(scen["building"], heldout) & scen["has_row"]
    train = ~np.isin(scen["building"], heldout) & scen["has_row"]
    out = {}
    for name in features:
        column = FEATURE_NAMES.index(name)
        a = scen["features"][held, column].astype(float)
        b = scen["features"][train, column].astype(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 20 or b.size < 20:
            continue
        stat = float(ks_2samp(a, b).statistic)
        iqr = float(np.subtract(*np.percentile(b, [75, 25]))) or 1e-9
        out[name] = {
            "ks": round(stat, 3),
            "median_heldout": round(float(np.median(a)), 5),
            "median_train": round(float(np.median(b)), 5),
            "median_shift_in_train_iqr": round(
                float((np.median(a) - np.median(b)) / iqr), 3
            ),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["ks"]))


def beta_dispersion(prep: dict) -> dict:
    """Per-building medians: is beta's SCALE building-bound while the RISE is not?"""
    frame = prep["frame"]
    scen = prep["scen"]

    def stats(values_by_building: dict[str, float]) -> dict:
        values = np.asarray(
            [v for v in values_by_building.values() if np.isfinite(v)], dtype=float
        )
        if values.size < 3:
            return {}
        spread = {
            "n_buildings": int(values.size),
            "median_of_medians": round(float(np.median(values)), 5),
            "iqr_over_median": round(
                float(np.subtract(*np.percentile(values, [75, 25])))
                / max(abs(np.median(values)), 1e-9),
                3,
            ),
            "cv": round(float(np.std(values) / max(abs(np.mean(values)), 1e-9)), 3),
            "max_over_min": round(
                float(np.max(values) / np.min(values)), 2
            )
            if np.min(values) > 0
            else None,
        }
        return spread

    report: dict = {"per_building_medians": {}, "dispersion": {}}
    columns = {
        "beta_30": BETA30_COL,
        "beta_rise": BETARISE_COL,
        "v_std_30": FEATURE_NAMES.index("v_std_30"),
        "v_std_rise": FEATURE_NAMES.index("v_std_rise"),
        "voltage": VOLTAGE_COL,
    }
    for name, column in columns.items():
        medians = {}
        for building in np.unique(frame.building):
            rows = frame.building == building
            values = frame.features[rows, column].astype(float)
            values = values[np.isfinite(values)]
            if values.size >= 40:
                medians[building] = float(np.median(values))
        report["per_building_medians"][name] = {
            k: round(v, 5) for k, v in sorted(medians.items())
        }
        report["dispersion"][name] = stats(medians)

    # Margin quantile within fleet-day: the building-invariant voltage candidate.
    margin = scen["features"][:, VOLTAGE_COL].astype(float) - EOL_THRESHOLD
    quantile = np.full(len(margin), np.nan)
    for s in np.unique(scen["scenario_index"]):
        rows = np.flatnonzero((scen["scenario_index"] == s) & scen["has_row"])
        finite = rows[np.isfinite(margin[rows])]
        if finite.size:
            quantile[finite] = rankdata(margin[finite], method="average") / finite.size
    medians_q, medians_v = {}, {}
    for building in np.unique(scen["building"]):
        rows = (scen["building"] == building) & np.isfinite(quantile)
        if rows.sum() >= 40:
            medians_q[building] = float(np.median(quantile[rows]))
            medians_v[building] = float(
                np.median(margin[rows][np.isfinite(margin[rows])])
            )
    report["dispersion"]["margin_scenario_rows"] = stats(medians_v)
    report["dispersion"]["margin_quantile_within_fleet_day"] = stats(medians_q)
    return report


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def phase_report(args, work: Path, prep: dict) -> None:
    scen = prep["scen"]
    fits_dir = work / "fits"
    fits: dict[tuple[str, str], dict] = {}
    for path in sorted(fits_dir.glob("*.joblib")):
        record = joblib.load(path)
        fits[(record["fold"], record["variant"])] = record

    hard_folds = make_hard_folds(prep)
    loo_folds = make_loo_folds(prep)

    results: dict = {
        "config": {
            "stride": prep["stride"],
            "max_iter": args.max_iter,
            "n_windows_cens": int(len(prep["bank"]["drop_raw"])),
            "n_windows_v7": int((~prep["bank"]["crossed"]).sum()),
            "scenario_rows": int(len(scen["due"])),
            "scenario_due_total": int(scen["due"].sum()),
        },
        "reconstruction_check": prep["reconstruction_check"],
        "hard_folds": {},
        "loo_pooled": {},
        "loo_per_fold": {},
    }

    # The 5-fold-by-building OOF reference the team validated with (raw model,
    # scenario cutoffs, same 19890 rows): the number the near-twin folds gave.
    five_fold = {}
    for variant, name in (("v7", "prod_v7_cal.json"), ("cens", "prod_cens_cal.json")):
        path = REPO_ROOT / "outputs" / name
        if path.exists():
            record = json.loads(path.read_text())
            five_fold[variant] = {
                "sum_p_raw_per_scenario": record["mean_predicted_per_scenario"],
                "realized_per_scenario": record["mean_actual_per_scenario"],
                "inflation_raw": record["overall_ratio"],
            }
    results["five_fold_oof_reference"] = five_fold

    for fold, heldout in hard_folds.items():
        entry = {"heldout": list(heldout)}
        sizes = prep["building_sizes"]
        eol = prep["building_eol"]
        entry["heldout_devices"] = int(sum(sizes.get(b, 0) for b in heldout))
        entry["heldout_eol_events"] = int(sum(eol.get(b, 0) for b in heldout))
        for variant in VARIANTS:
            record = fits.get((fold, variant))
            if record is None:
                continue
            entry[variant] = fold_metrics(scen, record["p_raw"].astype(float), list(heldout))
            entry[variant]["n_train_windows"] = record["n_train_windows"]
            entry[variant]["fit_seconds"] = record["seconds"]
        if len(entry) > 3:
            results["hard_folds"][fold] = entry

    for variant in VARIANTS:
        pooled = pooled_loo(scen, fits, variant, loo_folds)
        if pooled is not None:
            results["loo_pooled"][variant] = pooled

    for fold, heldout in loo_folds.items():
        entry = {}
        for variant in VARIANTS:
            record = fits.get((fold, variant))
            if record is None:
                continue
            metrics = fold_metrics(scen, record["p_raw"].astype(float), list(heldout))
            entry[variant] = {
                k: metrics[k]
                for k in (
                    "n_rows",
                    "n_due",
                    "sum_p_cal_per_scenario",
                    "realized_per_scenario",
                    "inflation_raw",
                    "inflation_cal",
                    "pr_auc_level",
                    "pr_auc_rank",
                )
                if k in metrics
            }
        if entry:
            results["loo_per_fold"][fold] = entry

    print("permutation importance (production cens drift)...", flush=True)
    importance = permutation_importance(prep)
    results["drift_permutation_importance_top15"] = importance[:15]

    top_names = [
        r["feature"] for r in importance[:10] if r["feature"] in FEATURE_NAMES
    ]
    forced = [
        "voltage",
        "voltage_compensated",
        "staleness",
        "slope_30",
        "slope_90",
        "beta_30",
        "beta_7",
        "days_below_2.45",
        "days_below_2.50",
        "v_std_30",
        "beta_rise",
        "v_std_rise",
    ]
    ks_features = list(dict.fromkeys(top_names + forced))

    # Worst-transferring hard folds by calibrated inflation distance from 1.
    ranked = sorted(
        (
            (fold, entry)
            for fold, entry in results["hard_folds"].items()
            if "cens" in entry
        ),
        key=lambda kv: -abs(np.log(max(kv[1]["cens"]["inflation_cal"], 1e-3))),
    )
    results["feature_shift_worst_folds"] = {}
    for fold, entry in ranked[:3]:
        results["feature_shift_worst_folds"][fold] = ks_shift(
            scen, entry["heldout"], ks_features
        )

    print("beta dispersion...", flush=True)
    results["beta_dispersion"] = beta_dispersion(prep)

    out_json = REPO_ROOT / "outputs/transfer_stress.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}", flush=True)

    out_md = REPO_ROOT / "outputs/transfer_stress.md"
    out_md.write_text(render_markdown(results))
    print(f"wrote {out_md}", flush=True)


def _verdict_lines(results: dict) -> list[str]:
    lines: list[str] = []
    hard = results.get("hard_folds", {})
    pooled = results.get("loo_pooled", {})
    five = results.get("five_fold_oof_reference", {})
    recon = results.get("reconstruction_check", {})

    def collect(variant: str, key: str) -> list[float]:
        return [
            entry[variant][key]
            for entry in hard.values()
            if variant in entry and key in entry[variant]
        ]

    # (a) ranking transfer on hard holdouts
    for metric, label in (("pr_auc_level", "PR-AUC"), ("pr_auc_rank", "rank PR-AUC")):
        v7 = collect("v7", metric)
        cens = collect("cens", metric)
        if v7 and cens:
            wins = sum(c > v for c, v in zip(cens, v7))
            lines.append(
                f"(a) {label} on {len(v7)} hard holdouts: cens mean "
                f"{np.mean(cens):.3f} vs v7 {np.mean(v7):.3f} "
                f"(cens better on {wins}/{len(v7)})."
            )

    # (b) sum-p inflation ladder: in-sample -> 5-fold OOF -> LOO -> hard folds
    for variant in VARIANTS:
        parts = []
        if variant in recon:
            parts.append(f"in-sample raw {recon[variant]['sum_p_raw_per_scenario']}")
        if variant in five:
            parts.append(f"5-fold OOF raw {five[variant]['sum_p_raw_per_scenario']}")
        if variant in pooled:
            parts.append(
                f"LOO raw {pooled[variant]['sum_p_raw_per_scenario']} "
                f"(x{pooled[variant]['inflation_raw']}), "
                f"LOO calibrated {pooled[variant]['sum_p_cal_per_scenario']} "
                f"(x{pooled[variant]['inflation_cal']})"
            )
        worst = max(
            (entry[variant]["inflation_cal"] for entry in hard.values() if variant in entry),
            default=None,
        )
        if worst is not None:
            parts.append(f"worst hard-fold calibrated x{worst}")
        lines.append(f"(b) {variant}: " + "; ".join(parts) + ".")

    # (c) allocation robustness
    for variant in VARIANTS:
        entry = pooled.get(variant, {})
        selectors = entry.get("equal_volume_selectors")
        if selectors:
            lines.append(
                f"(c) {variant} at equal volume k={selectors['k_total']}: "
                f"global-by-level precision {selectors['global_by_level']['precision']}, "
                f"per-scenario quota {selectors['per_scenario_quota']['precision']}; "
                f"building-volume Spearman {entry.get('building_volume_spearman')}; "
                f"top-bucket >=0.7 realized {entry.get('top_bucket_cal', {}).get('realized_mean')} "
                f"on n={entry.get('top_bucket_cal', {}).get('n')}."
            )

    dispersion = results.get("beta_dispersion", {}).get("dispersion", {})
    if "beta_30" in dispersion and "beta_rise" in dispersion:
        lines.append(
            "(d) per-building median dispersion: beta_30 CV "
            f"{dispersion['beta_30']['cv']} (max/min {dispersion['beta_30']['max_over_min']}x) "
            f"vs beta_rise CV {dispersion['beta_rise']['cv']} "
            f"(max/min {dispersion['beta_rise']['max_over_min']}x); v_std_30 CV "
            f"{dispersion['v_std_30']['cv']} vs v_std_rise CV {dispersion['v_std_rise']['cv']}. "
            "Substitute the within-day SCALE features with their rise ratios."
        )
    return lines


def render_markdown(results: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Transfer stress: leave-building(s)-out at scenario cutoffs")
    add("")
    add("## Verdict")
    add("")
    for line in _verdict_lines(results):
        add(f"- {line}")
    add("")
    add(
        "Both target variants refit per fold on windows excluding the held-out "
        "buildings (stride {stride}, max_iter {mi}); all metrics are on held-out "
        "buildings' scenario rows only. `cal` = RemainingCalibration fitted on the "
        "fold's training buildings, i.e. the production procedure applied to a "
        "fresh building.".format(
            stride=results["config"]["stride"], mi=results["config"]["max_iter"]
        )
    )
    add("")
    add("## Reconstruction check (shipped artifacts over the rebuilt scenario frame)")
    add("")
    add("```json")
    add(json.dumps(results["reconstruction_check"], indent=2))
    add("```")
    add("")

    add("## Hard grouped holdouts")
    add("")
    header = (
        "| fold | held-out (dev/EOL) | variant | sum p/scen raw | cal | realized "
        "| infl raw | infl cal | PR-AUC lvl | PR-AUC rank | P@5 lvl | P@5 rank "
        "| P@10 lvl | P@10 rank | top>=0.7 cal (n, realized) |"
    )
    add(header)
    add("|" + "---|" * 15)
    for fold, entry in results["hard_folds"].items():
        for variant in VARIANTS:
            metrics = entry.get(variant)
            if not metrics:
                continue
            pk_l = metrics.get("precision_at_k_level", {})
            pk_r = metrics.get("precision_at_k_rank", {})
            top = metrics["top_bucket_cal"]
            add(
                f"| {fold} | {entry['heldout_devices']}/{entry['heldout_eol_events']} "
                f"| {variant} | {metrics['sum_p_raw_per_scenario']} "
                f"| {metrics['sum_p_cal_per_scenario']} | {metrics['realized_per_scenario']} "
                f"| {metrics['inflation_raw']} | {metrics['inflation_cal']} "
                f"| {metrics.get('pr_auc_level', '-')} | {metrics.get('pr_auc_rank', '-')} "
                f"| {pk_l.get(5, '-')} | {pk_r.get(5, '-')} "
                f"| {pk_l.get(10, '-')} | {pk_r.get(10, '-')} "
                f"| {top['n']}, {top['realized_mean']} |"
            )
    add("")

    add("### Policies on held-out rows (level threshold vs rank+quota)")
    add("")
    add(
        "| fold | variant | policy | swaps/scen | precision | recall |"
    )
    add("|" + "---|" * 6)
    for fold, entry in results["hard_folds"].items():
        for variant in VARIANTS:
            metrics = entry.get(variant)
            if not metrics:
                continue
            for policy, values in metrics["policies"].items():
                add(
                    f"| {fold} | {variant} | {policy} | {values['swaps_per_scenario']} "
                    f"| {values['precision']} | {values['recall']} |"
                )
    add("")

    add("## Pooled leave-one-building-out (24 folds, every row out-of-building)")
    add("")
    if results.get("five_fold_oof_reference"):
        add("5-fold-by-building OOF reference (the validation the team used):")
        add("```json")
        add(json.dumps(results["five_fold_oof_reference"], indent=2))
        add("```")
        add("")
    for variant, pooled in results.get("loo_pooled", {}).items():
        add(f"### {variant}")
        add("```json")
        slim = {k: v for k, v in pooled.items() if k != "per_building"}
        add(json.dumps(slim, indent=2))
        add("```")
        add("")
        worst = sorted(
            (
                (b, v)
                for b, v in pooled["per_building"].items()
                if v["inflation_cal"] is not None
            ),
            key=lambda kv: -kv[1]["inflation_cal"],
        )[:6]
        add("worst buildings by calibrated inflation: " + ", ".join(
            f"{b} x{v['inflation_cal']} ({v['due']} due)" for b, v in worst
        ))
        add("")

    if results.get("loo_per_fold"):
        add("### Per-building LOO (calibrated sum p vs realized, per scenario)")
        add("")
        add("| building | variant | rows | due | sum p cal/scen | realized/scen | infl cal | PR-AUC lvl | PR-AUC rank |")
        add("|" + "---|" * 9)
        for fold, entry in results["loo_per_fold"].items():
            for variant, metrics in entry.items():
                add(
                    f"| {fold.replace('loo_', '')} | {variant} | {metrics['n_rows']} "
                    f"| {metrics['n_due']} | {metrics['sum_p_cal_per_scenario']} "
                    f"| {metrics['realized_per_scenario']} | {metrics['inflation_cal']} "
                    f"| {metrics.get('pr_auc_level', '-')} | {metrics.get('pr_auc_rank', '-')} |"
                )
        add("")

    add("## Drift permutation importance (production cens model)")
    add("")
    add("```json")
    add(json.dumps(results.get("drift_permutation_importance_top15", []), indent=2))
    add("```")
    add("")

    add("## Feature shift on the worst-transferring holdouts (KS held-out vs train)")
    add("")
    for fold, table in results.get("feature_shift_worst_folds", {}).items():
        add(f"### {fold}")
        add("")
        add("| feature | KS | median held-out | median train | shift / train IQR |")
        add("|---|---|---|---|---|")
        for name, row in table.items():
            add(
                f"| {name} | {row['ks']} | {row['median_heldout']} "
                f"| {row['median_train']} | {row['median_shift_in_train_iqr']} |"
            )
        add("")

    add("## Beta scale vs beta rise: per-building dispersion")
    add("")
    add("```json")
    add(json.dumps(results.get("beta_dispersion", {}).get("dispersion", {}), indent=2))
    add("```")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset/train")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--phase", choices=("all", "prep", "fit", "report"), default="all")
    parser.add_argument("--scope", choices=("hard", "loo", "all"), default="all")
    parser.add_argument("--only", nargs="*", default=None, help="fold names to fit")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    prep = phase_prep(args, args.work)
    if args.phase in ("all", "fit"):
        phase_fit(args, args.work, prep)
    if args.phase in ("all", "report"):
        phase_report(args, args.work, prep)


if __name__ == "__main__":
    main()
