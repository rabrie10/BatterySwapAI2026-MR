"""V12 transfer stress: judge the variant feature sets on hard holdouts.

The public leaderboard punished two locally-validated models, and
``tools/transfer_stress.py`` measured why: probability levels inflate x1.05-1.30
out-of-building and the fragile features are the absolute-scale ones
(docs/TRANSFER_STRESS.md). The V12 arms replace fragile features with
within-device contrasts and add the raw-daily channel (bsai/rawdaily.py); see
ARM_FEATURES. This harness answers ONE question before any planner run: does a
V12 arm rank and level better than the incumbents on buildings its training
never saw?

Judged exactly on transfer_stress's grounds, reusing its fold definitions,
metrics and (alignment-verified) stored incumbent predictions:

  hard grouped holdouts  hard_large5 / hard_small10 / hard_mosteol5 /
                         hard_hirate6 / hard_betashift5
  pooled LOO             24 folds, every scenario row out-of-building

Success gate: mean hard-holdout PR-AUC > cens's 0.428 AND sum-p inflation in
v7's class (pooled LOO raw <= ~1.10).

Targets are censor-aware ("cens" rule: windows may end past the crossing and a
window containing the crossing counts at least the starting margin), matching
production bsai/wiener.py. Fidelity: stride 8 / max_iter 150, same as the
incumbent measurements.

Usage (phases checkpoint into --work, safe to re-run):
    python tools/v12_transfer.py --phase prep
    python tools/v12_transfer.py --phase fit --scope hard
    python tools/v12_transfer.py --phase fit --scope loo
    python tools/v12_transfer.py --phase report
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
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import transfer_stress as ts  # fold definitions, metrics, window bank
from v12_fit import KNEE_DAYS, fit_wiener_weighted, knee_weights, window_end_sidecar
from v12_frame import build_variant_training_frame

from batteryswap_public.utils import load_devices

from bsai.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    FEATURE_NAMES_INVARIANT,
    FEATURE_NAMES_INVARIANT2,
    RAW_FEATURE_NAMES,
    DeviceView,
    FeatureContext,
    feature_row,
    fleet_climatology,
)
from bsai.rawdaily import RawDailyCache
from bsai.shape import ShapeCache, align_to
from bsai.smoothing import SmoothingCache
from bsai.v12_rawany import RAW_ANY_FEATURE_NAMES, RawAnyCache
from bsai.wiener import FIT_HORIZONS, WienerModel

from functools import partial

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42
N_BASE = len(FEATURE_NAMES)
_RAW_ALL = list(RAW_FEATURE_NAMES) + list(RAW_ANY_FEATURE_NAMES)

# Arms under judgment. "v12" is round 1 exactly as measured (absolute scales
# dropped, no raw channels). The v12b family is the single allowed feature
# iteration -- keep the absolute scales, replace only the temperature levels
# with within-device contrasts, carry both raw channels -- plus the
# coordinator's knee re-weighting sweep (roadblock report L2 ii-iv, w_knee in
# {1, 3}) and ablations isolating the raw channels.
ARM_FEATURES: dict[str, list[str]] = {
    "v12": list(FEATURE_NAMES_INVARIANT),
    "v12b": list(FEATURE_NAMES_INVARIANT2),
    "v12b_k1": list(FEATURE_NAMES_INVARIANT2),
    "v12b_k3": list(FEATURE_NAMES_INVARIANT2),
    "v12b_noany": [
        name for name in FEATURE_NAMES_INVARIANT2 if name not in RAW_ANY_FEATURE_NAMES
    ],
    "v12b_noraw": [
        name for name in FEATURE_NAMES_INVARIANT2 if name not in _RAW_ALL
    ],
}
ARM_KNEE_WEIGHT: dict[str, float] = {"v12b_k1": 1.0, "v12b_k3": 3.0}
ARM_COLUMNS = {
    arm: np.asarray([FEATURE_NAMES_EXTENDED.index(n) for n in names], dtype=int)
    for arm, names in ARM_FEATURES.items()
}
DEFAULT_ARMS = ("v12b", "v12b_k1", "v12b_k3")
INCUMBENTS = ("v7", "cens")

# Success gate constants, from docs/TRANSFER_STRESS.md.
CENS_HARD_PRAUC_MEAN = 0.428
V7_LOO_INFLATION_RAW = 1.052
INFLATION_GATE = 1.10

DEFAULT_WORK = ts.DEFAULT_WORK.parent / "v12_transfer"
OLD_FITS_DIR = ts.DEFAULT_WORK / "fits"


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


# --------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------

def build_scenario_frame_extended(
    cache, shape_cache, raw_cache, raw_any_cache, devices, eol, scenarios
) -> dict:
    """ts.build_scenario_frame with extended rows (base columns first)."""
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
        for k in ("scenario_index", "device", "building", "remaining", "due", "has_row")
    }
    feature_rows: list[list[float] | None] = []

    for s_index, scenario in enumerate(scenarios):
        start = pd.Timestamp(scenario["start_time"])
        start_ordinal = _ordinal(start)
        horizon_end = start + pd.Timedelta(days=DECISION_HORIZON)
        for device_id in devices["device_id"]:
            moment = eol.get(device_id)
            if not pd.isna(moment) and moment <= start:
                continue
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
                    index = min(index, len(series) - 1)
                    row = feature_row(
                        view,
                        index,
                        series.origin + index,
                        context,
                        shape_view,
                        variant="extended",
                        raw=partial(raw_cache.features_at, device_id)
                        if raw_cache is not None
                        else None,
                        raw_any=partial(raw_any_cache.features_at, device_id)
                        if raw_any_cache is not None
                        else None,
                    )
            columns["scenario_index"].append(s_index)
            columns["device"].append(device_id)
            columns["building"].append(building_of.get(device_id, ""))
            columns["remaining"].append(remaining)
            columns["due"].append(due)
            columns["has_row"].append(row is not None)
            feature_rows.append(row)

    n = len(feature_rows)
    features = np.full((n, len(FEATURE_NAMES_EXTENDED)), np.nan, dtype=np.float32)
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


def check_old_alignment(scen: dict) -> bool:
    """Are transfer_stress's stored incumbent predictions valid on this frame?"""
    old_path = ts.DEFAULT_WORK / "prep.joblib"
    if not old_path.exists():
        print("  no old prep checkpoint; incumbents would need refitting", flush=True)
        return False
    old = joblib.load(old_path)["scen"]
    keys_equal = (
        np.array_equal(old["due"], scen["due"])
        and np.array_equal(old["scenario_index"], scen["scenario_index"])
        and np.array_equal(old["device"], scen["device"])
        and np.array_equal(old["building"], scen["building"])
        and np.allclose(old["remaining"], scen["remaining"])
        and np.array_equal(old["has_row"], scen["has_row"])
    )
    base_equal = bool(
        np.allclose(
            old["features"], scen["features"][:, :N_BASE], equal_nan=True, atol=0.0
        )
    )
    print(f"  old-prep alignment: keys={keys_equal} base_features={base_equal}", flush=True)
    return keys_equal and base_equal


def phase_prep(args, work: Path) -> dict:
    out = work / "v12_prep.joblib"
    if out.exists() and not args.force:
        prep = joblib.load(out)
        if prep["scen"]["features"].shape[1] == len(FEATURE_NAMES_EXTENDED):
            print(f"prep checkpoint exists: {out}", flush=True)
            return prep
        print(
            f"prep checkpoint is stale ({prep['scen']['features'].shape[1]} cols "
            f"vs {len(FEATURE_NAMES_EXTENDED)} expected); rebuilding",
            flush=True,
        )

    started = time.time()
    dataset = args.dataset
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]
    scenarios = json.loads((dataset / "scenarios.json").read_text())

    print("smoothing + within-day shape + raw dailies (filtered and any-temp)...", flush=True)
    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    raw_cache = RawDailyCache()
    raw_cache.update(raw)
    raw_any_cache = RawAnyCache()
    raw_any_cache.update(raw)
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

    print(f"extended stride frame (stride={args.stride})...", flush=True)
    frame = build_variant_training_frame(
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
        raw_cache=raw_cache,
        raw_any_cache=raw_any_cache,
        variant="extended",
        stride=args.stride,
    )
    assert frame.features.shape[1] == len(FEATURE_NAMES_EXTENDED)
    print(f"  {len(frame)} cutoffs x {frame.features.shape[1]}, {time.time()-started:.0f}s", flush=True)

    print("extended scenario frame...", flush=True)
    scen = build_scenario_frame_extended(
        cache, shape_cache, raw_cache, raw_any_cache, devices, eol, scenarios
    )
    print(
        f"  {len(scen['due'])} rows, {int(scen['due'].sum())} due, "
        f"{int(scen['has_row'].sum())} with rows, {time.time()-started:.0f}s",
        flush=True,
    )

    print("window bank...", flush=True)
    bank = ts.build_window_bank(frame, cache, FIT_HORIZONS)
    print(
        f"  {len(bank['drop_raw'])} cens windows / {int((~bank['crossed']).sum())} v7 windows, "
        f"{time.time()-started:.0f}s",
        flush=True,
    )

    # Window-end distance to the crossing, for the knee re-weighting. Built by
    # an independent replica of the bank enumeration; proven aligned by exact
    # equality of the cens target it derives on the way.
    end_to_crossing, drop_check = window_end_sidecar(frame, cache, FIT_HORIZONS)
    _, cens_target = ts.variant_rows(bank, "cens")
    assert end_to_crossing.shape[0] == len(bank["drop_raw"])
    assert np.allclose(drop_check, cens_target, atol=1e-6, equal_nan=True)
    knee_share = float(
        (np.isfinite(end_to_crossing) & (np.abs(end_to_crossing) <= KNEE_DAYS)).mean()
    )
    print(f"  sidecar aligned; knee-window share {knee_share:.4f}", flush=True)

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    # Regression guard: the first N_BASE columns of the extended rows must be
    # the base features exactly -- shipped artifacts over them must land on the
    # known production sums (v7 8.717 / cens 12.37 raw per scenario).
    check = {}
    for name, path in (
        ("v7", REPO_ROOT / "models/v7_wiener.joblib"),
        ("cens", REPO_ROOT / "models/v8_cens.joblib"),
    ):
        if not path.exists():
            continue
        model = joblib.load(path)
        mask = scen["has_row"]
        h_eff = np.clip(np.minimum(42.0, scen["remaining"][mask]), 0.0, None)
        raw_p = model.probabilities(
            scen["features"][mask][:, :N_BASE].astype(np.float32), h_eff
        )
        p = np.zeros(len(scen["due"]))
        p[mask] = np.where(h_eff <= 0.0, 0.0, raw_p)
        check[name] = {
            "sum_p_raw_per_scenario": round(float(p.sum() / scen["n_scenarios"]), 3),
            "realized_per_scenario": round(
                float(scen["due"].sum() / scen["n_scenarios"]), 3
            ),
        }
        print(f"  reconstruction {name}: {check[name]}", flush=True)

    old_fits_valid = check_old_alignment(scen)

    prep = {
        "frame": frame,
        "scen": scen,
        "bank": bank,
        "end_to_crossing": end_to_crossing,
        "climatology": climatology,
        "reconstruction_check": check,
        "old_fits_valid": old_fits_valid,
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
# fit
# --------------------------------------------------------------------------

def arm_design(bank: dict, arm: str) -> np.ndarray:
    """One arm's feature columns plus the trailing horizon column."""
    design = bank["design"]
    return np.hstack([design[:, ARM_COLUMNS[arm]], design[:, -1:]]).astype(np.float32)


def predict_scenario_arm(model: WienerModel, scen: dict, arm: str) -> np.ndarray:
    p = np.zeros(len(scen["due"]), dtype=float)
    mask = scen["has_row"]
    h_eff = np.clip(np.minimum(42.0, scen["remaining"][mask]), 0.0, None)
    raw = model.probabilities(
        scen["features"][mask][:, ARM_COLUMNS[arm]].astype(np.float32), h_eff
    )
    p[mask] = np.where(h_eff <= 0.0, 0.0, raw)
    return p


def phase_fit(args, work: Path, prep: dict) -> None:
    folds: dict[str, tuple[str, ...]] = {}
    if args.scope in ("hard", "all"):
        folds.update(ts.make_hard_folds(prep))
    if args.scope in ("loo", "all"):
        folds.update(ts.make_loo_folds(prep))
    if args.only:
        folds = {k: v for k, v in folds.items() if k in set(args.only)}

    bank, scen = prep["bank"], prep["scen"]
    rows, target = ts.variant_rows(bank, "cens")

    fits_dir = work / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        (fold, arm)
        for arm in args.arms
        for fold in folds
        if not (fits_dir / f"{fold}__{arm}.joblib").exists() or args.force
    ]
    print(f"{len(todo)} fits to run (arms: {list(args.arms)})", flush=True)

    designs = {arm: arm_design(bank, arm) for arm in set(a for _, a in todo)}
    all_weights = {
        arm: knee_weights(prep["end_to_crossing"], w)
        for arm, w in ARM_KNEE_WEIGHT.items()
    }
    for count, (fold, arm) in enumerate(todo):
        heldout = folds[fold]
        train = rows & ~np.isin(bank["building"], list(heldout))
        started = time.time()
        weights = all_weights.get(arm)
        model = fit_wiener_weighted(
            designs[arm][train],
            target[train],
            prep["climatology"],
            sample_weight=None if weights is None else weights[train],
            params={"max_iter": args.max_iter},
        )
        model.feature_names = tuple(ARM_FEATURES[arm])
        p_raw = predict_scenario_arm(model, scen, arm)
        seconds = time.time() - started
        joblib.dump(
            {
                "fold": fold,
                "variant": arm,
                "heldout": list(heldout),
                "p_raw": p_raw.astype(np.float32),
                "n_train_windows": int(train.sum()),
                "knee_weight": ARM_KNEE_WEIGHT.get(arm, 0.0),
                "seconds": round(seconds, 1),
                "max_iter": args.max_iter,
            },
            fits_dir / f"{fold}__{arm}.joblib",
        )
        print(
            f"  [{count+1}/{len(todo)}] {fold} {arm}: {int(train.sum())} windows, "
            f"{seconds:.0f}s",
            flush=True,
        )


def arm_permutation_importance(
    prep: dict, arm: str, sample: int = 30000, seed: int = 0
) -> list[dict]:
    """Drift permutation importance for one arm, fitted on every window."""
    bank = prep["bank"]
    rows, target = ts.variant_rows(bank, "cens")
    design = arm_design(bank, arm)[rows]
    truth = target[rows]
    w = ARM_KNEE_WEIGHT.get(arm)
    model = fit_wiener_weighted(
        design,
        truth,
        prep["climatology"],
        sample_weight=None if w is None else knee_weights(prep["end_to_crossing"], w)[rows],
        params={"max_iter": 150},
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(truth), size=min(sample, len(truth)), replace=False)
    block = design[chosen].astype(np.float64)
    sample_truth = truth[chosen]
    base = float(np.mean(np.abs(model.drift.predict(block) - sample_truth)))
    names = list(ARM_FEATURES[arm]) + ["horizon"]
    out = []
    for column in range(block.shape[1]):
        shuffled = block.copy()
        shuffled[:, column] = shuffled[rng.permutation(len(chosen)), column]
        mae = float(np.mean(np.abs(model.drift.predict(shuffled) - sample_truth)))
        out.append({"feature": names[column], "delta_mae": round(mae - base, 6)})
    out.sort(key=lambda r: -r["delta_mae"])
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def load_fits(work: Path, prep: dict) -> dict[tuple[str, str], dict]:
    fits: dict[tuple[str, str], dict] = {}
    for path in sorted((work / "fits").glob("*.joblib")):
        record = joblib.load(path)
        fits[(record["fold"], record["variant"])] = record
    if prep.get("old_fits_valid") and OLD_FITS_DIR.exists():
        for path in sorted(OLD_FITS_DIR.glob("*.joblib")):
            record = joblib.load(path)
            fits[(record["fold"], record["variant"])] = record
    return fits


def phase_report(args, work: Path, prep: dict) -> None:
    scen = prep["scen"]
    fits = load_fits(work, prep)
    hard_folds = ts.make_hard_folds(prep)
    loo_folds = ts.make_loo_folds(prep)
    arms = tuple(arm for arm in ARM_FEATURES if any(k[1] == arm for k in fits))
    variants = arms + (INCUMBENTS if prep.get("old_fits_valid") else ())

    results: dict = {
        "config": {
            "stride": prep["stride"],
            "max_iter": args.max_iter,
            "n_windows_cens": int(len(prep["bank"]["drop_raw"])),
            "arm_features": {arm: list(names) for arm, names in ARM_FEATURES.items()},
            "arm_n_features": {arm: len(names) for arm, names in ARM_FEATURES.items()},
            "scenario_rows": int(len(scen["due"])),
            "scenario_due_total": int(scen["due"].sum()),
        },
        "reconstruction_check": prep["reconstruction_check"],
        "old_fits_valid": bool(prep.get("old_fits_valid")),
        "hard_folds": {},
        "loo_pooled": {},
    }

    for fold, heldout in hard_folds.items():
        entry: dict = {"heldout": list(heldout)}
        entry["heldout_devices"] = int(
            sum(prep["building_sizes"].get(b, 0) for b in heldout)
        )
        entry["heldout_eol_events"] = int(
            sum(prep["building_eol"].get(b, 0) for b in heldout)
        )
        for variant in variants:
            record = fits.get((fold, variant))
            if record is None:
                continue
            entry[variant] = ts.fold_metrics(
                scen, record["p_raw"].astype(float), list(heldout)
            )
        results["hard_folds"][fold] = entry

    for variant in variants:
        pooled = ts.pooled_loo(scen, fits, variant, loo_folds)
        if pooled is not None:
            results["loo_pooled"][variant] = pooled

    # ---- success gate ------------------------------------------------------
    def hard_mean(variant: str, key: str) -> float | None:
        values = [
            entry[variant][key]
            for entry in results["hard_folds"].values()
            if variant in entry and key in entry[variant]
        ]
        return round(float(np.mean(values)), 4) if values else None

    gate: dict = {
        "hard_prauc_level_mean": {v: hard_mean(v, "pr_auc_level") for v in variants},
        "hard_prauc_rank_mean": {v: hard_mean(v, "pr_auc_rank") for v in variants},
        "hard_inflation_raw_mean": {v: hard_mean(v, "inflation_raw") for v in variants},
        "hard_inflation_cal_worst": {
            v: max(
                (
                    entry[v]["inflation_cal"]
                    for entry in results["hard_folds"].values()
                    if v in entry
                ),
                default=None,
            )
            for v in variants
        },
        "loo_inflation_raw": {
            v: results["loo_pooled"].get(v, {}).get("inflation_raw") for v in variants
        },
        "loo_inflation_cal": {
            v: results["loo_pooled"].get(v, {}).get("inflation_cal") for v in variants
        },
        "loo_prauc_level": {
            v: results["loo_pooled"].get(v, {}).get("pr_auc_level") for v in variants
        },
    }
    # Gate ladder: every v12b-family arm is judged on the same two criteria.
    # Raw inflation is target-bound -- with censor-aware targets even the
    # shipped cens sits at x1.297 raw / x1.06 calibrated -- so both readings
    # are recorded: strict = raw <= gate, production-procedure = cal <= gate.
    ladder: dict[str, dict] = {}
    candidates = [a for a in arms if a.startswith("v12b")]
    for arm in candidates:
        prauc = gate["hard_prauc_level_mean"].get(arm)
        infl_raw = gate["loo_inflation_raw"].get(arm)
        infl_cal = gate["loo_inflation_cal"].get(arm)
        ladder[arm] = {
            "criterion_prauc": None if prauc is None else bool(prauc > CENS_HARD_PRAUC_MEAN),
            "criterion_inflation_raw": None
            if infl_raw is None
            else bool(infl_raw <= INFLATION_GATE),
            "criterion_inflation_cal": None
            if infl_cal is None
            else bool(infl_cal <= INFLATION_GATE),
        }
        ladder[arm]["go_strict"] = bool(ladder[arm]["criterion_prauc"]) and bool(
            ladder[arm]["criterion_inflation_raw"]
        )
        ladder[arm]["go_production_procedure"] = bool(
            ladder[arm]["criterion_prauc"]
        ) and bool(ladder[arm]["criterion_inflation_cal"])
    gate["ladder"] = ladder

    passing = [a for a in candidates if ladder[a]["go_production_procedure"]]
    best_arm = (
        max(passing, key=lambda a: gate["hard_prauc_level_mean"][a])
        if passing
        else (
            max(
                candidates,
                key=lambda a: (gate["hard_prauc_level_mean"].get(a) or 0.0),
            )
            if candidates
            else None
        )
    )
    gate["gate_arm"] = best_arm
    if best_arm is not None:
        gate.update({k: ladder[best_arm][k] for k in ladder[best_arm]})
    results["gate"] = gate

    if best_arm is not None:
        print(f"permutation importance ({best_arm}, drift, in-sample fit)...", flush=True)
        importance = arm_permutation_importance(prep, best_arm)
        results["gate_arm_drift_importance_top20"] = importance[:20]
        results["gate_arm_drift_importance_raw_channel"] = [
            row for row in importance if row["feature"] in _RAW_ALL
        ]
        results["gate_arm_drift_importance_new_features"] = [
            row
            for row in importance
            if row["feature"] not in FEATURE_NAMES and row["feature"] != "horizon"
        ]

    out_json = REPO_ROOT / "outputs/v12_transfer.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}", flush=True)

    out_md = REPO_ROOT / "outputs/v12_transfer.md"
    out_md.write_text(render_markdown(results, variants))
    print(f"wrote {out_md}", flush=True)
    print(json.dumps(gate, indent=2), flush=True)


def render_markdown(results: dict, variants: tuple[str, ...]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# V12 variants vs incumbents on the transfer harness")
    add("")
    gate = results["gate"]
    arm = gate.get("gate_arm")
    add(
        f"Best arm {arm}: hard PR-AUC mean {gate['hard_prauc_level_mean'].get(arm)} "
        f"(needs > {CENS_HARD_PRAUC_MEAN}); pooled-LOO inflation raw "
        f"{gate['loo_inflation_raw'].get(arm)} / cal {gate['loo_inflation_cal'].get(arm)} "
        f"(gate <= {INFLATION_GATE}) -> strict "
        f"**{'GO' if gate.get('go_strict') else 'NO-GO'}**, production-procedure "
        f"**{'GO' if gate.get('go_production_procedure') else 'NO-GO'}**"
    )
    add("")
    if gate.get("ladder"):
        add("| arm | PR-AUC > 0.428 | LOO raw <= 1.10 | LOO cal <= 1.10 | strict | prod-proc |")
        add("|" + "---|" * 6)
        for name, entry in gate["ladder"].items():
            add(
                f"| {name} | {entry['criterion_prauc']} | {entry['criterion_inflation_raw']} "
                f"| {entry['criterion_inflation_cal']} | {entry['go_strict']} "
                f"| {entry['go_production_procedure']} |"
            )
        add("")
    add("## Hard grouped holdouts")
    add("")
    add(
        "| fold | held-out (dev/EOL) | variant | sum p/scen raw | cal | realized "
        "| infl raw | infl cal | PR-AUC lvl | PR-AUC rank | P@5 lvl | P@10 lvl "
        "| top>=0.7 cal (n, realized) |"
    )
    add("|" + "---|" * 13)
    for fold, entry in results["hard_folds"].items():
        for variant in variants:
            metrics = entry.get(variant)
            if not metrics:
                continue
            pk_l = metrics.get("precision_at_k_level", {})
            top = metrics["top_bucket_cal"]
            add(
                f"| {fold} | {entry['heldout_devices']}/{entry['heldout_eol_events']} "
                f"| {variant} | {metrics['sum_p_raw_per_scenario']} "
                f"| {metrics['sum_p_cal_per_scenario']} | {metrics['realized_per_scenario']} "
                f"| {metrics['inflation_raw']} | {metrics['inflation_cal']} "
                f"| {metrics.get('pr_auc_level', '-')} | {metrics.get('pr_auc_rank', '-')} "
                f"| {pk_l.get(5, '-')} | {pk_l.get(10, '-')} "
                f"| {top['n']}, {top['realized_mean']} |"
            )
    add("")
    add("## Means over the 5 hard folds")
    add("")
    add("| metric | " + " | ".join(variants) + " |")
    add("|" + "---|" * (len(variants) + 1))
    for key in (
        "hard_prauc_level_mean",
        "hard_prauc_rank_mean",
        "hard_inflation_raw_mean",
        "hard_inflation_cal_worst",
        "loo_inflation_raw",
        "loo_inflation_cal",
        "loo_prauc_level",
    ):
        row = gate[key]
        add(f"| {key} | " + " | ".join(str(row.get(v)) for v in variants) + " |")
    add("")
    add("## Pooled leave-one-building-out")
    add("")
    for variant, pooled in results.get("loo_pooled", {}).items():
        add(f"### {variant}")
        add("```json")
        slim = {k: v for k, v in pooled.items() if k != "per_building"}
        add(json.dumps(slim, indent=2))
        add("```")
        add("")
    if results.get("gate_arm_drift_importance_top20"):
        add("## Gate-arm drift permutation importance (top 20)")
        add("")
        add("```json")
        add(json.dumps(results["gate_arm_drift_importance_top20"], indent=2))
        add("```")
        add("")
        add("### Raw-daily channel")
        add("")
        add("```json")
        add(json.dumps(results.get("gate_arm_drift_importance_raw_channel", []), indent=2))
        add("```")
        add("")
        add("### All new (non-base) features")
        add("")
        add("```json")
        add(json.dumps(results.get("gate_arm_drift_importance_new_features", []), indent=2))
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
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--arms",
        nargs="*",
        choices=sorted(ARM_FEATURES),
        default=list(DEFAULT_ARMS),
        help="which V12 arms to fit; report always covers every arm with fits",
    )
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
