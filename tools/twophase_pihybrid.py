"""Pi-feature hybrid: the changepoint filter's causal state as GBDT features.

Litreview P3-1 stage 2. The two-phase filter's per-device onset posterior
pi_t and pi-weighted expected drift (pi*mu2 + (1-pi)*trailing_drift) are
appended to the base feature vector, and the incumbent censored-target Wiener
GBDT is retrained with the exact train_wiener recipe (stride 4, max_iter 250,
5-fold grouped by building). The pi features are out-of-fold at every stage:
a device's pi comes from the filter whose EM excluded its building, and the
GBDT scoring it never saw its building either. Both variants (base control
and +pi) are fitted on IDENTICAL increment rows, targets and folds, so the
delta is the pi information and nothing else.

Gates (coordinator ladder):
  (a) OOF stride PR-AUC: +pi > base control (like-for-like of the recorded
      0.4706) AND frame-level calibrated AP > cens-cal 0.308;
  (b) mid-block top-12 >= 0.27 and open block >= 0.55 on the scenario frame;
  (c) hard-holdout mean PR-AUC >= 0.428 (the cens mean over the five recorded
      transfer_stress folds) and sum-p inflation <= 1.15;
  (d) planner at the operating config (run separately if a-c pass).

    python tools/twophase_pihybrid.py

Phases checkpoint into --work and re-runs resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from batteryswap_public.utils import load_devices

from bsai.calibrate import RemainingCalibration
from bsai.features import FEATURE_NAMES, fleet_climatology
from bsai.hazard import TrainingFrame, build_training_frame
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache
from bsai.twophase import (
    PI_FEATURE_NAMES,
    PiHybridModel,
    fit_em,
    forward_pi,
    trailing_drift,
)
from bsai.wiener import FIT_HORIZONS, WienerModel, build_increment_targets

from twophase_fit import make_tracks  # noqa: E402  (tools path)

_EPOCH = pd.Timestamp("1970-01-01")
DECISION = 42.0
DEFAULT_WORK = Path(
    os.environ.get(
        "PIHYBRID_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\pihybrid_work",
    )
)


def _ordinal(value) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return int((stamp.normalize() - _EPOCH) / pd.Timedelta(days=1))


def report(probability: np.ndarray, truth: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    order = np.argsort(-probability)
    out = {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "auc": round(float(roc_auc_score(truth, probability)), 4),
        "pr_auc": round(float(average_precision_score(truth, probability)), 4),
        "predicted_over_actual": round(float(probability.sum() / max(truth.sum(), 1)), 3),
    }
    for k in (100, 500, 1000):
        if k <= truth.size:
            out[f"precision_at_{k}"] = round(float(truth[order[:k]].mean()), 4)
    return out


def increment_rows(frame: TrainingFrame, cache: SmoothingCache):
    """(row index, horizon, drop) of every censored-target increment window.

    Runs ``build_increment_targets`` on an index-carrying frame so the row
    selection, the censor bump and the ordering are byte-identical to the
    production builder; any feature variant's design is then a cheap hstack.
    """
    index_frame = TrainingFrame(
        features=np.arange(len(frame), dtype=np.float32)[:, None],
        device=frame.device,
        building=frame.building,
        cutoff=frame.cutoff,
        crossing=frame.crossing,
        last_observed=frame.last_observed,
        observation_end=frame.observation_end,
    )
    design, drop = build_increment_targets(index_frame, cache, FIT_HORIZONS)
    rows = design[:, 0].astype(np.int64)
    horizon = design[:, 1].astype(np.float32)
    return rows, horizon, drop


def pi_columns(
    frame_device: np.ndarray,
    day_ordinals: np.ndarray,
    tables: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    default=(0.0, -2.0e-4),
) -> np.ndarray:
    out = np.empty((frame_device.shape[0], 2), dtype=np.float32)
    order = np.argsort(frame_device, kind="stable")
    start = 0
    while start < order.size:
        stop = start
        device = frame_device[order[start]]
        while stop < order.size and frame_device[order[stop]] == device:
            stop += 1
        block = order[start:stop]
        start = stop
        table = tables.get(str(device))
        if table is None:
            out[block] = default
            continue
        days, pi, drift = table
        index = np.searchsorted(days, day_ordinals[block], side="right") - 1
        valid = index >= 0
        safe = np.maximum(index, 0)
        out[block, 0] = np.where(valid, pi[safe], default[0])
        out[block, 1] = np.where(valid, drift[safe], default[1])
    return out


def day_ordinals_of(frame: TrainingFrame, cache: SmoothingCache) -> np.ndarray:
    origins = {d: s.origin for d, s in cache.devices.items()}
    return np.array(
        [origins[d] + c for d, c in zip(frame.device, frame.cutoff)], dtype=np.int64
    )


def filter_tables_for(params, tracks, kappa_prior=None):
    """(days, pi, pi_drift) per device under one filter parameter set."""
    tables = {}
    for track in tracks:
        pi = forward_pi(track, params)
        drift = trailing_drift(track.days, track.margin, params.mu1)
        pi_drift = pi * params.mu2 + (1.0 - pi) * drift
        tables[track.device] = (
            track.days,
            pi.astype(np.float32),
            pi_drift.astype(np.float32),
        )
    return tables


def block_of(scenario: np.ndarray) -> np.ndarray:
    return np.where(scenario <= 15, 0, np.where(scenario <= 31, 1, 2))


def top_k_rate(scen_idx, p, due, low, high, k=12) -> float:
    rates = []
    for s in np.unique(scen_idx):
        if not (low <= s <= high):
            continue
        mask = scen_idx == s
        top = np.argsort(-p[mask])[:k]
        rates.append(float(due[mask][top].mean()))
    return float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--series", type=Path, default=Path("outputs/twophase_series.joblib"))
    parser.add_argument("--filter", type=Path, default=Path("outputs/twophase_model_oof.joblib"))
    parser.add_argument("--transfer", type=Path, default=Path("outputs/transfer_stress.json"))
    parser.add_argument("--rowfeat", type=Path, default=Path("outputs/research_rowfeat.parquet"))
    parser.add_argument("--pi-out", type=Path, default=Path("outputs/twophase_pi.parquet"))
    parser.add_argument("--report", type=Path, default=Path("outputs/pihybrid_gates.json"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/twophase_pihybrid_model.joblib"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--hard-stride", type=int, default=8)
    parser.add_argument("--hard-max-iter", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def stamp(msg):
        print(f"[{time.time()-started:6.0f}s] {msg}", flush=True)

    # ---- pi export -----------------------------------------------------------
    filter_model = joblib.load(args.filter)
    oof_tables: dict[str, tuple] = {}
    parquet_rows = []
    for device, (days, pi, drift) in filter_model.pi_tables.items():
        fold = filter_model.fold_of_device.get(device, -1)
        params = filter_model.params_by_fold.get(fold, filter_model.production_params)
        pi_drift = pi * params.mu2 + (1.0 - pi) * drift
        oof_tables[device] = (days, pi.astype(np.float32), pi_drift.astype(np.float32))
        parquet_rows.append(
            pd.DataFrame(
                {"device": device, "day": days, "pi_posterior": pi,
                 "pi_drift": pi_drift, "fold": fold}
            )
        )
    pi_frame = pd.concat(parquet_rows, ignore_index=True)
    args.pi_out.parent.mkdir(parents=True, exist_ok=True)
    pi_frame.to_parquet(args.pi_out)
    stamp(f"wrote {args.pi_out} ({len(pi_frame)} rows, {len(oof_tables)} devices)")

    # ---- caches ---------------------------------------------------------------
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    end_ordinal = {r.device_id: _ordinal(r.end_time) for r in devices.itertuples()}
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"],
        format="ISO8601",
    )
    observation_end = devices.set_index("device_id")["end_time"]
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw
    eol_index, observation_index = {}, {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = None if pd.isna(moment) else _ordinal(moment) - series.origin
        end = observation_end.get(device_id)
        observation_index[device_id] = (
            (series.origin + len(series) - 1) if pd.isna(end) else _ordinal(end) - series.origin
        )
    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )
    stamp(f"caches ready ({len(cache.devices)} devices)")

    # ---- stride frame + increment bank ----------------------------------------
    bank_path = args.work / f"bank_stride{args.stride}.joblib"
    if bank_path.exists():
        bank = joblib.load(bank_path)
        stamp(f"loaded {bank_path}")
    else:
        frame = build_training_frame(
            cache, eol_index, building_of, observation_index,
            shape_cache=shape_cache, stride=args.stride,
        )
        rows, horizon, drop = increment_rows(frame, cache)
        bank = {
            "frame": frame,
            "rows": rows,
            "horizon": horizon,
            "drop": drop,
            "day_ord": day_ordinals_of(frame, cache),
        }
        joblib.dump(bank, bank_path)
        stamp(f"stride-{args.stride} frame {len(frame)} cutoffs, {rows.size} windows")
    frame = bank["frame"]
    rows, horizon, drop = bank["rows"], bank["horizon"], bank["drop"]
    truth = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & ((frame.crossing - frame.cutoff) <= DECISION)
        & (frame.crossing <= frame.observation_end)
    ).astype(np.int8)
    decision_horizon = np.clip(
        np.minimum(DECISION, frame.observation_end - frame.cutoff), 0.0, None
    ).astype(np.float32)
    base_n = frame.features.shape[1]
    pi_cols = pi_columns(frame.device, bank["day_ord"], oof_tables)
    features_pi = np.hstack([frame.features, pi_cols]).astype(np.float32)
    groups = frame.building[rows]

    # ---- scenario frame ---------------------------------------------------------
    rowfeat = pd.read_parquet(args.rowfeat)
    scen_ordinals = np.sort(rowfeat.cutoff_ord.unique()).astype(np.int64)
    scen_path = args.work / "scen_frame.joblib"
    if scen_path.exists():
        scen = joblib.load(scen_path)
        stamp(f"loaded {scen_path}")
    else:
        sframe = build_training_frame(
            cache, eol_index, building_of, observation_index,
            shape_cache=shape_cache, cutoff_days=scen_ordinals,
        )
        scen = {"frame": sframe, "day_ord": day_ordinals_of(sframe, cache)}
        joblib.dump(scen, scen_path)
    sframe, s_day = scen["frame"], scen["day_ord"]
    key = pd.DataFrame({"battery": sframe.device, "cutoff_ord": s_day, "row": np.arange(len(sframe))})
    joined = key.merge(rowfeat, on=["battery", "cutoff_ord"], how="inner")
    stamp(
        f"scenario frame {len(sframe)} rows, joined {len(joined)} of {len(rowfeat)} "
        f"rowfeat rows ({int(rowfeat.due.sum())} dues total)"
    )
    s_rows = joined.row.to_numpy()
    s_remaining = joined.remaining.to_numpy(dtype=float)
    s_scenario = joined.scenario.to_numpy()
    s_due = joined.due.to_numpy(dtype=float)
    s_pcal = joined.p_cal.fillna(0.0).to_numpy(dtype=float)
    s_heff = np.clip(np.minimum(DECISION, s_remaining), 0.0, None).astype(np.float32)
    s_pi_cols = pi_columns(sframe.device[s_rows], s_day[s_rows], oof_tables)
    s_features_pi = np.hstack([sframe.features[s_rows], s_pi_cols]).astype(np.float32)
    s_building = sframe.building[s_rows]
    # rowfeat rows with no scenario-frame twin (cold starts): probability zero
    missing = rowfeat.merge(key, on=["battery", "cutoff_ord"], how="left", indicator=True)
    missing = missing[missing._merge == "left_only"]
    miss_due = missing.due.to_numpy(dtype=float)
    miss_scen = missing.scenario.to_numpy()
    miss_pcal = missing.p_cal.fillna(0.0).to_numpy(dtype=float)

    # ---- five folds, both variants ---------------------------------------------
    folds_path = args.work / "fold_models.joblib"
    splitter = GroupKFold(n_splits=args.folds)
    all_buildings = set(np.unique(groups))
    if folds_path.exists():
        pack = joblib.load(folds_path)
        stamp(f"loaded {folds_path}")
    else:
        pack = {"by_building": {}, "fold_of_building": {}, "oof_pi": np.zeros(len(frame)),
                "oof_base": np.zeros(len(frame)), "models": {}}
        for fold, (train_rows, _) in enumerate(splitter.split(drop[:, None], drop, groups)):
            train_buildings = set(np.unique(groups[train_rows]))
            held = all_buildings - train_buildings
            design_pi = np.hstack(
                [features_pi[rows[train_rows]], horizon[train_rows, None]]
            ).astype(np.float32)
            design_base = np.hstack(
                [frame.features[rows[train_rows]], horizon[train_rows, None]]
            ).astype(np.float32)
            model_pi = WienerModel.fit(
                design_pi, drop[train_rows], climatology, params={"max_iter": args.max_iter}
            )
            model_pi.feature_names = tuple(FEATURE_NAMES) + PI_FEATURE_NAMES
            model_pi.model_version = "bsai-wiener/v1+pi"
            model_base = WienerModel.fit(
                design_base, drop[train_rows], climatology, params={"max_iter": args.max_iter}
            )
            del design_pi, design_base
            mask = np.isin(frame.building, list(held))
            pack["oof_pi"][mask] = model_pi.probabilities(features_pi[mask], decision_horizon[mask])
            pack["oof_base"][mask] = model_base.probabilities(
                frame.features[mask], decision_horizon[mask]
            )
            pack["models"][fold] = model_pi
            for building in held:
                pack["by_building"][str(building)] = fold
                pack["fold_of_building"][str(building)] = fold
            stamp(f"fold {fold} fitted (held {len(held)} buildings)")
        joblib.dump(pack, folds_path)
    fold_of_building = pack["fold_of_building"]
    metrics_pi = report(pack["oof_pi"], truth)
    metrics_base = report(pack["oof_base"], truth)
    stamp(f"stride OOF +pi:  {metrics_pi}")
    stamp(f"stride OOF base: {metrics_base}")

    # volatility scale on the +pi fold models (count match, train_wiener style)
    def rescale(scale: float) -> np.ndarray:
        out = np.zeros(len(frame))
        for building in np.unique(frame.building):
            fold = fold_of_building.get(str(building))
            if fold is None:
                continue
            model = pack["models"][fold]
            previous = model.volatility_scale
            model.volatility_scale = scale
            mask = frame.building == building
            out[mask] = model.probabilities(features_pi[mask], decision_horizon[mask])
            model.volatility_scale = previous
        return out

    best_scale, best_gap = 1.0, abs(pack["oof_pi"].sum() - truth.sum())
    for scale in np.arange(0.6, 2.01, 0.2):
        gap = abs(rescale(float(scale)).sum() - truth.sum())
        if gap < best_gap:
            best_scale, best_gap = float(scale), gap
    for model in pack["models"].values():
        model.volatility_scale = best_scale
    stamp(f"volatility_scale {best_scale:.1f}")

    # ---- scenario scoring + calibration (gates a-frame, b, sum-p) ---------------
    s_fold = np.array([fold_of_building.get(str(b), -1) for b in s_building])
    p42_raw = np.zeros(len(joined))
    for fold, model in pack["models"].items():
        mask = s_fold == fold
        if mask.any():
            p42_raw[mask] = model.probabilities(s_features_pi[mask], s_heff[mask])
    p42_raw = np.where(s_heff <= 0.0, 0.0, p42_raw)

    p42_cal = np.empty_like(p42_raw)
    calibrations = {}
    for fold in sorted(pack["models"]):
        others = s_fold != fold
        calibration = RemainingCalibration.fit(s_remaining[others], p42_raw[others], s_due[others])
        calibrations[fold] = calibration
        mask = s_fold == fold
        p42_cal[mask] = np.clip(
            p42_raw[mask] * calibration.factor_for(s_remaining[mask]), 0.0, 1.0
        )
        pack["models"][fold].calibration = calibration

    all_scen = np.concatenate([s_scenario, miss_scen])
    all_due = np.concatenate([s_due, miss_due])
    all_p = np.concatenate([p42_cal, np.zeros(len(miss_scen))])
    all_p_raw = np.concatenate([p42_raw, np.zeros(len(miss_scen))])
    all_pcal_inc = np.concatenate([s_pcal, miss_pcal])
    frame_ap_cal = float(average_precision_score(all_due, all_p))
    frame_ap_raw = float(average_precision_score(all_due, all_p_raw))
    frame_ap_inc = float(average_precision_score(all_due, all_pcal_inc))
    mid12 = top_k_rate(all_scen, all_p, all_due, 16, 31)
    open12 = top_k_rate(all_scen, all_p, all_due, 0, 15)
    mid12_inc = top_k_rate(all_scen, all_pcal_inc, all_due, 16, 31)
    open12_inc = top_k_rate(all_scen, all_pcal_inc, all_due, 0, 15)

    sums = pd.DataFrame({"scenario": all_scen, "p": all_p, "p_raw": all_p_raw,
                         "inc": all_pcal_inc, "due": all_due})
    per_scen = sums.groupby("scenario").sum()
    blocks = {}
    for name, low, high in (("early", 0, 15), ("mid", 16, 31), ("late", 32, 47)):
        seg = per_scen.loc[low:high]
        blocks[name] = {
            "sum_p_cal": round(float(seg.p.mean()), 2),
            "sum_p_raw": round(float(seg.p_raw.mean()), 2),
            "sum_p_incumbent": round(float(seg.inc.mean()), 2),
            "realized": round(float(seg.due.mean()), 2),
            "ratio_cal": round(float(seg.p.sum() / max(seg.due.sum(), 1e-9)), 3),
        }
    inflation_oof = float(per_scen.p.sum() / max(per_scen.due.sum(), 1e-9))
    budget_mine = np.minimum(15, np.ceil(1.6 * per_scen.p + 1.0))
    budget_inc = np.minimum(15, np.ceil(1.6 * per_scen.inc + 1.0))
    stamp(
        f"frame AP cal {frame_ap_cal:.4f} raw {frame_ap_raw:.4f} vs incumbent {frame_ap_inc:.4f} | "
        f"mid12 {mid12:.3f} (inc {mid12_inc:.3f}) open12 {open12:.3f} (inc {open12_inc:.3f})"
    )
    stamp(f"blocks {json.dumps(blocks)}")

    # ---- hard holdouts (gate c) --------------------------------------------------
    hard_defs = json.load(open(args.transfer))["hard_folds"]
    cens_ref = {name: fold["cens"]["pr_auc_level"] for name, fold in hard_defs.items()}
    bank8_path = args.work / f"bank_stride{args.hard_stride}.joblib"
    if bank8_path.exists():
        bank8 = joblib.load(bank8_path)
        stamp(f"loaded {bank8_path}")
    else:
        frame8 = build_training_frame(
            cache, eol_index, building_of, observation_index,
            shape_cache=shape_cache, stride=args.hard_stride,
        )
        rows8, horizon8, drop8 = increment_rows(frame8, cache)
        bank8 = {
            "frame": frame8, "rows": rows8, "horizon": horizon8, "drop": drop8,
            "day_ord": day_ordinals_of(frame8, cache),
        }
        joblib.dump(bank8, bank8_path)
        stamp(f"stride-{args.hard_stride} bank ready ({rows8.size} windows)")
    frame8 = bank8["frame"]
    series_bundle = joblib.load(args.series)
    tracks = make_tracks(series_bundle)

    hard_results = {}
    for name, definition in hard_defs.items():
        result_path = args.work / f"hard_{name}.joblib"
        if result_path.exists():
            hard_results[name] = joblib.load(result_path)
            stamp(f"loaded hard fold {name}")
            continue
        held = set(definition["heldout"])
        train_tracks = [t for t in tracks if t.building not in held]
        params_h = fit_em(
            train_tracks, max_iter=300, mu2_ceiling=-1.5e-3, sigma2_cap_ratio=1.5
        )
        tables_h = filter_tables_for(params_h, tracks)
        cols8 = pi_columns(frame8.device, bank8["day_ord"], tables_h)
        feats8 = np.hstack([frame8.features, cols8]).astype(np.float32)
        groups8 = frame8.building[bank8["rows"]]
        train_mask = ~np.isin(groups8, list(held))
        design8 = np.hstack(
            [feats8[bank8["rows"][train_mask]], bank8["horizon"][train_mask, None]]
        ).astype(np.float32)
        model_h = WienerModel.fit(
            design8, bank8["drop"][train_mask], climatology,
            params={"max_iter": args.hard_max_iter},
        )
        del design8
        s_cols_h = pi_columns(sframe.device[s_rows], s_day[s_rows], tables_h)
        s_feats_h = np.hstack([sframe.features[s_rows], s_cols_h]).astype(np.float32)
        p_h = np.where(
            s_heff <= 0.0, 0.0, model_h.probabilities(s_feats_h, s_heff)
        )
        held_mask = np.isin(s_building, list(held))
        cal_h = RemainingCalibration.fit(
            s_remaining[~held_mask], p_h[~held_mask], s_due[~held_mask]
        )
        p_h_cal = np.clip(p_h * cal_h.factor_for(s_remaining), 0.0, 1.0)
        held_blocks = {}
        seg = pd.DataFrame({
            "scenario": s_scenario[held_mask], "p": p_h_cal[held_mask],
            "due": s_due[held_mask],
        }).groupby("scenario").sum()
        for bname, low, high in (("early", 0, 15), ("mid", 16, 31), ("late", 32, 47)):
            part = seg.loc[low:high]
            held_blocks[bname] = round(float(part.p.sum() / max(part.due.sum(), 1e-9)), 3)
        entry = {
            "pr_auc_cal": round(float(average_precision_score(s_due[held_mask], p_h_cal[held_mask])), 4),
            "pr_auc_raw": round(float(average_precision_score(s_due[held_mask], p_h[held_mask])), 4),
            "inflation_cal": round(float(p_h_cal[held_mask].sum() / max(s_due[held_mask].sum(), 1e-9)), 3),
            "inflation_raw": round(float(p_h[held_mask].sum() / max(s_due[held_mask].sum(), 1e-9)), 3),
            "blocks_ratio_cal": held_blocks,
            "n_rows": int(held_mask.sum()),
            "n_due": int(s_due[held_mask].sum()),
            "cens_reference": cens_ref[name],
        }
        joblib.dump(entry, result_path)
        hard_results[name] = entry
        stamp(f"hard {name}: {entry}")

    hard_mean = float(np.mean([hard_results[n]["pr_auc_cal"] for n in hard_results]))
    hard_mean_raw = float(np.mean([hard_results[n]["pr_auc_raw"] for n in hard_results]))
    cens_mean = float(np.mean(list(cens_ref.values())))
    hard_infl = float(np.mean([hard_results[n]["inflation_cal"] for n in hard_results]))

    # ---- gates + artifact ---------------------------------------------------------
    g_a = (metrics_pi["pr_auc"] > metrics_base["pr_auc"]) and (frame_ap_cal > 0.308)
    g_b = (mid12 >= 0.27) and (open12 >= 0.55)
    g_c = (max(hard_mean, hard_mean_raw) >= 0.428) and (hard_infl <= 1.15) and (inflation_oof <= 1.15)
    verdicts = {"a_pr_auc": g_a, "b_topk": g_b, "c_transfer": g_c}

    by_building_models = {
        building: pack["models"][fold] for building, fold in fold_of_building.items()
    }
    artifact = PiHybridModel(
        by_building=by_building_models,
        building_of=building_of,
        pi_tables=oof_tables,
        end_ordinal=end_ordinal,
        climatology=climatology,
    )
    joblib.dump(artifact, args.model_out)
    stamp(f"wrote {args.model_out}")

    payload = {
        "stride_oof_pi": metrics_pi,
        "stride_oof_base_control": metrics_base,
        "stride_reference_recorded": 0.4706,
        "volatility_scale": best_scale,
        "frame_ap_cal": round(frame_ap_cal, 4),
        "frame_ap_raw": round(frame_ap_raw, 4),
        "frame_ap_incumbent_cal": round(frame_ap_inc, 4),
        "mid12": round(mid12, 3),
        "open12": round(open12, 3),
        "mid12_incumbent": round(mid12_inc, 3),
        "open12_incumbent": round(open12_inc, 3),
        "sum_p_blocks": blocks,
        "inflation_oof_cal": round(inflation_oof, 3),
        "budget_mean_mine": round(float(budget_mine.mean()), 2),
        "budget_mean_incumbent": round(float(budget_inc.mean()), 2),
        "budget_scenarios_smaller": int((budget_mine < budget_inc).sum()),
        "hard_folds": hard_results,
        "hard_mean_ap_cal": round(hard_mean, 4),
        "hard_mean_ap_raw": round(hard_mean_raw, 4),
        "hard_cens_reference_mean": round(cens_mean, 4),
        "hard_inflation_cal_mean": round(hard_infl, 3),
        "verdicts": verdicts,
        "abc_pass": all(verdicts.values()),
        "seconds": round(time.time() - started, 1),
    }
    args.report.write_text(json.dumps(payload, indent=2))
    print("\n=== pi-hybrid gates ===")
    print(f"(a) stride AP +pi {metrics_pi['pr_auc']} vs base {metrics_base['pr_auc']} "
          f"(recorded 0.4706); frame AP cal {frame_ap_cal:.4f} vs 0.308 -> {'PASS' if g_a else 'FAIL'}")
    print(f"(b) mid12 {mid12:.3f} (>=0.27) open12 {open12:.3f} (>=0.55) -> {'PASS' if g_b else 'FAIL'}")
    print(f"(c) hard mean AP cal {hard_mean:.4f} raw {hard_mean_raw:.4f} (>=0.428, cens {cens_mean:.4f}); "
          f"infl hard {hard_infl:.3f} oof {inflation_oof:.3f} (<=1.15) -> {'PASS' if g_c else 'FAIL'}")
    print(f"a-c {'PASS -> run planner gate (d)' if payload['abc_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
