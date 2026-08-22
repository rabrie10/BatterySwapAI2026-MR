"""ROADBLOCK L2: re-derive the stack's self-imposed constraints from data.

(ii)  stride-4 cutoff sampling: knee-week (crossing within 21 d) share of
      training cutoffs vs the same share at scenario cutoffs.
(iii) FIT_HORIZONS support: windows and crossed windows per horizon; verify
      HORIZON_GRID[11] == 42 (exact, not interpolated).
(iv)  min_samples_leaf=60 / l2=1.0 pooling: drift/scatter prediction
      uniqueness for knee-cell rows vs plateau rows (same margin band), OOF
      fold models.
(v)   RemainingCalibration clamp [0.35, 2.75] binding + the min_events=25
      bucket skip; emergency_rank approximation error at the op point.
(vi)  candidate filter (margin 24 h, max 150) binding, from the audit ledger.
(vii) the 10-30 degC smoothing filter on FEATURES: scenario rows whose
      any-temperature raw daily median is fresher than the filtered channel.

Inputs: dataset/train (parquet, csvs), outputs/research_rowfeat.parquet,
outputs/audit_ledger.csv, outputs/v8_folds_cens.joblib, models/v8_cens.joblib.
Output: outputs/roadblock_l2.json. Analytic only; no planner runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsai.features import FEATURE_NAMES, fleet_climatology
from bsai.hazard import HORIZON_GRID, build_training_frame
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache
from bsai.wiener import FIT_HORIZONS

_EPOCH = pd.Timestamp("1970-01-01")
VOLT = FEATURE_NAMES.index("voltage")

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _ordinal(value) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return int((ts.normalize() - _EPOCH) / pd.Timedelta(days=1))


def main() -> None:
    out: dict = {}
    dataset = ROOT / "dataset" / "train"

    # ------------------------------------------------------------------ load
    log("loading hourly parquet ...")
    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    log(f"  {len(raw)} rows")
    rowfeat = pd.read_parquet(ROOT / "outputs" / "research_rowfeat.parquet")

    # ---------------------------------------------------------- (vii) filter
    log("(vii) any-temperature daily medians ...")
    work = pd.DataFrame(
        {
            "device_id": raw["device_id"].astype(str),
            "day": (
                (pd.to_datetime(raw["end_time"]).dt.normalize() - _EPOCH)
                // pd.Timedelta(days=1)
            ).astype(np.int64),
            "voltage": raw["voltage"].astype(float),
            "temperature": raw["temperature"].astype(float),
        }
    )
    grouped = work.groupby(["device_id", "day"], sort=True, observed=True)["voltage"]
    counts = grouped.size()
    any_days_1 = counts[counts >= 1].reset_index()[["device_id", "day"]]
    any_days_5 = counts[counts >= 5].reset_index()[["device_id", "day"]]

    def day_lookup(table: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            device: block["day"].to_numpy()
            for device, block in table.groupby("device_id", sort=False)
        }

    lookup_1 = day_lookup(any_days_1)
    lookup_5 = day_lookup(any_days_5)

    def staleness_of(lookup: dict[str, np.ndarray]) -> np.ndarray:
        values = np.full(len(rowfeat), np.inf)
        batteries = rowfeat["battery"].to_numpy()
        cutoffs = rowfeat["cutoff_ord"].to_numpy()
        for i in range(len(rowfeat)):
            days = lookup.get(batteries[i])
            if days is None:
                continue
            pos = np.searchsorted(days, cutoffs[i] - 1, side="right")
            if pos > 0:
                values[i] = (cutoffs[i] - 1) - days[pos - 1]
        return values

    stale_any1 = staleness_of(lookup_1)
    stale_any5 = staleness_of(lookup_5)
    filt = rowfeat["staleness"].to_numpy(dtype=float)
    due = rowfeat["due"].to_numpy(dtype=bool)
    p_cal = rowfeat["p_cal"].to_numpy(dtype=float)

    dark = filt > 30
    fresh_any1 = stale_any1 <= 7
    fresh_any5 = stale_any5 <= 7
    invis_due = due & (p_cal < 0.02)
    out["vii_smoothing_filter"] = {
        "rows_total": int(len(rowfeat)),
        "rows_filtered_dark_gt30": int(dark.sum()),
        "dark_with_raw_any_temp_fresh7_ge1": int((dark & fresh_any1).sum()),
        "dark_with_raw_any_temp_fresh7_ge5": int((dark & fresh_any5).sum()),
        "dark_due": int((dark & due).sum()),
        "dark_due_raw_fresh7_ge1": int((dark & due & fresh_any1).sum()),
        "dark_due_raw_fresh7_ge5": int((dark & due & fresh_any5).sum()),
        "invisible_due_total": int(invis_due.sum()),
        "invisible_due_dark": int((invis_due & dark).sum()),
        "invisible_due_dark_raw_fresh7_ge1": int(
            (invis_due & dark & fresh_any1).sum()
        ),
        "median_staleness_filtered_all": float(np.nanmedian(filt)),
        "median_staleness_any1_all": float(
            np.median(stale_any1[np.isfinite(stale_any1)])
        ),
        "mean_staleness_gap_on_dark_rows_ge1": float(
            np.mean((filt[dark] - stale_any1[dark])[np.isfinite(stale_any1[dark])])
        ),
        "note": "fresh7 = an any-temperature daily median exists within 7 d of "
        "cutoff-1; ge1/ge5 = min readings per day",
    }
    log(f"  dark rows {int(dark.sum())}, of them raw-fresh(>=1r) "
        f"{int((dark & fresh_any1).sum())}")

    # --------------------------------------------------- caches + train frame
    log("building SmoothingCache + ShapeCache ...")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    del raw, work

    devices = pd.read_csv(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]
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

    log("building training frame (stride 4) ...")
    frame = build_training_frame(
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
        stride=4,
    )
    log(f"  {len(frame)} cutoffs")

    # ------------------------------------------------------------ (ii) stride
    has_cross = frame.crossing >= 0
    gap = np.where(has_cross, frame.crossing - frame.cutoff, np.inf)
    knee21_train = (gap > 0) & (gap <= 21)
    knee42_train = (gap > 0) & (gap <= 42)
    d2e = rowfeat["days_to_eol"].to_numpy(dtype=float)
    knee21_scen = np.isfinite(d2e) & (d2e > 0) & (d2e <= 21)
    knee42_scen = np.isfinite(d2e) & (d2e > 0) & (d2e <= 42)
    out["ii_stride_sampling"] = {
        "train_cutoffs": int(len(frame)),
        "train_knee21_rows": int(knee21_train.sum()),
        "train_knee21_share": round(float(knee21_train.mean()), 5),
        "train_knee42_rows": int(knee42_train.sum()),
        "train_knee42_share": round(float(knee42_train.mean()), 5),
        "train_devices_with_crossing_sampled": int(
            len(np.unique(frame.device[knee42_train]))
        ),
        "scenario_rows": int(len(rowfeat)),
        "scenario_knee21_rows": int(knee21_scen.sum()),
        "scenario_knee21_share": round(float(knee21_scen.mean()), 5),
        "scenario_knee42_rows": int(knee42_scen.sum()),
        "scenario_knee42_share": round(float(knee42_scen.mean()), 5),
        "ratio_knee21_train_over_scenario": round(
            float(knee21_train.mean() / max(knee21_scen.mean(), 1e-9)), 3
        ),
        "ratio_knee42_train_over_scenario": round(
            float(knee42_train.mean() / max(knee42_scen.mean(), 1e-9)), 3
        ),
    }
    log(f"  knee21 train share {knee21_train.mean():.5f} vs scenario "
        f"{knee21_scen.mean():.5f}")

    # ------------------------------------------------- (iii) horizon support
    assert HORIZON_GRID[11] == 42, "42 d is not exact at index 11"
    margins = {
        d: s.smooth_voltage - 2.4 for d, s in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")
    per_h: dict[int, dict[str, int]] = {}
    for horizon in FIT_HORIZONS:
        n_windows = 0
        n_crossed = 0
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
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            n_windows += int(finite.sum())
            if crossing >= 0:
                crossed = (
                    (cutoffs[usable][finite] < crossing)
                    & (ends[usable][finite] >= crossing)
                )
                n_crossed += int(crossed.sum())
        per_h[horizon] = {"windows": n_windows, "crossed": n_crossed}
    out["iii_horizon_support"] = {
        "horizon_grid_index_11": HORIZON_GRID[11],
        "windows_by_horizon": per_h,
        "total_windows": int(sum(v["windows"] for v in per_h.values())),
        "min_samples_leaf": 60,
        "crossed_share_h7": round(
            per_h[7]["crossed"] / max(per_h[7]["windows"], 1), 6
        ),
        "crossed_share_h42": round(
            per_h[42]["crossed"] / max(per_h[42]["windows"], 1), 6
        ),
    }
    log(f"  windows/crossed: " + ", ".join(
        f"h{h}={v['windows']}/{v['crossed']}" for h, v in per_h.items()))

    # ------------------------------------------------------- (iv) leaf pooling
    log("(iv) drift prediction uniqueness on knee vs plateau rows ...")
    bundle = joblib.load(ROOT / "outputs" / "v8_folds_cens.joblib")
    by_building = bundle["by_building"]
    margin_at = frame.features[:, VOLT].astype(float) - 2.4
    band = (margin_at >= 0.05) & (margin_at <= 0.20)
    knee = band & (gap > 0) & (gap <= 42)
    plateau = band & (gap > 120)
    horizon_col = np.full((len(frame), 1), 42.0, dtype=np.float32)
    design = np.hstack([frame.features, horizon_col])

    drift_pred = np.full(len(frame), np.nan)
    scatter_pred = np.full(len(frame), np.nan)
    rows_of_interest = np.flatnonzero(knee | plateau)
    for building in np.unique(frame.building[rows_of_interest]):
        model = by_building.get(str(building))
        if model is None:
            continue
        mask = (frame.building == building) & (knee | plateau)
        idx = np.flatnonzero(mask)
        drift_pred[idx] = model.drift.predict(design[idx])
        scatter_pred[idx] = model.scatter.predict(design[idx])

    kd = drift_pred[knee]
    pdrift = drift_pred[plateau]
    kd = kd[np.isfinite(kd)]
    pdrift = pdrift[np.isfinite(pdrift)]
    # exact prediction-value collisions knee vs plateau (same-leaf proxy)
    plateau_set = set(np.round(pdrift, 10))
    collide = np.array([round(v, 10) in plateau_set for v in kd])
    needed = margin_at[knee]
    reach = drift_pred[knee] >= needed  # predicted fall covers the margin
    out["iv_leaf_pooling"] = {
        "band": "margin 0.05-0.20",
        "knee_rows(cross<=42d)": int(knee.sum()),
        "plateau_rows(cross>120d_or_never)": int(plateau.sum()),
        "knee_unique_drift_predictions": int(len(np.unique(np.round(kd, 10)))),
        "knee_pred_collides_with_plateau_share": round(float(collide.mean()), 4),
        "drift_mean_knee": round(float(np.mean(kd)), 5),
        "drift_mean_plateau": round(float(np.mean(pdrift)), 5),
        "drift_p90_knee": round(float(np.percentile(kd, 90)), 5),
        "drift_p90_plateau": round(float(np.percentile(pdrift, 90)), 5),
        "knee_rows_pred_drop_covers_margin_share": round(
            float(np.mean(reach[np.isfinite(drift_pred[knee])])), 4
        ),
        "note": "collision share = knee rows whose exact drift prediction also "
        "occurs among plateau rows (same-leaf-path proxy)",
    }
    log(f"  knee {int(knee.sum())} rows, collision share "
        f"{float(collide.mean()):.3f}")

    # ------------------------------------------- (v) clamp + emergency rank
    factors_by_model = []
    seen = set()
    for building, model in by_building.items():
        if id(model) in seen:
            continue
        seen.add(id(model))
        cal = getattr(model, "calibration", None)
        factors_by_model.append(
            [round(float(f), 3) for f in cal.factors] if cal else None
        )
    production = joblib.load(ROOT / "models" / "v8_cens.joblib")
    prod_cal = getattr(production, "calibration", None)
    prod_factors = (
        [round(float(f), 3) for f in prod_cal.factors] if prod_cal else None
    )
    all_factors = [f for fs in factors_by_model if fs for f in fs]
    # bucket-0 skip: folds where the 0-45 d factor fell back to 1.0
    skipped = sum(1 for fs in factors_by_model if fs and fs[0] == 1.0)
    building_factor0 = {
        str(b): (getattr(m, "calibration").factors[0] if getattr(m, "calibration", None) else None)
        for b, m in by_building.items()
    }
    rowfeat["factor0"] = rowfeat["building"].map(building_factor0)
    low_rem = rowfeat[rowfeat["remaining"] < 45]
    skip_rows = low_rem[low_rem["factor0"] == 1.0]
    fit_rows = low_rem[low_rem["factor0"] != 1.0]
    out["v_calibration_clamp"] = {
        "clamp": [0.35, 2.75],
        "fold_factors": factors_by_model,
        "production_factors": prod_factors,
        "min_factor_seen": min(all_factors),
        "max_factor_seen": max(all_factors),
        "clamp_binding": bool(
            min(all_factors) <= 0.351 or max(all_factors) >= 2.749
        ),
        "folds_with_bucket0_skipped_to_1.0": skipped,
        "bucket0_skip_effect_rows_remaining_lt45": {
            "skipped_folds_rows": int(len(skip_rows)),
            "skipped_folds_sum_p": round(float(skip_rows["p_cal"].sum()), 1),
            "skipped_folds_dues": int(skip_rows["due"].sum()),
            "fitted_folds_rows": int(len(fit_rows)),
            "fitted_folds_sum_p": round(float(fit_rows["p_cal"].sum()), 1),
            "fitted_folds_dues": int(fit_rows["due"].sum()),
        },
    }

    ledger = pd.read_csv(ROOT / "outputs" / "audit_ledger.csv")
    # expected rank under the independent-marginal approximation, from p_cal
    exp_rank_due = []
    for scenario, sub in rowfeat.groupby("scenario"):
        ids = sub["battery"].to_numpy()
        p = sub["p_cal"].to_numpy(dtype=float)
        pos = np.argsort(ids, kind="stable")
        cum = 0.0
        expected = np.empty(len(ids))
        for j in pos:
            expected[j] = cum
            cum += p[j]
        exp_rank_due.extend(expected[sub["due"].to_numpy(dtype=bool)])
    # realized queue position of misses, from the ledger
    misses = ledger[(ledger["due"]) & (~ledger["served"])].copy()
    real_positions = []
    for scenario, sub in misses.groupby("scenario"):
        ids = np.sort(sub["battery"].to_numpy())
        real_positions.extend(range(len(ids)))
    out["v_emergency_rank"] = {
        "mean_expected_rank_of_due": round(float(np.mean(exp_rank_due)), 2),
        "mean_realized_queue_position_of_miss": round(
            float(np.mean(real_positions)), 2
        ),
        "defer_cost_inflation_hours_per_due": round(
            10.0
            * (float(np.mean(exp_rank_due)) - float(np.mean(real_positions))),
            1,
        ),
        "note": "defer_cost uses expected rank over the WHOLE fleet's p; the "
        "real queue only holds the misses. Inflation biases toward servicing "
        "(both real dues and zombies), uniform-ish in battery id.",
    }

    # ------------------------------------------------- (vi) candidate filter
    miss_rows = ledger[ledger["miss_class"].notna() & (~ledger["served"])]
    gain = miss_rows["gain_hi"].to_numpy(dtype=float)
    in_filter = miss_rows["in_filter_hi"].to_numpy(dtype=bool)
    out["vi_candidate_filter"] = {
        "margin_hours": 24.0,
        "max_candidates": 150,
        "missed_due_rows": int(len(miss_rows)),
        "outside_filter": int((~in_filter).sum()),
        "outside_filter_with_gain_gt_-24": int(
            ((~in_filter) & (gain > -24.0)).sum()
        ),
        "outside_filter_med_p": round(
            float(miss_rows.loc[~in_filter, "p"].median()), 5
        ),
        "outside_filter_gain_p90": round(
            float(np.percentile(gain[~in_filter], 90)), 2
        ),
        "inside_filter_negative_gain": int((in_filter & (gain <= 0)).sum()),
        "inside_filter_positive_gain": int((in_filter & (gain > 0)).sum()),
        "note": "outside_filter & gain>-24 would indicate the 150-cap binding; "
        "0 means the 24 h margin (i.e. the p level) is the excluder",
    }

    out_path = ROOT / "outputs" / "roadblock_l2.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
