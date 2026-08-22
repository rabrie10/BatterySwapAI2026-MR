"""Data-physics verification report for the battery-swap dataset.

Measurement only -- no forecasting model is fitted. Answers the eight
questions (inventory, trajectory shape, warning window, monotonicity, shared
curve, IR lead time, temperature structure, censoring geometry) and writes

    outputs/diag_physics.json   every headline number
    outputs/diag_physics.md     human-readable report

Run:  ./.venv/Scripts/python.exe tools/diag/physics_report.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*generic.*unit for NumPy timedelta.*", category=DeprecationWarning
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402  (sets thread caps before numpy import)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    EOL_THRESHOLD,
    GLOBAL_BETA,
    OUTPUTS,
    REF_TEMP,
    align_to,
    auc_mann_whitney,
    compensate,
    crossing_index,
    detrend_rolling_median,
    dist_stats,
    from_ordinal,
    jsonable,
    ols_beta,
    ols_slope_time,
    resmooth,
    to_ordinal,
    value_at,
)

MD: list[str] = []
RESULT: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "n/a"
    return f"{xf:.{nd}f}"


# ------------------------------------------------------------------ section 1

def section1(bm, devices, eol, cache) -> dict:
    out: dict = {}
    out["n_devices"] = int(devices["device_id"].nunique())
    out["n_buildings"] = int(devices["building_id"].nunique())
    out["n_rooms"] = int(devices["room_id"].nunique())
    per_building = devices.groupby("building_id")["device_id"].nunique().sort_values()
    out["devices_per_building"] = per_building.to_dict()
    out["devices_per_building_stats"] = dist_stats(per_building.values)

    out["metrics_rows"] = int(len(bm))
    out["metrics_span"] = {
        "first": str(bm["end_time"].min()),
        "last": str(bm["end_time"].max()),
        "days": float(
            (pd.Timestamp(bm["end_time"].max()) - pd.Timestamp(bm["end_time"].min()))
            / pd.Timedelta(days=1)
        ),
    }

    out["n_eol"] = int(eol.notna().sum())
    dev_idx = devices.set_index("device_id")
    eol_dev = eol.dropna()
    eol_per_building = (
        dev_idx.loc[eol_dev.index, "building_id"].value_counts().sort_index().to_dict()
    )
    out["eol_per_building"] = eol_per_building
    out["eol_span"] = {"first": str(eol_dev.min()), "last": str(eol_dev.max())}

    # Is devices.start_time the install date? Compare with first measurement.
    firsts = bm.groupby("device_id", observed=True)["end_time"].min()
    firsts.index = firsts.index.astype(str)
    start = dev_idx["start_time"]
    delta_days = (firsts - start.loc[firsts.index]).dt.total_seconds() / 86400.0
    out["first_reading_minus_start_days"] = dist_stats(delta_days.values)
    out["frac_first_reading_within_1d_of_start"] = float((delta_days.abs() <= 1).mean())
    out["frac_first_reading_within_3d_of_start"] = float((delta_days.abs() <= 3).mean())

    # Voltage at first observation: first raw reading, and first smoothed value.
    order = bm.sort_values("end_time")
    first_rows = order.groupby("device_id", observed=True).head(1)
    first_v_raw = first_rows.set_index(first_rows["device_id"].astype(str))["voltage"].astype(float)
    out["first_raw_voltage"] = dist_stats(first_v_raw.values)

    first_smooth = {}
    for dev_id, series in cache.devices.items():
        finite = np.flatnonzero(np.isfinite(series.smooth_voltage))
        if finite.size:
            first_smooth[dev_id] = float(series.smooth_voltage[finite[0]])
    fs = np.array(list(first_smooth.values()))
    out["first_smooth_voltage"] = dist_stats(fs)
    out["first_smooth_voltage_bands"] = {
        ">3.05": float((fs > 3.05).mean()),
        "2.95-3.05": float(((fs > 2.95) & (fs <= 3.05)).mean()),
        "2.85-2.95": float(((fs > 2.85) & (fs <= 2.95)).mean()),
        "2.70-2.85": float(((fs > 2.70) & (fs <= 2.85)).mean()),
        "<=2.70": float((fs <= 2.70).mean()),
    }
    out["devices_start_time_span"] = {
        "first": str(devices["start_time"].min()),
        "last": str(devices["start_time"].max()),
    }
    out["devices_end_time_span"] = {
        "first": str(devices["end_time"].min()),
        "last": str(devices["end_time"].max()),
    }

    MD.append("## 1. Inventory\n")
    MD.append(f"- Devices: **{out['n_devices']}** across **{out['n_buildings']}** buildings "
              f"({out['n_rooms']} rooms); devices/building median {fmt(out['devices_per_building_stats']['median'],1)} "
              f"(min {int(per_building.min())}, max {int(per_building.max())}).")
    MD.append(f"- Metrics: {out['metrics_rows']:,} hourly rows, {out['metrics_span']['first']} to "
              f"{out['metrics_span']['last']} ({out['metrics_span']['days']:.0f} days).")
    MD.append(f"- Non-null EOL times: **{out['n_eol']}** (first {out['eol_span']['first'][:10]}, "
              f"last {out['eol_span']['last'][:10]}). Per building: {eol_per_building}")
    MD.append(f"- First reading minus devices.start_time: median {fmt(out['first_reading_minus_start_days']['median'],2)} d, "
              f"p90 {fmt(out['first_reading_minus_start_days']['p90'],2)} d; within 1 day for "
              f"{100*out['frac_first_reading_within_1d_of_start']:.1f}% of devices -> start_time is the install/first-observation date "
              f"(a few stragglers: max gap {fmt(out['first_reading_minus_start_days']['max'],0)} d).")
    MD.append(f"- First raw voltage: median {fmt(out['first_raw_voltage']['median'],3)} V "
              f"(IQR {fmt(out['first_raw_voltage']['p25'],3)}-{fmt(out['first_raw_voltage']['p75'],3)}). "
              f"First smoothed voltage: median {fmt(out['first_smooth_voltage']['median'],3)} V. Bands: "
              + ", ".join(f"{k}: {100*v:.1f}%" for k, v in out["first_smooth_voltage_bands"].items()))
    MD.append("")
    return out


# ------------------------------------------------------------------ section 2

def get_crossings(cache, eol) -> dict[str, dict]:
    """Per EOL device: grid crossing index / ordinal, agreement with CSV."""
    crossings = {}
    for dev_id, eol_time in eol.dropna().items():
        series = cache.devices.get(dev_id)
        if series is None:
            continue
        idx = crossing_index(series.smooth_voltage)
        if idx is None:
            continue
        ordinal = series.origin + idx
        crossings[dev_id] = {
            "index": idx,
            "ordinal": ordinal,
            "date": str(from_ordinal(ordinal).date()),
            "csv_date": str(pd.Timestamp(eol_time).date()),
            "csv_minus_grid_days": int(to_ordinal(eol_time) - ordinal),
        }
    return crossings


def section2(cache, crossings) -> dict:
    out: dict = {}
    diffs = [c["csv_minus_grid_days"] for c in crossings.values()]
    out["n_eol_with_grid_crossing"] = len(crossings)
    out["csv_vs_grid_crossing_agreement"] = {
        "exact_match_frac": float(np.mean(np.array(diffs) == 0)),
        "diff_days": dist_stats(diffs),
    }

    offsets = [7, 14, 28, 42, 60, 90, 180]
    values = {k: [] for k in offsets}
    at_cross = []
    frac_above_255_at_42 = []
    slopes_14 = []
    diff_14 = []
    observed_days_before_cross = []
    for dev_id, cross in crossings.items():
        series = cache.devices[dev_id]
        sv = series.smooth_voltage
        c = cross["index"]
        first_finite = int(np.flatnonzero(np.isfinite(sv))[0])
        observed_days_before_cross.append(c - first_finite)
        at_cross.append(value_at(sv, c))
        for k in offsets:
            values[k].append(value_at(sv, c - k))
        v42 = value_at(sv, c - 42)
        if np.isfinite(v42):
            frac_above_255_at_42.append(v42 > 2.55)
        # final-14-day slope on the smoothed series ending at the crossing day
        lo = max(0, c - 13)
        slopes_14.append(ols_slope_time(sv[lo : c + 1], min_points=8))
        v14 = value_at(sv, c - 14)
        v0 = value_at(sv, c)
        if np.isfinite(v14) and np.isfinite(v0):
            diff_14.append((v0 - v14) / 14.0)

    out["observed_days_before_crossing"] = dist_stats(observed_days_before_cross)
    out["smooth_v_at_crossing"] = dist_stats(at_cross)
    out["smooth_v_before_crossing"] = {
        f"minus_{k}d": dist_stats(values[k]) for k in offsets
    }
    out["frac_above_2.55_at_42d_out"] = {
        "frac": float(np.mean(frac_above_255_at_42)),
        "n": len(frac_above_255_at_42),
    }
    out["post_knee_slope_final14d_V_per_day"] = dist_stats(slopes_14)
    out["post_knee_diff_final14d_V_per_day"] = dist_stats(diff_14)

    MD.append("## 2. Trajectory shape (82 EOL devices, smoothed series)\n")
    ag = out["csv_vs_grid_crossing_agreement"]
    MD.append(f"- Grid crossing (first smooth_v < 2.4) matches eol_times.csv exactly for "
              f"{100*ag['exact_match_frac']:.1f}% of {len(crossings)} devices "
              f"(diff median {fmt(ag['diff_days']['median'],1)} d, max |diff| "
              f"{max(abs(ag['diff_days']['min']), abs(ag['diff_days']['max'])):.0f} d).")
    MD.append("\n| days before crossing | median smooth_v | IQR (p25-p75) | n |")
    MD.append("|---|---|---|---|")
    row = out["smooth_v_at_crossing"]
    MD.append(f"| 0 (at crossing) | {fmt(row['median'],3)} | {fmt(row['p25'],3)}-{fmt(row['p75'],3)} | {row['n']} |")
    for k in offsets:
        row = out["smooth_v_before_crossing"][f"minus_{k}d"]
        if row.get("n", 0):
            MD.append(f"| {k} | {fmt(row['median'],3)} | {fmt(row['p25'],3)}-{fmt(row['p75'],3)} | {row['n']} |")
    MD.append("")
    MD.append(f"- **Median smooth_v 42 d before crossing: {fmt(out['smooth_v_before_crossing']['minus_42d']['median'],3)} V**; "
              f"**{100*out['frac_above_2.55_at_42d_out']['frac']:.1f}%** of EOL devices still above 2.55 V at 42 d out "
              f"(n={out['frac_above_2.55_at_42d_out']['n']}).")
    MD.append(f"- Post-knee slope (OLS over final 14 d): median **{fmt(out['post_knee_slope_final14d_V_per_day']['median'],4)} V/day** "
              f"(IQR {fmt(out['post_knee_slope_final14d_V_per_day']['p25'],4)} to {fmt(out['post_knee_slope_final14d_V_per_day']['p75'],4)}); "
              f"14-day difference quotient median {fmt(out['post_knee_diff_final14d_V_per_day']['median'],4)} V/day.")
    o = out["observed_days_before_crossing"]
    MD.append(f"- Observed smoothed life before crossing: median {fmt(o['median'],0)} d "
              f"(min {fmt(o['min'],0)}, p10 {fmt(o['p10'],0)}); rows with n<82 above are devices whose "
              f"observation started less than k days before their crossing.")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 3

def _warning_windows(cache, crossings, thresholds, mode: str, beta_map=None) -> dict:
    per_threshold = {x: [] for x in thresholds}
    never_above = {x: 0 for x in thresholds}
    for dev_id, cross in crossings.items():
        series = cache.devices[dev_id]
        v = series.smooth_voltage
        if mode == "global":
            v = compensate(series.smooth_voltage, series.smooth_temperature)
        elif mode == "perdev":
            v = compensate(
                series.smooth_voltage,
                series.smooth_temperature,
                beta_map.get(dev_id, GLOBAL_BETA),
            )
        c = cross["index"]
        seg = v[: c + 1]
        for x in thresholds:
            above = np.flatnonzero(np.isfinite(seg) & (seg > x))
            if above.size == 0:
                never_above[x] += 1
                continue
            per_threshold[x].append(c - int(above[-1]))
    return {
        f"{x:.2f}": {**dist_stats(days), "n_never_above": never_above[x]}
        for x, days in per_threshold.items()
    }


def section3(cache, crossings) -> dict:
    thresholds = [2.70, 2.60, 2.55, 2.50, 2.45]
    det_beta = {}
    for dev_id in crossings:
        s = cache.devices[dev_id]
        b = ols_beta(
            detrend_rolling_median(s.temperature),
            detrend_rolling_median(s.voltage),
            min_points=60,
        )
        det_beta[dev_id] = b if np.isfinite(b) else GLOBAL_BETA
    out = {
        "raw": _warning_windows(cache, crossings, thresholds, "raw"),
        "compensated": _warning_windows(cache, crossings, thresholds, "global"),
        "compensated_perdev": _warning_windows(
            cache, crossings, thresholds, "perdev", beta_map=det_beta
        ),
        "compensation": {"beta": GLOBAL_BETA, "ref_temp": REF_TEMP,
                         "perdev": "detrended per-device beta"},
    }

    # Bonus: where would the compensated series cross 2.4 relative to official?
    shift = []
    for dev_id, cross in crossings.items():
        series = cache.devices[dev_id]
        comp = compensate(series.smooth_voltage, series.smooth_temperature)
        idx = crossing_index(comp)
        if idx is not None:
            shift.append(idx - cross["index"])
    out["compensated_crossing_minus_official_days"] = dist_stats(shift)

    MD.append("## 3. Warning window of a pure level threshold\n")
    MD.append("Days between the last day smooth_v > X and the 2.4 V crossing (per EOL device). "
              "comp = global 0.00463 V/degC; pd-comp = per-device detrended beta.\n")
    MD.append("| X | raw median | raw IQR | raw CV | comp median | comp IQR | comp CV | pd-comp median | pd-comp IQR | pd-comp CV |")
    MD.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in thresholds:
        key = f"{x:.2f}"
        r, c, p = out["raw"][key], out["compensated"][key], out["compensated_perdev"][key]
        MD.append(
            f"| {key} | {fmt(r['median'],1)} | {fmt(r['iqr'],1)} | {fmt(r['cv'],2)} "
            f"| {fmt(c['median'],1)} | {fmt(c['iqr'],1)} | {fmt(c['cv'],2)} "
            f"| {fmt(p['median'],1)} | {fmt(p['iqr'],1)} | {fmt(p['cv'],2)} |"
        )
    tighter = sum(
        1 for x in thresholds
        if out["compensated"][f"{x:.2f}"].get("iqr", np.inf) < out["raw"][f"{x:.2f}"].get("iqr", np.inf)
    )
    tighter_pd = sum(
        1 for x in thresholds
        if out["compensated_perdev"][f"{x:.2f}"].get("iqr", np.inf) < out["raw"][f"{x:.2f}"].get("iqr", np.inf)
    )
    MD.append("")
    MD.append(f"- Global compensation tightens the IQR for **{tighter}/{len(thresholds)}** thresholds; "
              f"per-device compensation for **{tighter_pd}/{len(thresholds)}**.")
    s = out["compensated_crossing_minus_official_days"]
    MD.append(f"- Compensated series' own 2.4 V crossing vs official: median shift {fmt(s.get('median'),1)} d "
              f"(IQR {fmt(s.get('iqr'),1)} d, n={s.get('n',0)}).")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 4

def _mono_metrics(values_by_device: dict[str, np.ndarray], lag: int = 1) -> dict:
    """Pooled ``lag``-day deltas: fraction of increases and up/down mass."""
    n_up = n_down = n_zero = 0
    up_sum = down_sum = 0.0
    up_big = down_big = 0.0
    per_device_frac = []
    for v in values_by_device.values():
        finite = np.isfinite(v)
        both = finite[lag:] & finite[:-lag]
        d = (v[lag:] - v[:-lag])[both]
        if d.size == 0:
            continue
        n_up += int((d > 0).sum())
        n_down += int((d < 0).sum())
        n_zero += int((d == 0).sum())
        up_sum += float(d[d > 0].sum())
        down_sum += float(-d[d < 0].sum())
        big = np.abs(d) > 0.005
        up_big += float(d[(d > 0) & big].sum())
        down_big += float(-d[(d < 0) & big].sum())
        per_device_frac.append(float((d > 0).mean()))
    total = n_up + n_down + n_zero
    return {
        "n_deltas": total,
        "frac_up_all": n_up / total if total else float("nan"),
        "frac_up_nonzero": n_up / (n_up + n_down) if (n_up + n_down) else float("nan"),
        "frac_zero": n_zero / total if total else float("nan"),
        "up_down_mass_ratio": up_sum / down_sum if down_sum else float("nan"),
        "up_down_mass_ratio_gt5mV": up_big / down_big if down_big else float("nan"),
        "per_device_frac_up": dist_stats(per_device_frac),
    }


def section4(cache) -> dict:
    eligible = {
        dev_id: s
        for dev_id, s in cache.devices.items()
        if int(np.isfinite(s.smooth_voltage).sum()) >= 180
    }
    out: dict = {"n_devices_ge_180d": len(eligible)}

    raw = {d: s.smooth_voltage for d, s in eligible.items()}
    glob = {
        d: compensate(s.smooth_voltage, s.smooth_temperature)
        for d, s in eligible.items()
    }
    per_beta = {}
    det_beta = {}
    per_dev = {}
    per_dev_resmoothed = {}
    glob_resmoothed = {}
    det_dev = {}
    det_dev_resmoothed = {}
    for d, s in eligible.items():
        beta = ols_beta(s.temperature, s.voltage, min_points=60)
        if not np.isfinite(beta):
            beta = GLOBAL_BETA
        per_beta[d] = beta
        per_dev[d] = compensate(s.smooth_voltage, s.smooth_temperature, beta)
        # variant: compensate the daily medians, then re-apply the official smoother
        per_dev_resmoothed[d] = resmooth(compensate(s.voltage, s.temperature, beta))
        glob_resmoothed[d] = resmooth(compensate(s.voltage, s.temperature, GLOBAL_BETA))
        # detrended beta: removes the aging-trend x season confound that inflates
        # the whole-life regression
        db = ols_beta(
            detrend_rolling_median(s.temperature),
            detrend_rolling_median(s.voltage),
            min_points=60,
        )
        if not np.isfinite(db):
            db = GLOBAL_BETA
        det_beta[d] = db
        det_dev[d] = compensate(s.smooth_voltage, s.smooth_temperature, db)
        det_dev_resmoothed[d] = resmooth(compensate(s.voltage, s.temperature, db))

    out["per_device_beta_wholelife_V_per_degC"] = dist_stats(list(per_beta.values()))
    out["per_device_beta_detrended_V_per_degC"] = dist_stats(list(det_beta.values()))
    out["a_raw"] = _mono_metrics(raw)
    out["b_global_comp"] = _mono_metrics(glob)
    out["c_per_device_comp"] = _mono_metrics(per_dev)
    out["b2_global_comp_resmoothed"] = _mono_metrics(glob_resmoothed)
    out["c2_per_device_comp_resmoothed"] = _mono_metrics(per_dev_resmoothed)
    out["d_per_device_detrended_comp"] = _mono_metrics(det_dev)
    out["d2_per_device_detrended_comp_resmoothed"] = _mono_metrics(det_dev_resmoothed)
    # Horizon test: does the upward movement survive at a 28-day lag, or is it
    # short-horizon jitter around a monotone trend?
    out["lag28_a_raw"] = _mono_metrics(raw, lag=28)
    out["lag28_b2_global_comp_resmoothed"] = _mono_metrics(glob_resmoothed, lag=28)
    out["lag28_d2_per_device_detrended_comp_resmoothed"] = _mono_metrics(
        det_dev_resmoothed, lag=28
    )

    MD.append("## 4. Monotonicity of the smoothed series\n")
    MD.append(f"Devices with >=180 smoothed days: **{len(eligible)}**. Adjacent-day deltas pooled.\n")
    MD.append("| series | frac deltas > 0 | frac >0 (excl. ties) | up/down mass ratio | ratio (only deltas >5 mV) | n deltas |")
    MD.append("|---|---|---|---|---|---|")
    for label, key in [
        ("(a) raw smooth_v", "a_raw"),
        ("(b) global comp (0.00463)", "b_global_comp"),
        ("(c) per-device whole-life-beta comp", "c_per_device_comp"),
        ("(b2) global comp, re-smoothed daily", "b2_global_comp_resmoothed"),
        ("(c2) per-device whole-life comp, re-smoothed", "c2_per_device_comp_resmoothed"),
        ("(d) per-device detrended-beta comp", "d_per_device_detrended_comp"),
        ("(d2) per-device detrended comp, re-smoothed", "d2_per_device_detrended_comp_resmoothed"),
    ]:
        m = out[key]
        MD.append(f"| {label} | **{fmt(m['frac_up_all'],4)}** | {fmt(m['frac_up_nonzero'],4)} "
                  f"| **{fmt(m['up_down_mass_ratio'],4)}** | {fmt(m['up_down_mass_ratio_gt5mV'],4)} | {m['n_deltas']:,} |")
    MD.append("")
    MD.append("28-day-lag deltas (does upward movement survive a monthly horizon?):\n")
    MD.append("| series (lag 28) | frac deltas > 0 | up/down mass ratio |")
    MD.append("|---|---|---|")
    for label, key in [
        ("(a) raw smooth_v", "lag28_a_raw"),
        ("(b2) global comp, re-smoothed", "lag28_b2_global_comp_resmoothed"),
        ("(d2) per-device detrended comp, re-smoothed", "lag28_d2_per_device_detrended_comp_resmoothed"),
    ]:
        m = out[key]
        MD.append(f"| {label} | **{fmt(m['frac_up_all'],4)}** | **{fmt(m['up_down_mass_ratio'],4)}** |")
    b = out["per_device_beta_wholelife_V_per_degC"]
    db = out["per_device_beta_detrended_V_per_degC"]
    MD.append("")
    MD.append(f"- Whole-life per-device beta: median {fmt(b['median'],5)} V/degC "
              f"(IQR {fmt(b['p25'],5)}-{fmt(b['p75'],5)}); detrended per-device beta: median {fmt(db['median'],5)} "
              f"(IQR {fmt(db['p25'],5)}-{fmt(db['p75'],5)}). The whole-life regression is inflated by the "
              f"aging-trend x season confound, which is why variant (c) can over-correct.")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 5

def section5(cache, crossings) -> dict:
    levels = [2.9, 2.8, 2.7, 2.6, 2.5, 2.4]
    bands = list(zip(levels[:-1], levels[1:]))
    out: dict = {}

    band_times: dict[str, dict[str, float]] = {}
    n_high_start = 0
    for dev_id, cross in crossings.items():
        series = cache.devices[dev_id]
        comp = compensate(series.smooth_voltage, series.smooth_temperature)
        finite = np.flatnonzero(np.isfinite(comp))
        if finite.size == 0:
            continue
        first_v = comp[finite[0]]
        if first_v <= 2.85:
            continue
        n_high_start += 1
        fp = {}
        for lv in levels:
            idx = crossing_index(comp, lv)
            fp[lv] = idx
        times = {}
        for hi, lo in bands:
            if first_v > hi and fp[hi] is not None and fp[lo] is not None:
                dt = fp[lo] - fp[hi]
                if dt >= 0:
                    times[f"{hi:.1f}->{lo:.1f}"] = float(dt)
        # coarse halves; note they share the fp(2.6) boundary, whose noise
        # enters the two times with opposite signs (see the separated-band
        # correlations for a boundary-free test)
        if first_v > 2.8 and fp[2.8] is not None and fp[2.6] is not None and fp[2.4] is not None:
            times["half_2.8->2.6"] = float(fp[2.6] - fp[2.8])
            times["half_2.6->2.4"] = float(fp[2.4] - fp[2.6])
            times["total_2.8->2.4"] = float(fp[2.4] - fp[2.8])
        if times:
            band_times[dev_id] = times

    out["n_eol_first_comp_v_gt_2.85"] = n_high_start
    band_keys = [f"{hi:.1f}->{lo:.1f}" for hi, lo in bands]
    per_band = {
        k: dist_stats([t[k] for t in band_times.values() if k in t]) for k in band_keys
    }
    out["band_first_passage_days"] = per_band

    # Proportionality: correlation of log band-times across bands, over devices.
    full = pd.DataFrame.from_dict(band_times, orient="index")
    tbl = full.reindex(columns=band_keys)
    logt = np.log(tbl.where(tbl > 0))
    corr = logt.corr(min_periods=8)
    out["log_bandtime_correlation"] = corr.round(3).to_dict()
    adj, nonadj = [], []
    for i in range(len(band_keys)):
        for j in range(i + 1, len(band_keys)):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                (adj if j == i + 1 else nonadj).append(float(v))
    pairs = adj + nonadj
    out["log_bandtime_mean_pairwise_corr"] = float(np.mean(pairs)) if pairs else float("nan")
    out["log_bandtime_mean_corr_adjacent"] = float(np.mean(adj)) if adj else float("nan")
    out["log_bandtime_mean_corr_nonadjacent"] = float(np.mean(nonadj)) if nonadj else float("nan")

    # Coarse halves 2.8->2.6 vs 2.6->2.4 (share the fp(2.6) boundary).
    halves = full.reindex(columns=["half_2.8->2.6", "half_2.6->2.4", "total_2.8->2.4"])
    hl = np.log(halves.where(halves > 0))
    ok = hl[["half_2.8->2.6", "half_2.6->2.4"]].dropna()
    out["n_devices_halves"] = int(len(ok))
    if len(ok) >= 8:
        out["halves_log_corr_pearson"] = float(ok.corr().iloc[0, 1])
        out["halves_log_corr_spearman"] = float(ok.corr(method="spearman").iloc[0, 1])
    else:
        out["halves_log_corr_pearson"] = float("nan")
        out["halves_log_corr_spearman"] = float("nan")
    out["total_2.8->2.4_days"] = dist_stats(halves["total_2.8->2.4"].dropna().values)

    # Boundary-free proportionality tests: bands separated by a full band, so
    # no first-passage boundary is shared between the two times.
    def _sep_corr(col_a: str, col_b: str) -> dict:
        pair = logt[[col_a, col_b]].dropna()
        if len(pair) < 8:
            return {"n": int(len(pair))}
        return {
            "n": int(len(pair)),
            "pearson": float(pair.corr().iloc[0, 1]),
            "spearman": float(pair.corr(method="spearman").iloc[0, 1]),
        }

    out["separated_corr_2.8-2.7_vs_2.6-2.5"] = _sep_corr("2.8->2.7", "2.6->2.5")
    out["separated_corr_2.9-2.8_vs_2.5-2.4"] = _sep_corr("2.9->2.8", "2.5->2.4")
    out["separated_corr_2.8-2.7_vs_2.5-2.4"] = _sep_corr("2.8->2.7", "2.5->2.4")

    # Two-way additive decomposition of log band-time: band effect + device effect.
    stacked = logt.stack().rename("logt").reset_index()
    stacked.columns = ["device", "band", "logt"]
    if len(stacked) > 10:
        grand = stacked["logt"].mean()
        band_eff = stacked.groupby("band")["logt"].mean() - grand
        dev_eff = stacked.groupby("device")["logt"].mean() - grand
        fitted = grand + stacked["band"].map(band_eff) + stacked["device"].map(dev_eff)
        resid = stacked["logt"] - fitted
        sst = float(((stacked["logt"] - grand) ** 2).sum())
        out["log_bandtime_additive_R2"] = 1.0 - float((resid ** 2).sum()) / sst if sst else float("nan")
        sst_band_removed = float(((stacked["logt"] - grand - stacked["band"].map(band_eff)) ** 2).sum())
        out["log_bandtime_device_R2_after_band"] = (
            1.0 - float((resid ** 2).sum()) / sst_band_removed if sst_band_removed else float("nan")
        )
    else:
        out["log_bandtime_additive_R2"] = float("nan")
        out["log_bandtime_device_R2_after_band"] = float("nan")

    MD.append("## 5. Shared curve / rate constancy (EOL devices starting > 2.85 V, compensated)\n")
    MD.append(f"- Devices qualifying (first compensated smooth_v > 2.85): **{n_high_start}** of {len(crossings)} EOL devices.\n")
    MD.append("| band | median days | IQR | CV | n |")
    MD.append("|---|---|---|---|---|")
    for k in band_keys:
        s = per_band[k]
        if s.get("n", 0):
            MD.append(f"| {k} | {fmt(s['median'],1)} | {fmt(s['p25'],1)}-{fmt(s['p75'],1)} | **{fmt(s['cv'],2)}** | {s['n']} |")
    MD.append("")
    MD.append(f"- Mean pairwise correlation of log band-times: **{fmt(out['log_bandtime_mean_pairwise_corr'],3)}** "
              f"(adjacent bands {fmt(out['log_bandtime_mean_corr_adjacent'],3)}, "
              f"non-adjacent {fmt(out['log_bandtime_mean_corr_nonadjacent'],3)}).")
    MD.append(f"- Halves test (shares fp(2.6) boundary), log t(2.8->2.6) vs log t(2.6->2.4), n={out['n_devices_halves']}: "
              f"Pearson {fmt(out['halves_log_corr_pearson'],3)}, Spearman {fmt(out['halves_log_corr_spearman'],3)}.")
    for label, key in [
        ("t(2.8->2.7) vs t(2.6->2.5)", "separated_corr_2.8-2.7_vs_2.6-2.5"),
        ("t(2.9->2.8) vs t(2.5->2.4)", "separated_corr_2.9-2.8_vs_2.5-2.4"),
        ("t(2.8->2.7) vs t(2.5->2.4)", "separated_corr_2.8-2.7_vs_2.5-2.4"),
    ]:
        s = out[key]
        MD.append(f"- Boundary-free {label} (n={s.get('n',0)}): Pearson **{fmt(s.get('pearson'),3)}**, "
                  f"Spearman {fmt(s.get('spearman'),3)}.")
    t = out["total_2.8->2.4_days"]
    MD.append(f"- Total 2.8->2.4 time: median {fmt(t.get('median'),1)} d (IQR {fmt(t.get('p25'),1)}-{fmt(t.get('p75'),1)}, "
              f"CV {fmt(t.get('cv'),2)}).")
    MD.append(f"- Additive model log(t) = band + device explains R^2 = {fmt(out['log_bandtime_additive_R2'],3)}; "
              f"device effect after removing band means: R^2 = {fmt(out['log_bandtime_device_R2_after_band'],3)}.")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 6

def section6(cache, shape_cache, crossings, devices, eol) -> dict:
    out: dict = {}
    leads = [0, 14, 28, 42, 60, 90, 120, 180]

    views = {}
    for dev_id, series in cache.devices.items():
        shape = shape_cache.devices.get(dev_id)
        views[dev_id] = (series.origin, align_to(shape, series.origin, len(series)))

    beta_at = {k: [] for k in leads}
    exceed_leads = []
    n_no_baseline = 0
    n_never_exceed = 0
    baselines = []
    for dev_id, cross in crossings.items():
        origin, view = views[dev_id]
        c = cross["index"]
        for k in leads:
            beta_at[k].append(view.trailing_mean("beta", c - k, 7))
        # 180-day baseline: median of daily beta over days before crossing-180
        shape = shape_cache.devices.get(dev_id)
        baseline = float("nan")
        if shape is not None:
            arr = np.full(len(cache.devices[dev_id]), np.nan)
            offset = shape.origin - origin
            src0 = max(0, -offset)
            tgt0 = max(0, offset)
            span = min(shape.beta.size - src0, arr.size - tgt0)
            if span > 0:
                arr[tgt0 : tgt0 + span] = shape.beta[src0 : src0 + span]
            early = arr[: max(0, c - 180)]
            early = early[np.isfinite(early)]
            if early.size >= 20:
                baseline = float(np.median(early))
        if not np.isfinite(baseline):
            n_no_baseline += 1
            continue
        baselines.append(baseline)
        lead = None
        for d in range(max(0, c - 180), c + 1):
            b = view.trailing_mean("beta", d, 7)
            if np.isfinite(b) and b >= 2.0 * baseline:
                lead = c - d
                break
        if lead is None:
            n_never_exceed += 1
        else:
            exceed_leads.append(lead)

    out["trailing7_beta_at_lead"] = {f"minus_{k}d": dist_stats(beta_at[k]) for k in leads}
    out["baseline_beta_pre180"] = dist_stats(baselines)
    out["lead_first_exceed_2x_baseline_days"] = dist_stats(exceed_leads)
    out["n_no_baseline"] = n_no_baseline
    out["n_never_exceed_2x_within_180d"] = n_never_exceed
    # A lead of exactly 180 means beta was already >= 2x baseline at the scan
    # start, i.e. the true lead is right-censored at 180 days.
    out["n_lead_censored_at_180"] = int(sum(1 for v in exceed_leads if v == 180))

    # Matched control comparison at 42-day lead.
    dev_idx = devices.set_index("device_id")
    obs_end_ord = {d: to_ordinal(t) for d, t in dev_idx["end_time"].items()}
    cross_ord = {d: c["ordinal"] for d, c in crossings.items()}
    positives, controls = [], []
    for dev_id, cross in crossings.items():
        origin, view = views[dev_id]
        c = cross["index"]
        b = view.trailing_mean("beta", c - 42, 7)
        if not np.isfinite(b):
            continue
        positives.append(b)
        date_ord = cross["ordinal"] - 42
        for other, (o_origin, o_view) in views.items():
            oc = cross_ord.get(other)
            if oc is not None and oc <= date_ord + 120:
                continue  # crosses within 120 days (or already crossed)
            if oc is None and obs_end_ord.get(other, -1) < date_ord + 120:
                continue  # censored before survival is established
            ob = o_view.trailing_mean("beta", date_ord - o_origin, 7)
            if np.isfinite(ob):
                controls.append(ob)
    positives = np.asarray(positives)
    controls = np.asarray(controls)
    out["auc_beta_42d_lead"] = auc_mann_whitney(positives, controls)
    out["beta_42d_lead_positives"] = dist_stats(positives)
    out["beta_42d_lead_controls"] = dist_stats(controls)

    MD.append("## 6. IR signal (within-day dV/dT beta) lead time\n")
    MD.append("| lead (days before crossing) | median trailing-7d beta | IQR | n |")
    MD.append("|---|---|---|---|")
    for k in leads:
        s = out["trailing7_beta_at_lead"][f"minus_{k}d"]
        if s.get("n", 0):
            MD.append(f"| {k} | {fmt(s['median'],5)} | {fmt(s['p25'],5)}-{fmt(s['p75'],5)} | {s['n']} |")
    MD.append("")
    bl = out["baseline_beta_pre180"]
    le = out["lead_first_exceed_2x_baseline_days"]
    MD.append(f"- Own baseline (median daily beta, days < crossing-180): median {fmt(bl.get('median'),5)} V/degC (n={bl.get('n',0)}).")
    MD.append(f"- Lead time at which trailing-7d beta first exceeds 2x own baseline: median **{fmt(le.get('median'),1)} d** "
              f"(IQR {fmt(le.get('p25'),1)}-{fmt(le.get('p75'),1)}, n={le.get('n',0)}; "
              f"{out['n_never_exceed_2x_within_180d']} never exceed within 180 d, {out['n_no_baseline']} lacked a baseline; "
              f"**{out['n_lead_censored_at_180']}** already exceeded at the 180 d scan edge, so their true lead is >=180 d).")
    MD.append(f"- 42-day-lead separation, EOL (n={out['beta_42d_lead_positives'].get('n',0)}, "
              f"median {fmt(out['beta_42d_lead_positives'].get('median'),5)}) vs matched surviving device-days "
              f"(n={out['beta_42d_lead_controls'].get('n',0)}, median {fmt(out['beta_42d_lead_controls'].get('median'),5)}): "
              f"**AUC = {fmt(out['auc_beta_42d_lead'],3)}**.")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 7

def section7(cache, shape_cache, devices, crossings) -> dict:
    out: dict = {}
    corrs, betas = {}, {}
    for dev_id, s in cache.devices.items():
        dv = detrend_rolling_median(s.voltage)
        dt = detrend_rolling_median(s.temperature)
        ok = np.isfinite(dv) & np.isfinite(dt)
        if ok.sum() < 90:
            continue
        x, y = dt[ok], dv[ok]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        corrs[dev_id] = float(np.corrcoef(x, y)[0, 1])
        betas[dev_id] = float(((x - x.mean()) * (y - y.mean())).mean() / x.var())

    out["detrended_daily_V_T_correlation"] = dist_stats(list(corrs.values()))
    out["detrended_beta_V_per_degC"] = dist_stats(list(betas.values()))

    # Beta near EOL vs earlier (per EOL device, detrended within each window).
    final_betas, early_betas, ratios = [], [], []
    for dev_id, cross in crossings.items():
        s = cache.devices[dev_id]
        c = cross["index"]
        dv = detrend_rolling_median(s.voltage)
        dt = detrend_rolling_median(s.temperature)
        f0 = max(0, c - 89)
        b_final = ols_beta(dt[f0 : c + 1], dv[f0 : c + 1], min_points=30)
        b_early = ols_beta(dt[:f0], dv[:f0], min_points=30)
        if np.isfinite(b_final):
            final_betas.append(b_final)
        if np.isfinite(b_early):
            early_betas.append(b_early)
        if np.isfinite(b_final) and np.isfinite(b_early) and b_early > 0:
            ratios.append(b_final / b_early)
    out["beta_final90d_of_EOL_devices"] = dist_stats(final_betas)
    out["beta_earlier_life_of_EOL_devices"] = dist_stats(early_betas)
    out["beta_final90_over_early_ratio"] = dist_stats(ratios)
    out["frac_final90_beta_gt_early"] = (
        float(np.mean(np.array(ratios) > 1)) if ratios else float("nan")
    )

    # Within-day temperature excitation (no temp filter; from ShapeCache).
    t_ranges = np.concatenate(
        [sh.t_range[np.isfinite(sh.t_range)] for sh in shape_cache.devices.values()]
    )
    out["daily_temperature_range_degC"] = dist_stats(t_ranges)

    # Annual swing of smoothed temperature, per building.
    frame = cache.frame()
    dev_map = devices.set_index("device_id")["building_id"]
    frame["building_id"] = frame["device_id"].map(dev_map)
    frame = frame.dropna(subset=["temperature"])
    frame["month"] = frame["end_time"].dt.month
    monthly = frame.groupby(["building_id", "month"], observed=True)["temperature"].mean()
    swings = monthly.groupby(level=0).agg(lambda m: m.max() - m.min())
    out["annual_smoothT_swing_per_building"] = swings.round(2).to_dict()
    out["annual_smoothT_swing_stats"] = dist_stats(swings.values)

    MD.append("## 7. Temperature structure\n")
    c = out["detrended_daily_V_T_correlation"]
    b = out["detrended_beta_V_per_degC"]
    MD.append(f"- Detrended (60 d rolling-median) daily V vs T: per-device correlation median **{fmt(c['median'],3)}** "
              f"(IQR {fmt(c['p25'],3)}-{fmt(c['p75'],3)}, n={c['n']}).")
    MD.append(f"- Detrended per-device beta: median **{fmt(b['median'],5)} V/degC** (IQR {fmt(b['p25'],5)}-{fmt(b['p75'],5)}).")
    MD.append(f"- EOL devices, beta in final 90 d: median {fmt(out['beta_final90d_of_EOL_devices'].get('median'),5)} "
              f"vs earlier life {fmt(out['beta_earlier_life_of_EOL_devices'].get('median'),5)}; "
              f"paired ratio median {fmt(out['beta_final90_over_early_ratio'].get('median'),2)}x, "
              f"final>early for {100*out['frac_final90_beta_gt_early']:.1f}% of devices.")
    t = out["daily_temperature_range_degC"]
    MD.append(f"- Within-day temperature range (unfiltered): median {fmt(t['median'],2)} degC "
              f"(IQR {fmt(t['p25'],2)}-{fmt(t['p75'],2)}, p90 {fmt(t['p90'],2)}).")
    s = out["annual_smoothT_swing_stats"]
    MD.append(f"- Annual swing of smoothed T per building (max minus min monthly mean): median {fmt(s['median'],2)} degC "
              f"(min {fmt(s['min'],2)}, max {fmt(s['max'],2)} across {s['n']} buildings).")
    MD.append("")
    return out


# ------------------------------------------------------------------ section 8

def section8(cache, devices, eol, scenarios, crossings) -> dict:
    out: dict = {}
    windows = [
        (pd.Timestamp(sc["start_time"]), int(sc["settings"]["planning_window_days"]))
        for sc in scenarios
    ]
    out["n_scenarios"] = len(scenarios)
    out["planning_window_days"] = sorted({w for _, w in windows})
    out["scenario_start_span"] = {
        "first": str(min(w[0] for w in windows)),
        "last": str(max(w[0] for w in windows)),
    }
    out["unobserved_eol_days_setting"] = sorted(
        {sc["settings"]["unobserved_eol_days"] for sc in scenarios}
    )

    end_times = devices.set_index("device_id")["end_time"]
    data_end = end_times.max()
    inside_counts, fully_inside = [], []
    for start, days in windows:
        stop = start + pd.Timedelta(days=days)
        n = int(((end_times >= start) & (end_times < stop)).sum())
        inside_counts.append(n)
        if stop <= data_end:
            fully_inside.append(n)
    out["devices_ending_inside_window"] = dist_stats(inside_counts)
    out["devices_ending_inside_window_per_scenario"] = inside_counts
    out["n_windows_extending_past_data_end"] = len(inside_counts) - len(fully_inside)
    out["devices_ending_inside_window_fully_inside_data"] = dist_stats(fully_inside)

    # How many recorded EOL crossings land inside each window?
    eol_dates = pd.to_datetime(eol.dropna().values)
    eol_inside = []
    for start, days in windows:
        stop = start + pd.Timedelta(days=days)
        eol_inside.append(int(((eol_dates >= start) & (eol_dates < stop)).sum()))
    out["eol_crossings_inside_window"] = dist_stats(eol_inside)

    global_end = end_times.max()
    at_global_end = (global_end - end_times) < pd.Timedelta(days=1)
    out["n_devices_ending_at_dataset_end"] = int(at_global_end.sum())
    out["n_devices_ending_before_dataset_end"] = int((~at_global_end).sum())

    has_eol = eol.notna()
    # Last smoothed voltage per device.
    last_smooth = {}
    for dev_id, s in cache.devices.items():
        finite = np.flatnonzero(np.isfinite(s.smooth_voltage))
        if finite.size:
            last_smooth[dev_id] = float(s.smooth_voltage[finite[-1]])
    ls = pd.Series(last_smooth)

    no_eol_ids = eol.index[~has_eol]
    no_eol_last_v = ls.reindex(no_eol_ids).dropna()
    out["no_eol_last_smooth_v"] = dist_stats(no_eol_last_v.values)
    out["no_eol_frac_last_v_above_2.4"] = float((no_eol_last_v >= EOL_THRESHOLD).mean())

    early = end_times[~at_global_end]
    early_no_eol = [d for d in early.index if d in set(no_eol_ids)]
    out["n_end_early_total"] = int(len(early))
    out["n_end_early_no_eol"] = int(len(early_no_eol))
    out["n_end_early_with_eol"] = int(len(early)) - int(len(early_no_eol))
    out["frac_end_early_without_eol"] = (
        len(early_no_eol) / len(early) if len(early) else float("nan")
    )
    early_no_eol_v = ls.reindex(early_no_eol).dropna()
    out["end_early_no_eol_last_smooth_v"] = dist_stats(early_no_eol_v.values)
    out["end_early_no_eol_frac_above_2.4"] = (
        float((early_no_eol_v >= EOL_THRESHOLD).mean()) if len(early_no_eol_v) else float("nan")
    )

    # For EOL devices: how long after the EOL date does observation continue?
    gaps = []
    for dev_id, c in crossings.items():
        gaps.append(
            float((end_times.loc[dev_id] - pd.Timestamp(c["csv_date"])) / pd.Timedelta(days=1))
        )
    out["obs_end_minus_eol_days"] = dist_stats(gaps)

    # Rate of unobserved-EOL removals per scenario window: devices that end
    # inside the window without an EOL recorded by the window end.
    eol_ord = {d: to_ordinal(t) for d, t in eol.dropna().items()}
    unobs_counts, unobs_fully_inside = [], []
    for start, days in windows:
        stop = start + pd.Timedelta(days=days)
        in_win = end_times[(end_times >= start) & (end_times < stop)]
        n_unobs = sum(
            1
            for d, t in in_win.items()
            if d not in eol_ord or from_ordinal(eol_ord[d]) > stop
        )
        unobs_counts.append(n_unobs)
        if stop <= data_end:
            unobs_fully_inside.append(n_unobs)
    out["unobserved_end_inside_window"] = dist_stats(unobs_counts)
    out["unobserved_end_inside_window_fully_inside_data"] = dist_stats(unobs_fully_inside)

    MD.append("## 8. Censoring geometry\n")
    w = out["devices_ending_inside_window"]
    MD.append(f"- {out['n_scenarios']} scenarios, planning windows all {out['planning_window_days']} days, "
              f"starts {out['scenario_start_span']['first'][:10]} to {out['scenario_start_span']['last'][:10]}, "
              f"unobserved_eol_days setting = {out['unobserved_eol_days_setting']}.")
    wf = out["devices_ending_inside_window_fully_inside_data"]
    MD.append(f"- Devices whose observation ends inside a 42 d window: median **{fmt(w['median'],1)}** per scenario "
              f"(min {fmt(w['min'],0)}, max {fmt(w['max'],0)}; the max comes from the "
              f"{out['n_windows_extending_past_data_end']} windows that extend past the dataset end and sweep up all "
              f"administratively-censored devices. Windows fully inside the data: median {fmt(wf.get('median'),1)}, "
              f"max {fmt(wf.get('max'),0)}).")
    e = out["eol_crossings_inside_window"]
    MD.append(f"- Recorded EOL crossings inside a window: median {fmt(e['median'],1)} per scenario "
              f"(min {fmt(e['min'],0)}, max {fmt(e['max'],0)}).")
    MD.append(f"- {out['n_devices_ending_at_dataset_end']} devices are observed to the dataset end; "
              f"**{out['n_devices_ending_before_dataset_end']}** end earlier. Of those early enders, "
              f"{out['n_end_early_no_eol']} have NO recorded EOL "
              f"({100*out['frac_end_early_without_eol']:.1f}%) and "
              f"{100*(out['end_early_no_eol_frac_above_2.4'] or 0):.1f}% of them were last seen above 2.4 V "
              f"(median last smooth_v {fmt(out['end_early_no_eol_last_smooth_v'].get('median'),3)} V).")
    MD.append(f"- Devices with no EOL overall: last smooth_v median {fmt(out['no_eol_last_smooth_v'].get('median'),3)} V; "
              f"{100*out['no_eol_frac_last_v_above_2.4']:.1f}% end above 2.4 V.")
    MD.append(f"- After the recorded EOL date, observation continues for median {fmt(out['obs_end_minus_eol_days'].get('median'),1)} d "
              f"(IQR {fmt(out['obs_end_minus_eol_days'].get('p25'),1)}-{fmt(out['obs_end_minus_eol_days'].get('p75'),1)}).")
    u = out["unobserved_end_inside_window"]
    uf = out["unobserved_end_inside_window_fully_inside_data"]
    MD.append(f"- Unobserved-EOL removals inside a window (end of observation with no EOL by window close): "
              f"median {fmt(u['median'],1)} per scenario (max {fmt(u['max'],0)}; windows fully inside the data: "
              f"median {fmt(uf.get('median'),1)}, max {fmt(uf.get('max'),0)}).")
    MD.append("")
    return out


# ------------------------------------------------------------------ main

def build_flags(r: dict) -> list[str]:
    """Loud statements where the measurements bear on the hypothesis."""
    lines = ["## Key flags (hypothesis check)\n"]
    m4 = r["4_monotonicity"]
    lines.append(
        f"1. **Monotone-state hypothesis only PARTIALLY supported.** Temperature compensation removes "
        f"some but NOT most of the upward movement: up/down mass ratio goes "
        f"{fmt(m4['a_raw']['up_down_mass_ratio'],3)} (raw) -> {fmt(m4['b2_global_comp_resmoothed']['up_down_mass_ratio'],3)} "
        f"(global comp) -> {fmt(m4['d2_per_device_detrended_comp_resmoothed']['up_down_mass_ratio'],3)} "
        f"(per-device comp, best variant). At a 28-day lag it is "
        f"{fmt(m4['lag28_a_raw']['up_down_mass_ratio'],3)} -> {fmt(m4['lag28_d2_per_device_detrended_comp_resmoothed']['up_down_mass_ratio'],3)}: "
        f"a quarter of the monthly-scale movement is still upward after removing the linear temperature term. "
        f"NOTE: the task-specified whole-life per-device beta (variant c) makes monotonicity WORSE "
        f"({fmt(m4['c_per_device_comp']['up_down_mass_ratio'],3)}) because the whole-life regression is "
        f"confounded by trend x season; the detrended beta is the physical one."
    )
    s5 = r["5_shared_curve"]
    lines.append(
        f"2. **Shared-curve x per-device-rate model CONTRADICTED.** Band first-passage times have CV "
        f"0.77-0.90 and boundary-free cross-band correlations of log times are ~0 "
        f"(t(2.8->2.7) vs t(2.6->2.5): r={fmt(s5['separated_corr_2.8-2.7_vs_2.6-2.5'].get('pearson'),2)}; "
        f"t(2.9->2.8) vs t(2.5->2.4): r={fmt(s5['separated_corr_2.9-2.8_vs_2.5-2.4'].get('pearson'),2)}). "
        f"A device slow in one band is NOT slow in the others. Yet total 2.8->2.4 time is far tighter "
        f"(CV {fmt(s5['total_2.8->2.4_days']['cv'],2)}) than any single band -- plateau dwell and knee speed "
        f"trade off rather than scale together."
    )
    m3 = r["3_warning_window"]
    lines.append(
        f"3. **Temperature compensation does NOT tighten the level-threshold warning window** "
        f"(IQR improves for only 2/5 global, 1/5 per-device thresholds). Warning-time spread is dominated "
        f"by knee-shape heterogeneity, not temperature: last-day-above-2.55 gives median "
        f"{fmt(m3['raw']['2.55']['median'],0)} d warning with IQR {fmt(m3['raw']['2.55']['iqr'],0)} d."
    )
    m6 = r["6_ir_lead_time"]
    m7 = r["7_temperature_structure"]
    lines.append(
        f"4. **The rising-dV/dT (internal-resistance) part of the hypothesis is CONFIRMED.** Within-day beta "
        f"roughly doubles from its pre-180 d baseline (median {fmt(m6['baseline_beta_pre180']['median'],5)}) "
        f"well before EOL -- median first 2x exceedance {fmt(m6['lead_first_exceed_2x_baseline_days']['median'],0)} d "
        f"before crossing ({m6['n_lead_censored_at_180']} of 80 censored at >=180 d), and AUC vs matched "
        f"surviving device-days at 42 d lead = {fmt(m6['auc_beta_42d_lead'],3)}. Final-90 d beta is "
        f"{fmt(m7['beta_final90_over_early_ratio']['median'],2)}x early-life beta for EOL devices "
        f"({100*m7['frac_final90_beta_gt_early']:.0f}% increase)."
    )
    m2 = r["2_trajectory_shape"]
    lines.append(
        f"5. **Plateau-then-knee shape CONFIRMED**: smoothed EOL trajectories fall only "
        f"~{1000*(m2['smooth_v_before_crossing']['minus_180d']['median']-m2['smooth_v_before_crossing']['minus_42d']['median']):.0f} mV "
        f"over days -180..-42 but ~{1000*(m2['smooth_v_before_crossing']['minus_42d']['median']-2.4):.0f} mV over the last 42 d; "
        f"final-14 d slope median {fmt(m2['post_knee_slope_final14d_V_per_day']['median'],4)} V/d. "
        f"But the knee is NOT sharp at 42 d: only {100*m2['frac_above_2.55_at_42d_out']['frac']:.0f}% are still above 2.55 V then."
    )
    lines.append("")
    return lines


def main() -> None:
    t0 = time.time()
    log("loading data ...")
    bm = common.load_metrics()
    devices = common.load_devices()
    eol = common.load_eol()
    scenarios = common.load_scenarios()

    log("building smoothing cache ...")
    cache = common.build_smoothing(bm)
    log(f"  {len(cache.devices)} devices smoothed ({time.time()-t0:.1f}s)")
    log("building shape cache ...")
    shape_cache = common.build_shape(bm)
    log(f"  {len(shape_cache.devices)} devices shaped ({time.time()-t0:.1f}s)")

    MD.append("# Data-physics verification report")
    MD.append("")
    MD.append(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from dataset/train. Measurement only; no model fitted._")
    MD.append(f"_Smoothing: exact reimplementation of official smooth_series (7-day trailing median of daily medians, "
              f"10<T<30 filter, >=5 readings/day, min_periods=int(0.5*7)=3 as pinned by tests/test_smoothing.py)._")
    MD.append("")

    crossings = get_crossings(cache, eol)

    log("section 1 ...")
    RESULT["1_inventory"] = section1(bm, devices, eol, cache)
    log("section 2 ...")
    RESULT["2_trajectory_shape"] = section2(cache, crossings)
    log("section 3 ...")
    RESULT["3_warning_window"] = section3(cache, crossings)
    log("section 4 ...")
    RESULT["4_monotonicity"] = section4(cache)
    log("section 5 ...")
    RESULT["5_shared_curve"] = section5(cache, crossings)
    log("section 6 ...")
    RESULT["6_ir_lead_time"] = section6(cache, shape_cache, crossings, devices, eol)
    log("section 7 ...")
    RESULT["7_temperature_structure"] = section7(cache, shape_cache, devices, crossings)
    log("section 8 ...")
    RESULT["8_censoring"] = section8(cache, devices, eol, scenarios, crossings)

    OUTPUTS.mkdir(exist_ok=True)
    json_path = OUTPUTS / "diag_physics.json"
    md_path = OUTPUTS / "diag_physics.md"
    import json as _json

    # Insert the hypothesis-check flags right after the report header.
    md_full = MD[:5] + build_flags(RESULT) + MD[5:]

    json_path.write_text(_json.dumps(jsonable(RESULT), indent=2))
    md_path.write_text("\n".join(md_full), encoding="utf-8")
    log(f"wrote {json_path}")
    log(f"wrote {md_path}")
    log(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
