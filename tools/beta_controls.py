"""Two controls on the beta-as-state result, before anything is built on it.

``tools/beta_state.py`` reports that the within-day slope rises with a median
Spearman of 0.811 on the 82 devices that crossed, and that its value at crossing
sits inside a factor of 2.2 from the 10th to the 90th percentile. Both readings
pass the stated test for a state variable. Both can also be produced by
something other than a state variable, and each has a cheap control.

**Control 1 -- is the threshold vacuous?**
End of life is *defined* as the smoothed voltage falling below 2.4 V. If beta is
largely a function of that same voltage, then its value at crossing is tight by
construction -- it is just beta evaluated at margin zero -- and would be equally
tight evaluated at margin 0.3 V, where nothing is failing. So: take each crossed
device's beta at the first day its margin drops below each of several levels and
compare the spreads. A real barrier tightens as the margin closes. A re-encoding
of voltage does not.

**Control 2 -- is the rise seasonal rather than degradation?**
beta is a within-day regression of voltage on temperature, and the daily
temperature swing itself has a season. The 379 surviving devices also rise, at a
median Spearman of 0.628, and they are not dying. If the fleet rises together,
per-device monotonicity is calendar, not state. Subtracting the fleet's monthly
mean removes the shared ride and leaves whatever is specific to the device.

    python tools/beta_controls.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices

from bsai.margin import EOL_THRESHOLD
from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
MARGIN_LEVELS = (0.40, 0.30, 0.20, 0.10, 0.05, 0.02)


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    present = np.isfinite(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(present, values, 0.0))])
    counts = np.concatenate([[0], np.cumsum(present)])
    out = np.full(values.size, np.nan)
    low = np.maximum(np.arange(values.size) - window + 1, 0)
    high = np.arange(values.size) + 1
    count = counts[high] - counts[low]
    total = sums[high] - sums[low]
    np.divide(total, count, out=out, where=count > 0)
    return out


def _spread(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return {"n": int(values.size)}
    p10, p50, p90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "n": int(values.size),
        "median": round(float(p50), 6),
        "p10": round(float(p10), 6),
        "p90": round(float(p90), 6),
        "p90_over_p10": round(float(p90 / p10), 3) if p10 > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_beta_controls.json"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    del devices

    print("reading and shaping...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    smoothing = SmoothingCache()
    smoothing.update(raw)
    shape = ShapeCache()
    shape.update(raw)
    del raw

    # One tidy table of (device, absolute day, beta, margin, t_range).
    records: list[pd.DataFrame] = []
    for device_id, device_shape in shape.devices.items():
        smoothed = _trailing_mean(device_shape.beta, args.window)
        days = np.arange(device_shape.origin, device_shape.origin + len(device_shape))
        series = smoothing.devices.get(device_id)
        margin = np.full(days.size, np.nan)
        if series is not None:
            offset = days - series.origin
            inside = (offset >= 0) & (offset < len(series))
            margin[inside] = series.smooth_voltage[offset[inside]] - EOL_THRESHOLD
        records.append(
            pd.DataFrame(
                {
                    "device": device_id,
                    "day": days,
                    "beta": smoothed,
                    "margin": margin,
                    "t_range": _trailing_mean(device_shape.t_range, args.window),
                }
            )
        )
    frame = pd.concat(records, ignore_index=True).dropna(subset=["beta"])
    frame["month"] = (
        _EPOCH + pd.to_timedelta(frame["day"].to_numpy(), unit="D")
    ).month
    frame["crossed"] = frame["device"].map(lambda d: not pd.isna(eol.get(d)))
    print(f"  {len(frame)} device-days, {frame.device.nunique()} devices", flush=True)

    # ---- Control 1: beta at fixed margins ---------------------------------
    crossed_ids = sorted({d for d in frame.device.unique() if not pd.isna(eol.get(d))})
    at_margin: dict[str, list[float]] = {f"{m:.2f}": [] for m in MARGIN_LEVELS}
    at_margin["crossing"] = []
    for device_id, block in frame[frame.crossed].groupby("device", sort=False):
        block = block.sort_values("day")
        margin = block["margin"].to_numpy()
        beta = block["beta"].to_numpy()
        day = block["day"].to_numpy()
        for level in MARGIN_LEVELS:
            hit = np.flatnonzero(np.isfinite(margin) & (margin <= level))
            if hit.size:
                at_margin[f"{level:.2f}"].append(float(beta[hit[0]]))
        moment = eol.get(device_id)
        if not pd.isna(moment):
            target = _ordinal(moment)
            index = int(np.argmin(np.abs(day - target)))
            if abs(int(day[index]) - target) <= 7:
                at_margin["crossing"].append(float(beta[index]))

    control_one = {key: _spread(np.asarray(values)) for key, values in at_margin.items()}

    # ---- Control 2: is the rise the fleet's, or the device's? -------------
    # The fleet's monthly mean is built from surviving devices only, so a
    # dying device is never compared against a baseline it helped set.
    survivors = frame[~frame.crossed]
    monthly = survivors.groupby("month")["beta"].mean()
    monthly_t_range = survivors.groupby("month")["t_range"].mean()
    frame["beta_adjusted"] = frame["beta"] - frame["month"].map(monthly).to_numpy()

    def per_device_rho(block: pd.DataFrame, column: str) -> float:
        if len(block) < 30:
            return float("nan")
        return float(spearmanr(block["day"], block[column]).statistic)

    rows = []
    for device_id, block in frame.groupby("device", sort=False):
        rows.append(
            {
                "device": device_id,
                "crossed": bool(not pd.isna(eol.get(device_id))),
                "rho_raw": per_device_rho(block, "beta"),
                "rho_adjusted": per_device_rho(block, "beta_adjusted"),
                "n": len(block),
            }
        )
    rho_frame = pd.DataFrame(rows).dropna(subset=["rho_raw"])
    control_two = {
        "fleet_monthly_mean_beta": {
            int(month): round(float(value), 6) for month, value in monthly.items()
        },
        "fleet_monthly_mean_t_range": {
            int(month): round(float(value), 3)
            for month, value in monthly_t_range.items()
        },
        "fleet_month_max_over_min": round(
            float(monthly.max() / monthly.min()), 3
        ),
        "crossed_rho_raw": _spread(rho_frame[rho_frame.crossed].rho_raw.to_numpy()),
        "crossed_rho_adjusted": _spread(
            rho_frame[rho_frame.crossed].rho_adjusted.to_numpy()
        ),
        "survivor_rho_raw": _spread(rho_frame[~rho_frame.crossed].rho_raw.to_numpy()),
        "survivor_rho_adjusted": _spread(
            rho_frame[~rho_frame.crossed].rho_adjusted.to_numpy()
        ),
    }

    # ---- How much of beta is simply voltage? ------------------------------
    usable = frame.dropna(subset=["margin"])
    pooled = float(spearmanr(usable["margin"], usable["beta"]).statistic)
    within = []
    for _, block in usable.groupby("device", sort=False):
        if len(block) >= 30:
            within.append(float(spearmanr(block["margin"], block["beta"]).statistic))
    redundancy = {
        "pooled_spearman_beta_vs_margin": round(pooled, 4),
        "within_device_spearman_beta_vs_margin": _spread(np.asarray(within)),
        "n_device_days": int(len(usable)),
    }

    summary = {
        "window_days": args.window,
        "n_crossed": len(crossed_ids),
        "control_1_beta_at_fixed_margin": control_one,
        "control_2_seasonal": control_two,
        "redundancy_with_voltage": redundancy,
    }
    print()
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
