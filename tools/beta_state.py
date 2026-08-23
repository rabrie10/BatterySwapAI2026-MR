"""Is the within-day voltage/temperature slope a state variable or a correlate?

``bsai/shape.py`` computes ``beta``, the within-day regression of voltage on
temperature. It separates due from not-due at AUC 0.871 on exactly the
population level-and-slope cannot rank, and the physical story is attractive: a
CR2477T is a primary Li/MnO2 cell, it consumes its cathode once and never
recovers, so internal resistance rises monotonically and ``beta`` is a direct
observation of it. If that holds, the Wiener process -- chosen because the
*margin* is non-monotonic -- was fitted to the wrong quantity, and the monotone
degradation processes that were ruled out on that basis come back.

A state variable has two properties a correlate does not:

1. its per-device trajectory rises monotonically, and
2. it reaches a consistent value at failure across devices.

This measures both on the 82 devices that reached EOL, and adds the control the
first two cannot supply on their own: how often a *surviving* device's ``beta``
sits above the failure level. A threshold that healthy devices cross all the
time is not a threshold.

    python tools/beta_state.py
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

from bsai.shape import ShapeCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` grid days, skipping missing days.

    Matches ``ShapeView.trailing_mean``: absent days are skipped rather than
    imputed, so an intermittent reporter still gets a mean over what it filed.
    """
    present = np.isfinite(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(present, values, 0.0))])
    counts = np.concatenate([[0], np.cumsum(present)])
    out = np.full(values.size, np.nan)
    for index in range(values.size):
        low = max(0, index - window + 1)
        count = counts[index + 1] - counts[low]
        if count > 0:
            out[index] = (sums[index + 1] - sums[low]) / count
    return out


def _describe(values: np.ndarray, name: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"name": name, "n": 0}
    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "name": name,
        "n": int(values.size),
        "min": round(float(values.min()), 6),
        "p10": round(float(quantiles[0]), 6),
        "p25": round(float(quantiles[1]), 6),
        "median": round(float(quantiles[2]), 6),
        "p75": round(float(quantiles[3]), 6),
        "p90": round(float(quantiles[4]), 6),
        "max": round(float(values.max()), 6),
        "p90_over_p10": round(float(quantiles[4] / quantiles[0]), 3)
        if quantiles[0] > 0
        else None,
        "iqr_over_median": round(
            float((quantiles[3] - quantiles[1]) / quantiles[2]), 3
        )
        if quantiles[2] != 0
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--window", type=int, default=14, help="trailing smoothing days")
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_beta_state.json"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    observation_end = devices.set_index("device_id")["end_time"]
    eol = pd.to_datetime(
        pd.read_csv(args.dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )

    print("reading and shaping...", flush=True)
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    smoothing = SmoothingCache()
    smoothing.update(raw)
    shape = ShapeCache()
    shape.update(raw)
    del raw
    print(f"  {len(shape.devices)} devices shaped", flush=True)

    crossed: list[dict] = []
    survivor_rows: list[dict] = []
    survivor_values: list[np.ndarray] = []
    for device_id, device_shape in shape.devices.items():
        smoothed = _trailing_mean(device_shape.beta, args.window)
        finite = np.flatnonzero(np.isfinite(smoothed))
        if finite.size < 30:
            continue
        first, last = int(finite[0]), int(finite[-1])
        days = np.arange(device_shape.origin, device_shape.origin + len(device_shape))
        moment = eol.get(device_id)

        # Rank correlation against time is the honest reading of "monotone":
        # the daily estimate is noisy, so demanding every increment be positive
        # would fail on measurement noise alone.
        values = smoothed[finite]
        rho = float(spearmanr(days[finite], values).statistic)
        increments = np.diff(values)
        rising = float((increments > 0).mean()) if increments.size else float("nan")

        if pd.isna(moment):
            end = observation_end.get(device_id)
            survivor_rows.append(
                {
                    "device": device_id,
                    "rho": rho,
                    "fraction_rising": rising,
                    "beta_final": float(smoothed[last]),
                    "beta_max": float(np.nanmax(values)),
                    "beta_median": float(np.nanmedian(values)),
                    "n_days": int(finite.size),
                    "observed_to": None if pd.isna(end) else str(pd.Timestamp(end).date()),
                }
            )
            survivor_values.append(values)
            continue

        crossing = _ordinal(moment) - device_shape.origin
        if crossing < first or crossing > last + 7:
            continue
        at_index = min(max(crossing, first), last)
        life = at_index - first
        if life < 30:
            continue

        def value_at(fraction: float) -> float:
            target = first + int(round(fraction * life))
            target = min(max(target, first), last)
            return float(smoothed[target])

        crossed.append(
            {
                "device": device_id,
                "beta_at_crossing": float(smoothed[at_index]),
                "beta_at_50pct": value_at(0.5),
                "beta_at_80pct": value_at(0.8),
                "beta_baseline": float(np.nanmedian(smoothed[first : first + 30])),
                "rho": rho,
                "fraction_rising": rising,
                "life_days": int(life),
                "n_days": int(finite.size),
            }
        )

    crossed_frame = pd.DataFrame(crossed)
    survivor_frame = pd.DataFrame(survivor_rows)
    print(f"  {len(crossed_frame)} devices with a usable crossing, "
          f"{len(survivor_frame)} survivors", flush=True)

    ratio = crossed_frame["beta_at_crossing"] / crossed_frame["beta_baseline"]
    threshold = float(crossed_frame["beta_at_crossing"].median())
    p10_threshold = float(crossed_frame["beta_at_crossing"].quantile(0.1))

    # The control. If a survivor's beta routinely climbs past the level a dying
    # device reaches, beta is not marking a state, it is marking a population.
    above_median = float((survivor_frame["beta_max"] >= threshold).mean())
    above_p10 = float((survivor_frame["beta_max"] >= p10_threshold).mean())
    final_above_median = float((survivor_frame["beta_final"] >= threshold).mean())
    # A maximum over a long record is biased upward simply by having more days
    # to draw from, so also report the rate: what share of all survivor
    # device-days sits above the level a dying device reaches.
    pooled = np.concatenate(survivor_values) if survivor_values else np.array([])
    day_fraction = (
        float((pooled >= threshold).mean()) if pooled.size else float("nan")
    )

    summary = {
        "window_days": args.window,
        "n_crossed": int(len(crossed_frame)),
        "n_survivors": int(len(survivor_frame)),
        "monotonicity": {
            "crossed_spearman": _describe(crossed_frame["rho"], "rho"),
            "crossed_fraction_rising": _describe(
                crossed_frame["fraction_rising"], "fraction_rising"
            ),
            "crossed_rho_above_0.5": round(
                float((crossed_frame["rho"] > 0.5).mean()), 3
            ),
            "crossed_rho_above_0.8": round(
                float((crossed_frame["rho"] > 0.8).mean()), 3
            ),
            "crossed_rho_negative": round(float((crossed_frame["rho"] < 0).mean()), 3),
            "survivor_spearman": _describe(survivor_frame["rho"], "rho"),
        },
        "threshold": {
            "beta_at_crossing": _describe(
                crossed_frame["beta_at_crossing"], "beta_at_crossing"
            ),
            "beta_at_80pct_life": _describe(
                crossed_frame["beta_at_80pct"], "beta_at_80pct"
            ),
            "beta_at_50pct_life": _describe(
                crossed_frame["beta_at_50pct"], "beta_at_50pct"
            ),
            "crossing_over_baseline": _describe(ratio, "crossing_over_baseline"),
        },
        "control": {
            "survivor_beta_max": _describe(survivor_frame["beta_max"], "beta_max"),
            "survivor_max_above_crossing_median": round(above_median, 3),
            "survivor_max_above_crossing_p10": round(above_p10, 3),
            "survivor_final_above_crossing_median": round(final_above_median, 3),
            "survivor_day_fraction_above_crossing_median": round(day_fraction, 4),
            "survivor_device_days": int(pooled.size),
        },
    }
    print()
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "summary": summary,
                "crossed": crossed_frame.round(6).to_dict("records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
