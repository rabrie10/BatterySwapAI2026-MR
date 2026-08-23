"""Are the Wiener assumptions right near the threshold?

Not "does another feature predict EOL" but "is the stochastic process used to
turn a state into a crossing probability the right one". Four diagnostics, no
model fitted, all on the smoothed daily series V8 actually operates on.

**The null has to be simulated, not looked up.** `smooth_series` is a seven-day
rolling median, so consecutive daily values share six of seven inputs. That alone
induces increment autocorrelation and bends variance scaling away from the
textbook straight line. Comparing the data against `Var = h * sigma^2` would
therefore find "non-Brownian" behaviour in a process that is exactly Brownian.
So every diagnostic here is run twice: once on the data and once on synthetic
Brownian paths pushed through the same smoother, matched in length and in
missing-data pattern. The test is data against that null.

**Censoring has to be handled or it fabricates mean reversion.** Windows whose
*end* is required to precede the crossing keep only the near-threshold devices
that did not cross, which is precisely the population that looks like it has a
floor. Three window populations are reported side by side:

    v8_convention   both ends strictly before the crossing (what V8 trains on)
    censor_safe     start before the crossing, end anywhere observed (V10's fix)
    never_cross     devices with no crossing at all

If a floor appears only in the first, it is survivor conditioning, not physics.

    python tools/fj_process.py --report outputs/fj_process.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fj_terminality import load_series  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
EOL_THRESHOLD = 2.4
HORIZONS = (1, 2, 3, 7, 14, 21, 28, 42)
BANDS = ((0.0, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9.0))
WINDOW = 7
MIN_PERIODS = 3


def crossing_index(series: dict, dataset: Path) -> dict[str, int]:
    table = pd.read_csv(dataset / "eol_times.csv")
    recorded = table[table["end_time"].notna()]
    stamps = {
        str(d): pd.Timestamp(t).normalize()
        for d, t in zip(recorded["device_id"], recorded["end_time"])
    }
    out: dict[str, int] = {}
    for device, (_v, _t, origin) in series.items():
        moment = stamps.get(str(device))
        if moment is not None:
            out[str(device)] = int((moment - _EPOCH) / pd.Timedelta(days=1)) - origin
    return out


def rolling_median(values: np.ndarray) -> np.ndarray:
    """The shipped smoother's second stage, for the synthetic null."""
    return (
        pd.Series(values)
        .rolling(window=WINDOW, min_periods=MIN_PERIODS)
        .median()
        .to_numpy()
    )


def brownian_null(series: dict, crossing: dict, rng: np.random.Generator) -> dict:
    """One Brownian device per real device: same length, same missing pattern.

    The per-day increment scale is taken from the real device's own smoothed
    series so the null is matched in magnitude, and the path is then pushed
    through the same seven-day rolling median.
    """
    out: dict = {}
    for device, (voltage, temperature, origin) in series.items():
        valid = ~np.isnan(voltage)
        if valid.sum() < 60:
            continue
        steps = np.diff(voltage[valid])
        scale = float(np.nanstd(steps))
        if not np.isfinite(scale) or scale <= 0:
            continue
        # A rolling median of a random walk has roughly the increment scale of
        # the walk itself at long lag; start from the real daily scale.
        raw = np.full(voltage.size, np.nan)
        walk = np.cumsum(rng.normal(0.0, scale * np.sqrt(WINDOW), valid.sum()))
        raw[valid] = float(np.nanmean(voltage)) + walk
        out[device] = (rolling_median(raw), temperature, origin)
    return out


def window_stats(
    series: dict, crossing: dict, population: str
) -> dict[tuple[float, float], dict[int, list[float]]]:
    """Increments by margin band and horizon, under one window convention."""
    collected = {band: {h: [] for h in HORIZONS} for band in BANDS}
    owners = {band: {h: set() for h in HORIZONS} for band in BANDS}
    for device, (voltage, _t, _origin) in series.items():
        valid = np.flatnonzero(~np.isnan(voltage))
        if valid.size < 60:
            continue
        last = int(valid[-1])
        cross = crossing.get(device, -1)
        if population == "never_cross" and cross >= 0:
            continue
        margin = voltage - EOL_THRESHOLD
        for h in HORIZONS:
            starts = valid[valid + h <= last]
            if cross >= 0:
                starts = starts[starts < cross]
                if population == "v8_convention":
                    starts = starts[starts + h < cross]
            if starts.size == 0:
                continue
            here = margin[starts]
            there = margin[starts + h]
            good = np.isfinite(here) & np.isfinite(there)
            here, there = here[good], there[good]
            delta = here - there
            for band in BANDS:
                inside = (here >= band[0]) & (here < band[1])
                if inside.any():
                    collected[band][h].extend(delta[inside].tolist())
                    owners[band][h].add(device)
    return collected, owners


def variance_table(pair) -> dict:
    collected, owners = pair
    out = {}
    for band, horizons in collected.items():
        row = {}
        for h, values in horizons.items():
            array = np.asarray(values)
            if array.size < 50:
                row[h] = None
                continue
            row[h] = {
                "n": int(array.size),
                "devices": len(owners[band][h]),
                "var_over_h": float(array.var() / h),
                "sd": float(array.std()),
                "mean_drop": float(array.mean()),
            }
        out[f"{band[0]}-{band[1]}"] = row
    return out


def autocorrelation(series: dict, crossing: dict, lags=range(1, 15)) -> dict:
    """ACF of daily smoothed increments, by state and by fate.

    Segments are sampled every 30 days across each device's whole pre-crossing
    life, not once at the start, and averaged within a device before being
    averaged across devices -- otherwise a handful of long-lived devices
    dominate and every segment lands in the healthy bucket.
    """
    per_device: dict[str, dict[str, list]] = {}
    for device, (voltage, _t, _origin) in series.items():
        valid = np.flatnonzero(~np.isnan(voltage))
        if valid.size < 120:
            continue
        cross = crossing.get(device, -1)
        margin = voltage - EOL_THRESHOLD
        limit = cross if cross >= 0 else voltage.size
        # Segments may run past the crossing: the series keeps declining there
        # (median -0.033 V at 42 days, no replacement), and requiring 45 days
        # of pre-crossing room would structurally exclude every imminent case.
        stop_at = int(valid[-1])
        for start in range(int(valid[0]), limit, 15):
            here = margin[start]
            if not np.isfinite(here):
                continue
            imminent = cross >= 0 and 0 < cross - start <= 42
            if here < 0.10:
                key = "near_threshold_imminent" if imminent else "near_threshold_survivor"
            elif here < 0.20:
                key = "mid_margin"
            else:
                key = "healthy"
            segment = margin[start : min(start + 45, stop_at)]
            if segment.size < 40 or np.isnan(segment).mean() > 0.2:
                continue
            steps = np.diff(pd.Series(segment).interpolate().to_numpy())
            steps = steps - steps.mean()
            denominator = float((steps**2).sum())
            if denominator <= 1e-18:
                continue
            row = {
                lag: float((steps[lag:] * steps[:-lag]).sum() / denominator)
                for lag in lags if steps.size > lag + 5
            }
            per_device.setdefault(device, {}).setdefault(key, []).append(row)

    out: dict = {}
    for key in ("near_threshold_survivor", "near_threshold_imminent",
                "mid_margin", "healthy"):
        device_means = []
        for blocks in per_device.values():
            rows = blocks.get(key)
            if not rows:
                continue
            device_means.append({
                lag: float(np.mean([r[lag] for r in rows if lag in r]))
                for lag in lags
                if any(lag in r for r in rows)
            })
        if not device_means:
            continue
        out[key] = {
            "devices": len(device_means),
            "segments": sum(
                len(b.get(key, [])) for b in per_device.values()
            ),
            "acf": {
                lag: round(float(np.mean([d[lag] for d in device_means if lag in d])), 4)
                for lag in lags
                if any(lag in d for d in device_means)
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_process.json"))
    args = parser.parse_args()

    started = time.time()
    series = load_series(args.series)
    crossing = crossing_index(series, args.dataset)
    rng = np.random.default_rng(20260823)
    null = brownian_null(series, crossing, rng)
    print(f"{len(series)} devices, {len(crossing)} crossings, "
          f"{len(null)} synthetic Brownian twins ({time.time() - started:.0f}s)")

    report: dict = {"variance": {}, "autocorrelation": {}}
    for population in ("v8_convention", "censor_safe", "never_cross"):
        report["variance"][population] = variance_table(
            window_stats(series, crossing, population)
        )
    report["variance"]["brownian_null"] = variance_table(
        window_stats(null, crossing, "censor_safe")
    )
    print(f"variance tables built ({time.time() - started:.0f}s)")

    print()
    print("=== 1. Var(dV_h)/h, x1e6.  Brownian => flat across h ===")
    for population in ("censor_safe", "v8_convention", "brownian_null"):
        print(f"-- {population}")
        print(f"{'margin band':>14} " + " ".join(f"{'h=' + str(h):>9}" for h in HORIZONS))
        for band, row in report["variance"][population].items():
            cells = []
            for h in HORIZONS:
                cell = row.get(h)
                cells.append(f"{cell['var_over_h'] * 1e6:9.2f}" if cell else f"{'--':>9}")
            print(f"{band:>14} " + " ".join(cells))
        print(f"{'n at h=42':>14} " + " ".join(
            f"{row[42]['n']:9d}" if row.get(42) else f"{'--':>9}"
            for row in [report["variance"][population][b]]
        ) if False else "", end="")
        for band, row in report["variance"][population].items():
            cell = row.get(42)
            print(f"{'  n(h=42) ' + band:>26} {cell['n'] if cell else 0:>9d}")
        print()

    report["autocorrelation"]["data"] = autocorrelation(series, crossing)
    report["autocorrelation"]["brownian_null"] = autocorrelation(null, crossing)
    print("=== 2. ACF of daily smoothed increments ===")
    for label, block in report["autocorrelation"].items():
        print(f"-- {label}")
        for key, entry in block.items():
            acf = entry["acf"]
            row = " ".join(f"{acf.get(lag, float('nan')):+6.3f}" for lag in range(1, 15))
            print(f"{key:>26} n={entry['devices']:3d}  {row}")
        print()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {args.report} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
