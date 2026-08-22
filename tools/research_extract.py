"""RESEARCH extraction: one pass over the hourly parquet -> per-row features.

Produces, for every row of outputs/frame_oof_raw_beta.parquet (scenario x alive
battery), three families of strictly-causal features (computed at cutoff-1):

1. raw-daily channel (bsai.rawdaily.RawDailyCache): raw_last/min3/min7/slope7,
   days-below counts -- the sub-smoothing-lag information.
2. within-day hourly shape beyond beta: pulse depth (p50-p05), upper room
   (p95-p50), night-vs-day median gap, deep-reading fraction, mean-median skew;
   each as a trailing-14d mean plus a rise ratio against the device's own
   prefix median (building-invariant form).
3. last-90-day smoothed-margin trajectory (for the kNN shape probe), saved to
   outputs/research_traj.npz.

Output: outputs/research_rowfeat.parquet, outputs/research_traj.npz.
Runtime: a few minutes (single hourly-parquet load, vectorized groupbys).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsai.rawdaily import RawDailyCache
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")


def main() -> None:
    frame = pd.read_parquet(ROOT / "outputs" / "frame_oof_raw_beta.parquet")
    cal = pd.read_parquet(ROOT / "outputs" / "frame_oof_cal.parquet")
    frame = frame.merge(
        cal[["scenario", "battery", "p"]].rename(columns={"p": "p_cal"}),
        on=["scenario", "battery"],
        how="left",
    )
    import json

    scenarios = json.load(open(ROOT / "dataset" / "train" / "scenarios.json"))
    starts = pd.to_datetime([s["start_time"] for s in scenarios]).normalize()
    start_ord = ((starts - _EPOCH) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)
    frame["cutoff_ord"] = frame["scenario"].map(dict(enumerate(start_ord)))

    print("loading hourly parquet ...", flush=True)
    raw = pd.read_parquet(
        ROOT / "dataset" / "train" / "battery_metrics.parquet", engine="fastparquet"
    )
    print(f"  {len(raw)} rows", flush=True)

    # ---------------------------------------------------------------- caches
    print("building SmoothingCache + RawDailyCache ...", flush=True)
    smooth = SmoothingCache()
    smooth.update(raw)
    rawdaily = RawDailyCache()
    rawdaily.update(raw)

    # ------------------------------------------------- hourly per-day stats
    print("hourly within-day stats ...", flush=True)
    work = pd.DataFrame(
        {
            "device_id": raw["device_id"].astype(str),
            "end_time": pd.to_datetime(raw["end_time"]),
            "voltage": raw["voltage"].astype(np.float64),
        }
    )
    del raw
    work["day"] = ((work["end_time"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)).astype(
        np.int64
    )
    work["hour"] = work["end_time"].dt.hour

    g = work.groupby(["device_id", "day"], sort=True, observed=True)["voltage"]
    stats = g.agg(n="size", vmean="mean", p50="median")
    q = g.quantile([0.05, 0.95]).unstack()
    stats["p05"] = q[0.05]
    stats["p95"] = q[0.95]

    night = (
        work[work["hour"] < 6]
        .groupby(["device_id", "day"], observed=True)["voltage"]
        .median()
        .rename("night_med")
    )
    dayv = (
        work[(work["hour"] >= 12) & (work["hour"] < 18)]
        .groupby(["device_id", "day"], observed=True)["voltage"]
        .median()
        .rename("day_med")
    )
    stats = stats.join(night).join(dayv)

    # deep-reading fraction: share of readings 25mV below the day median
    p50_map = stats["p50"]
    work = work.join(p50_map.rename("day_p50"), on=["device_id", "day"])
    work["deep"] = (work["voltage"] < work["day_p50"] - 0.025).astype(np.float64)
    deep = (
        work.groupby(["device_id", "day"], observed=True)["deep"].mean().rename("deep_frac")
    )
    stats = stats.join(deep)
    del work

    stats["depth"] = stats["p50"] - stats["p05"]
    stats["room"] = stats["p95"] - stats["p50"]
    stats["nightday"] = stats["night_med"] - stats["day_med"]
    stats["skew"] = stats["vmean"] - stats["p50"]
    # thin days are unreliable
    usable = stats["n"] >= 6
    for c in ("depth", "room", "nightday", "skew", "deep_frac"):
        stats.loc[~usable, c] = np.nan

    stat_cols = ["depth", "room", "nightday", "skew", "deep_frac"]
    per_device: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device_id, block in stats.groupby(level=0, sort=False, observed=True):
        days = block.index.get_level_values(1).to_numpy(dtype=np.int64)
        per_device[str(device_id)] = (days, block[stat_cols].to_numpy(dtype=np.float64))

    def trailing(device: str, cutoff: int, window: int) -> np.ndarray:
        entry = per_device.get(device)
        if entry is None:
            return np.full(len(stat_cols), np.nan)
        days, mat = entry
        hi = np.searchsorted(days, cutoff, side="right")  # days <= cutoff-1 -> use cutoff-1
        lo = np.searchsorted(days, cutoff - window, side="left")
        if hi <= lo:
            return np.full(len(stat_cols), np.nan)
        return np.nanmean(mat[lo:hi], axis=0)

    def prefix_median(device: str, cutoff: int, gap: int = 90) -> np.ndarray:
        entry = per_device.get(device)
        if entry is None:
            return np.full(len(stat_cols), np.nan)
        days, mat = entry
        hi = np.searchsorted(days, cutoff - gap, side="left")
        if hi < 30:
            return np.full(len(stat_cols), np.nan)
        return np.nanmedian(mat[:hi], axis=0)

    # ------------------------------------------------------------- per-row loop
    print("per-row features ...", flush=True)
    n = len(frame)
    raw_feats = np.full((n, 7), np.nan)
    hour_feats = np.full((n, len(stat_cols)), np.nan)
    rise_feats = np.full((n, len(stat_cols)), np.nan)
    traj = np.full((n, 90), np.nan, dtype=np.float32)
    traj_valid = np.zeros(n, dtype=np.int16)

    batteries = frame["battery"].to_numpy()
    cutoffs = frame["cutoff_ord"].to_numpy()
    for i in range(n):
        b, c = batteries[i], int(cutoffs[i])
        raw_feats[i] = rawdaily.features_at(b, c - 1)
        t14 = trailing(b, c - 1, 14)
        hour_feats[i] = t14
        base = prefix_median(b, c - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rise_feats[i] = np.where(np.abs(base) > 1e-6, t14 / base, np.nan)
        series = smooth.devices.get(b)
        if series is not None:
            j = c - 1 - series.origin
            if j >= 0:
                j = min(j, len(series) - 1)
                seg = series.smooth_voltage[max(0, j - 89) : j + 1] - 2.4
                traj[i, 90 - len(seg) :] = seg
                traj_valid[i] = int(np.isfinite(seg).sum())

    from bsai.rawdaily import RAW_FEATURE_NAMES

    for k, name in enumerate(RAW_FEATURE_NAMES):
        frame[name] = raw_feats[:, k]
    for k, name in enumerate(stat_cols):
        frame[f"h14_{name}"] = hour_feats[:, k]
        frame[f"rise_{name}"] = rise_feats[:, k]

    out = ROOT / "outputs" / "research_rowfeat.parquet"
    frame.to_parquet(out)
    np.savez_compressed(
        ROOT / "outputs" / "research_traj.npz", traj=traj, valid=traj_valid
    )
    print(f"wrote {out} ({len(frame)} rows) and research_traj.npz")


if __name__ == "__main__":
    main()
