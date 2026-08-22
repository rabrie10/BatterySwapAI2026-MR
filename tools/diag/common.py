"""Shared loading + helpers for the data-physics diagnostics (tools/diag).

Everything here is measurement-only. Single-threaded by construction: BLAS/OMP
thread caps are set before numpy is imported, because a heavy training job is
sharing this machine.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bsai.shape import ShapeCache, align_to  # noqa: E402
from bsai.smoothing import _EPOCH, SmoothingCache, _rolling_median  # noqa: E402

DATA = REPO / "dataset" / "train"
OUTPUTS = REPO / "outputs"

# Global temperature-compensation used throughout the diagnostics: the task's
# stated fleet coefficient, referenced to 20 degC.
GLOBAL_BETA = 0.00463
REF_TEMP = 20.0
EOL_THRESHOLD = 2.4


# ---------------------------------------------------------------- loading

def load_devices() -> pd.DataFrame:
    dev = pd.read_csv(DATA / "devices.csv", index_col=0)
    for col in ("start_time", "end_time"):
        dev[col] = pd.to_datetime(dev[col], utc=True, format="ISO8601").dt.tz_localize(None)
    return dev


def load_eol() -> pd.Series:
    eol = pd.read_csv(DATA / "eol_times.csv")
    return pd.Series(
        pd.to_datetime(eol["end_time"]).values, index=eol["device_id"], name="eol_time"
    )


def load_scenarios() -> list[dict]:
    with open(DATA / "scenarios.json") as fh:
        return json.load(fh)


def load_metrics() -> pd.DataFrame:
    return pd.read_parquet(DATA / "battery_metrics.parquet")


def build_smoothing(bm: pd.DataFrame) -> SmoothingCache:
    cache = SmoothingCache()
    cache.update(bm)
    return cache


def build_shape(bm: pd.DataFrame) -> ShapeCache:
    cache = ShapeCache()
    cache.update(bm)
    return cache


# ---------------------------------------------------------------- helpers

def to_ordinal(ts) -> int:
    return int((pd.Timestamp(ts).normalize() - _EPOCH) // pd.Timedelta(days=1))


def from_ordinal(day: int) -> pd.Timestamp:
    return _EPOCH + pd.Timedelta(days=int(day))


def crossing_index(smooth_v: np.ndarray, threshold: float = EOL_THRESHOLD) -> int | None:
    """Index of the first finite smoothed value strictly below ``threshold``."""
    below = np.isfinite(smooth_v) & (smooth_v < threshold)
    if not below.any():
        return None
    return int(np.argmax(below))


def compensate(v: np.ndarray, t: np.ndarray, beta: float = GLOBAL_BETA) -> np.ndarray:
    """Remove the temperature term: v - beta * (t - REF_TEMP). NaN-propagating."""
    return v - beta * (t - REF_TEMP)


def value_at(series: np.ndarray, index: int, lookback: int = 6) -> float:
    """Series value at ``index``; if NaN, the nearest finite value up to
    ``lookback`` days earlier (grid days can be gaps)."""
    if index < 0 or index >= series.size:
        return float("nan")
    lo = max(0, index - lookback)
    window = series[lo : index + 1]
    finite = np.flatnonzero(np.isfinite(window))
    if finite.size == 0:
        return float("nan")
    return float(window[finite[-1]])


def ols_beta(x: np.ndarray, y: np.ndarray, min_points: int = 30) -> float:
    """Slope of y on x over finite pairs; NaN when under-identified."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_points:
        return float("nan")
    xs, ys = x[ok], y[ok]
    var = xs.var()
    if var <= 1e-12:
        return float("nan")
    return float(((xs - xs.mean()) * (ys - ys.mean())).mean() / var)


def ols_slope_time(values: np.ndarray, min_points: int = 8) -> float:
    """Slope per grid day of ``values`` against its own index (finite only)."""
    idx = np.arange(values.size, dtype=float)
    return ols_beta(idx, values, min_points=min_points)


def detrend_rolling_median(values: np.ndarray, window: int = 60, min_periods: int = 30) -> np.ndarray:
    """Centered rolling-median detrend; NaN where the trend is undefined."""
    s = pd.Series(values)
    trend = s.rolling(window=window, min_periods=min_periods, center=True).median()
    return (s - trend).to_numpy(dtype=float)


def resmooth(values: np.ndarray) -> np.ndarray:
    """Apply the official 7-day trailing median to a daily grid series."""
    return _rolling_median(values)


def dist_stats(values, extra_percentiles=(10, 90)) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    out["iqr"] = out["p75"] - out["p25"]
    out["cv"] = out["std"] / out["mean"] if out["mean"] not in (0.0,) else float("nan")
    for p in extra_percentiles:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg) + 0.5 P(tie), rank-based, single-threaded."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def jsonable(obj):
    """Recursively convert numpy scalars / NaN to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    return obj
