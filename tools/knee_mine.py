"""Mine the knee pool one level deeper: what separates the dues inside it?

Population: rows with margin in [0.05, 0.15), elevated beta_30, remaining >= 30
-- the same rows the KneeBoost floor governs. Question: among candidate
features computable at the cutoff (slope_30, knee_worst_14d_drop, v_range_rise,
season, and friends), which separate the ~0.2-realized from the ~0.02-realized?

Features are recomputed at every scenario cutoff with the exact production
code path (bsai.features.feature_row on the smoothing + shape caches) and
cached in the session scratchpad, so this is heavy once and cheap after.

    python tools/knee_mine.py [--cache PATH]

Prints AUCs pooled and within remaining bands (the axis everything is
confounded with), plus a split check inside the active floor cells. Appends
nothing to outputs/ except what fit_knee/knee_analytic already write; the
numbers land in outputs/knee_findings.md by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES, DeviceView, FeatureContext, feature_row
from bsai.shape import ShapeCache, align_to
from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")

KEEP = [
    "slope_7",
    "slope_14",
    "slope_30",
    "slope_ratio_14_90",
    "knee_slope_vs_history",
    "knee_trend_residual",
    "knee_worst_14d_drop",
    "knee_recent_vs_baseline",
    "beta_rise",
    "v_std_rise",
    "v_range_rise",
    "v_std_30",
    "v_range_30",
    "days_below_2.45",
    "days_below_2.50",
    "drawdown",
    "range_90",
]


def build_features(frame: pd.DataFrame, dataset: Path, folds: Path) -> pd.DataFrame:
    bundle = joblib.load(folds)
    context = FeatureContext(climatology=bundle["climatology"])

    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    smooth = SmoothingCache()
    smooth.update(raw)
    shapes = ShapeCache()
    shapes.update(raw)
    del raw

    scen = json.loads((dataset / "scenarios.json").read_text())
    start_ordinal = {
        i: int((pd.Timestamp(s["start_time"]).normalize() - _EPOCH) / pd.Timedelta(days=1))
        for i, s in enumerate(scen)
    }

    indices = [FEATURE_NAMES.index(name) for name in KEEP]
    out = np.full((len(frame), len(KEEP)), np.nan)
    frame = frame.reset_index(drop=True)
    for battery, group in frame.groupby("battery"):
        series = smooth.devices.get(battery)
        if series is None:
            continue
        view = DeviceView(series.smooth_voltage, series.smooth_temperature)
        shape_view = align_to(shapes.devices.get(battery), series.origin, len(series))
        for row in group.itertuples():
            index = min(start_ordinal[row.scenario] - series.origin, len(series) - 1)
            if index < 0:
                continue
            values = feature_row(
                view, index, series.origin + index, context, shape_view
            )
            if values is None:
                continue
            out[row.Index] = np.asarray(values, dtype=float)[indices]
    result = frame.copy()
    for j, name in enumerate(KEEP):
        result[name] = out[:, j]
    months = {i: pd.Timestamp(s["start_time"]).month for i, s in enumerate(scen)}
    days = {i: pd.Timestamp(s["start_time"]).dayofyear for i, s in enumerate(scen)}
    result["month"] = result.scenario.map(months)
    result["winterness"] = np.cos(
        2 * np.pi * (result.scenario.map(days) - 15) / 365.25
    )
    return result


def auc(values: np.ndarray, label: np.ndarray) -> tuple[float, int, int]:
    """Rank AUC of value against label on finite rows (ties averaged)."""
    finite = np.isfinite(values)
    values, label = values[finite], label[finite].astype(bool)
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan"), pos, neg
    order = pd.Series(values).rank(method="average").to_numpy()
    stat = (order[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(stat), pos, neg


def report(pool: pd.DataFrame, title: str, columns: list[str]) -> list[dict]:
    print(f"\n== {title}: n={len(pool)}, dues={int(pool.due.sum())} "
          f"(rate {pool.due.mean():.3f}) ==")
    rows = []
    for name in columns:
        stat, pos, neg = auc(pool[name].to_numpy(dtype=float), pool.due.to_numpy())
        rows.append({"feature": name, "auc": stat, "pos": pos, "neg": neg})
    rows.sort(key=lambda r: abs(r["auc"] - 0.5) if np.isfinite(r["auc"]) else -1, reverse=True)
    for r in rows:
        direction = "high->due" if r["auc"] >= 0.5 else "low->due"
        print(f"  {r['feature']:>24}: AUC {r['auc']:.3f} ({direction}, "
              f"{r['pos']} due / {r['neg']} not)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/frame_oof_raw_beta.parquet"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    cache = args.cache
    if cache is None:
        scratch = os.environ.get("CLAUDE_SCRATCHPAD") or str(
            Path(os.environ.get("TEMP", ".")) / "knee_scratch"
        )
        Path(scratch).mkdir(parents=True, exist_ok=True)
        cache = Path(scratch) / "knee_features.parquet"

    if cache.exists():
        mined = pd.read_parquet(cache)
        print(f"loaded cached features from {cache}")
    else:
        frame = pd.read_parquet(args.frame)
        print("computing features at every scenario cutoff (heavy, once)...")
        mined = build_features(frame, args.dataset, args.folds)
        mined.to_parquet(cache)
        print(f"cached to {cache}")

    candidates = ["slope_30", "knee_worst_14d_drop", "v_range_rise", "winterness"]
    extras = [
        "slope_7", "slope_14", "slope_ratio_14_90", "knee_slope_vs_history",
        "knee_trend_residual", "knee_recent_vs_baseline", "beta_rise",
        "v_std_rise", "v_std_30", "v_range_30", "days_below_2.50",
        "drawdown", "range_90", "staleness", "remaining",
    ]

    pool = mined[
        (mined.margin >= 0.05) & (mined.margin < 0.15)
        & np.isfinite(mined.beta30) & (mined.beta30 >= 0.008)
        & (mined.remaining >= 30)
    ]
    report(pool, "pool margin[0.05,0.15) x beta30>=0.008, remaining>=30",
           candidates + extras)

    # The remaining axis dominates; the honest question is what separates
    # within a regime, not what proxies for remaining.
    stock = pool[pool.remaining >= 220]
    flow = pool[pool.remaining < 220]
    report(stock, "stock regime (remaining >= 220)", candidates + extras[:8])
    report(flow, "flow regime (remaining 30-220)", candidates + extras[:8])

    # Would a third axis pay inside the active floor cells (>=25 events/half)?
    print("\n== split check inside active banded cells ==")
    for mlo, mhi in [(0.05, 0.10), (0.10, 0.15)]:
        cell = mined[
            (mined.margin >= mlo) & (mined.margin < mhi)
            & (mined.beta30 >= 0.012) & (mined.remaining >= 220)
        ]
        for name in ["slope_30", "knee_worst_14d_drop", "v_range_rise", "beta_rise"]:
            values = cell[name].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() < 20:
                continue
            median = np.nanmedian(values)
            low = cell[finite & (values <= median)]
            high = cell[finite & (values > median)]
            print(f"  m[{mlo},{mhi}) {name:>20}: "
                  f"<=med {len(low):>3} rows/{int(low.due.sum()):>2} ev (rate {low.due.mean():.2f})  "
                  f">med {len(high):>3} rows/{int(high.due.sum()):>2} ev (rate {high.due.mean():.2f})")


if __name__ == "__main__":
    main()
