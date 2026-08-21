"""Rank by extrapolated days to crossing, with no model at all.

This is the control every learned forecast has to beat. Take the current
smoothed margin, fit a slope over a recent window, and extrapolate to 2.4 V:

    days_to_crossing = margin / -slope        (infinite if the slope is flat or up)

If a rule this crude ranks batteries about as well as the gradient-boosted
model, then the model is not extracting anything from its fifty-one features and
the problem is the pipeline, not the sample size. Scoring is identical to
``tools/ranking_v7.py`` so the numbers sit side by side.

    python tools/physics_baseline.py --window 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset

from bsai.margin import EOL_THRESHOLD
from bsai.smoothing import SmoothingCache

SWAP_COUNTS = (8, 10, 12, 15, 18, 21, 25, 30)
EMERGENCY_OFFSET_DAYS = 48.0
_EPOCH = pd.Timestamp("1970-01-01")


def days_to_crossing(
    cache: SmoothingCache, battery_ids: list[str], origin_ordinal: int, window: int
) -> pd.Series:
    """Straight-line extrapolation of the smoothed margin to zero."""
    out = np.full(len(battery_ids), np.inf)
    for position, device_id in enumerate(battery_ids):
        series = cache.devices.get(device_id)
        if series is None:
            continue
        index = min(series.index_of(origin_ordinal), len(series) - 1)
        if index < window:
            continue
        segment = series.smooth_voltage[max(0, index - window) : index + 1]
        valid = np.flatnonzero(~np.isnan(segment))
        if valid.size < max(5, window // 4):
            continue
        level = float(segment[valid[-1]]) - EOL_THRESHOLD
        if level <= 0.0:
            out[position] = 0.0
            continue
        slope = float(np.polyfit(valid.astype(float), segment[valid], 1)[0])
        if slope >= -1e-6:
            continue
        out[position] = level / -slope
    return pd.Series(out, index=battery_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--report", type=Path, default=Path("outputs/v7_physics.json"))
    args = parser.parse_args()

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    cache = SmoothingCache()
    rows: list[dict] = []

    for scenario, locs, cut, not_dead in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        cache.update(cut)
        start = pd.Timestamp(scenario["start_time"]).normalize()
        settings = scenario["settings"]
        horizon = int(settings.planning_window_days)
        horizon_end = start + pd.Timedelta(days=horizon)
        origin_ordinal = int((start - _EPOCH) / pd.Timedelta(days=1))

        battery_ids = locs["battery"].astype(str).tolist()
        estimate = days_to_crossing(cache, battery_ids, origin_ordinal, args.window)

        end_time = pd.to_datetime(locs["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        substitute = end_time.dt.normalize() + pd.Timedelta(
            days=float(settings.unobserved_eol_days)
        )
        substitute.index = np.asarray(battery_ids)
        recorded = not_dead.reindex(substitute.index)
        effective = recorded.fillna(substitute)
        days_to_eol = ((effective - start) / pd.Timedelta(days=1)).astype(float)
        due = (recorded.notna()) & (recorded <= horizon_end)

        # Soonest predicted crossing first.
        ranked = estimate.sort_values(ascending=True)
        for k in SWAP_COUNTS:
            chosen = ranked.index[:k]
            chosen_due = due.reindex(chosen).fillna(False).to_numpy()
            hits = int(chosen_due.sum())
            wasted = np.clip(
                days_to_eol.reindex(chosen).to_numpy() - horizon, 0.0, None
            )
            early = float(0.5 * wasted[~chosen_due].sum() + 0.5 * 5.0 * hits)
            missed = due & ~due.index.isin(chosen)
            late = float(
                10.0
                * np.clip(
                    EMERGENCY_OFFSET_DAYS
                    - days_to_eol.reindex(missed[missed].index).to_numpy(),
                    0.0,
                    None,
                ).sum()
            )
            rows.append(
                {
                    "k": k,
                    "due": int(due.sum()),
                    "hits": hits,
                    "early": early,
                    "late": late,
                    "timing": early + late,
                }
            )
        print(f"  {scenario['name']:>5}  due={int(due.sum()):3d}", flush=True)

    frame = pd.DataFrame(rows)
    summary = []
    for k, block in frame.groupby("k"):
        summary.append(
            {
                "k": int(k),
                "recall": round(float(block.hits.sum() / max(block.due.sum(), 1)), 4),
                "precision": round(float(block.hits.sum() / (int(k) * len(block))), 4),
                "early": round(float(block.early.mean()), 1),
                "late": round(float(block.late.mean()), 1),
                "timing": round(float(block.timing.mean()), 1),
            }
        )
    print()
    print(f"=== physics baseline: margin / -slope, window {args.window}d ===")
    print(f"{'k':>4} {'recall':>8} {'precis':>8} {'early':>9} {'late':>9} {'timing':>9}")
    for row in summary:
        print(
            f"{row['k']:>4} {row['recall']:>8.3f} {row['precision']:>8.3f} "
            f"{row['early']:>9.1f} {row['late']:>9.1f} {row['timing']:>9.1f}"
        )
    best = min(summary, key=lambda r: r["timing"])
    print(f"\nbest k = {best['k']}  timing = {best['timing']}")
    print("V6 learned model for comparison: k=12 precision 0.300, best timing 1813.5")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"window": args.window, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
