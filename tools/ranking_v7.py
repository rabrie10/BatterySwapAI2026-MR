"""Compare forecasts at the operating point the leaderboard actually charges.

PR-AUC integrates over every threshold, but we only ever swap somewhere between
ten and twenty-five batteries per scenario. First place spends 24.0 of early
cost per planned swap; we spend 56.6. So the question is not "which model has
the better curve" but "at k swaps per scenario, which model catches more due
batteries" -- and what that costs.

The cost estimate uses the evaluator's own structure: a swap on a battery that
is not due costs ``0.5 x (effective EOL - swap day)``, and a due battery left
unswapped costs ``10 x days late`` on its emergency visit. No planner runs here,
so this is cheap enough to compare models on every change.

    python tools/ranking_v7.py --folds outputs/v7_folds.joblib
    python tools/ranking_v7.py --folds outputs/v6_folds.joblib --label v6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel

SWAP_COUNTS = (8, 10, 12, 15, 18, 21, 25, 30)

# A fixed k per scenario cannot be right: the due count ranges from 2 to 19
# across the train scenarios, so any single k is far too many in the quiet ones
# and too few in the busy ones. A probability threshold adapts by itself, and it
# is what the planner's expected-cost rule actually does. Break-even is about
# 0.26 -- a wasted swap costs roughly 87, a missed one 270.
THRESHOLDS = (0.10, 0.15, 0.20, 0.26, 0.32, 0.40, 0.50, 0.62)
EMERGENCY_OFFSET_DAYS = 48.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--label", default="model")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="strip the remaining-observation correction; compare raw models",
    )
    parser.add_argument(
        "--volatility-scale",
        type=float,
        default=None,
        help="override the Wiener volatility scale",
    )
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    for fold_model in bundle["by_building"].values():
        if args.no_calibration:
            fold_model.calibration = None
        if args.volatility_scale is not None:
            fold_model.volatility_scale = args.volatility_scale
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    for scenario, locs, cut, not_dead in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        start = pd.Timestamp(scenario["start_time"])
        settings = scenario["settings"]
        horizon = int(settings.planning_window_days)
        horizon_end = start + pd.Timedelta(days=horizon)

        forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probability = forecaster.last_probabilities

        # Effective EOL: the recorded one, else the evaluator's substitute.
        end_time = pd.to_datetime(locs["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        substitute = (
            end_time.dt.normalize()
            + pd.Timedelta(days=float(settings.unobserved_eol_days))
        )
        substitute.index = locs["battery"].astype(str).to_numpy()
        recorded = not_dead.reindex(substitute.index)
        effective = recorded.fillna(substitute)
        days_to_eol = ((effective - start.normalize()) / pd.Timedelta(days=1)).astype(
            float
        )
        due = (recorded.notna()) & (recorded <= horizon_end)

        ranked = probability.reindex(substitute.index).fillna(0.0).sort_values(
            ascending=False
        )
        selections = [(f"k={k}", ranked.index[:k]) for k in SWAP_COUNTS]
        selections += [
            (f"p>{t}", ranked.index[ranked.to_numpy() > t]) for t in THRESHOLDS
        ]
        for rule, chosen in selections:
            k = len(chosen)
            chosen_due = due.reindex(chosen).fillna(False).to_numpy()
            hits = int(chosen_due.sum())
            # Swap a due battery a few days before its EOL; swap a healthy one as
            # late as the window allows, which is what the planner does.
            wasted_days = np.clip(days_to_eol.reindex(chosen).to_numpy() - horizon, 0.0, None)
            early = float(
                0.5 * wasted_days[~chosen_due].sum()
                + 0.5 * 5.0 * max(hits, 0)  # a few days early on the ones that matter
            )
            missed = due & ~due.index.isin(chosen)
            late_days = np.clip(
                EMERGENCY_OFFSET_DAYS - days_to_eol.reindex(missed[missed].index).to_numpy(),
                0.0,
                None,
            )
            late = float(10.0 * late_days.sum())
            rows.append(
                {
                    "scenario": scenario["name"],
                    "rule": rule,
                    "k": k,
                    "due": int(due.sum()),
                    "hits": hits,
                    "missed": int(missed.sum()),
                    "early": early,
                    "late": late,
                    "timing": early + late,
                }
            )
        print(f"  {scenario['name']:>5}  due={int(due.sum()):3d}", flush=True)

    frame = pd.DataFrame(rows)
    summary = []
    for rule, block in frame.groupby("rule", sort=False):
        swaps = float(block.k.sum())
        summary.append(
            {
                "rule": str(rule),
                "swaps_per_scenario": round(float(block.k.mean()), 2),
                "recall": round(float(block.hits.sum() / max(block.due.sum(), 1)), 4),
                "precision": round(float(block.hits.sum() / max(swaps, 1)), 4),
                "early": round(float(block.early.mean()), 1),
                "late": round(float(block.late.mean()), 1),
                "timing": round(float(block.timing.mean()), 1),
                "early_per_swap": round(float(block.early.sum() / max(swaps, 1)), 1),
            }
        )
    print()
    print(f"=== {args.label}: swap the top k by predicted probability ===")
    print(
        f"{'rule':>8} {'swaps':>7} {'recall':>8} {'precis':>8} {'early':>9} {'late':>9} "
        f"{'timing':>9} {'early/swap':>11}"
    )
    for row in summary:
        print(
            f"{row['rule']:>8} {row['swaps_per_scenario']:>7.1f} {row['recall']:>8.3f} {row['precision']:>8.3f} "
            f"{row['early']:>9.1f} {row['late']:>9.1f} {row['timing']:>9.1f} "
            f"{row['early_per_swap']:>11.1f}"
        )
    best = min(summary, key=lambda r: r["timing"])
    print(f"\nbest rule = {best['rule']}  swaps = {best['swaps_per_scenario']}  timing = {best['timing']}")
    print("leaderboard reference: J2W early/swap 24.0 at ~12.6 planned swaps")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"label": args.label, "summary": summary}, indent=2)
        )


if __name__ == "__main__":
    main()
