"""Every swap and every workday the planner produces, with what each one cost.

The leaderboard hands out ten cost columns per team, and two of them are almost
free readouts of the plan itself: ``battery_swap / 0.25`` is exactly the number
of swaps performed (planned plus emergency -- verified against our own local run,
5.40/0.25 = 21.6 = 17.65 served + 3.94 missed), and ``early_swap`` divided by
that is what an average swap costs in earliness.

On that measure first place pays about 24 per planned swap and we pay about 49.
Precision alone does not obviously explain a factor of two, because
``early_swap = 0.5 x (effective EOL - swap day)`` has a second lever nobody here
has measured: **the day**. For 96% of devices the substitute EOL is a fixed
calendar date (last data + 30 days), so a wasted swap costs
``0.5 x (that date - scenario start - swap day)``. Moving it from day 0 to day 42
saves 21, every time, with no change to the model.

This dumps the ledger needed to price that:

* one row per planned swap -- day offset, predicted probability, whether the
  battery was really due, its effective EOL, and the early and late cost paid;
* one row per workday -- hours, overtime, whether a limit was hit;
* one row per week -- hours and whether the weekly limit fired.

With that, the counterfactuals are arithmetic rather than another planner run:
what the timing would cost with perfect hindsight on the same chosen set, and
what pushing the not-due swaps to the end of the window would save.

    python tools/swap_ledger.py --folds outputs/v7_folds.joblib
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

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import replay_operational_cost

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    parser.add_argument("--swaps-out", type=Path, default=Path("outputs/v9_swaps.csv"))
    parser.add_argument("--days-out", type=Path, default=Path("outputs/v9_days.csv"))
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    for model in bundle["by_building"].values():
        model.volatility_scale = args.volatility_scale
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    planner = CompetitionPlanner(
        forecaster=forecaster,
        config=PlannerConfig(
            local_search_evaluations=80,
            uncertain_local_search_evaluations=35,
            optimizer=OptimizationConfig(solver_seconds=1.0),
        ),
    )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    swap_rows: list[dict] = []
    day_rows: list[dict] = []
    week_rows: list[dict] = []

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if args.limit is not None and index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"]).normalize()
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        horizon = float(settings.planning_window_days)
        horizon_end = start + pd.Timedelta(days=horizon)

        plan = planner.plan(cut, locs, travel, settings)
        probability = forecaster.last_probabilities

        # The evaluator's effective EOL: the record if there is one, else the
        # device's last data plus unobserved_eol_days.
        loc = locs.set_index(locs["battery"].astype(str))
        end_time = pd.to_datetime(loc["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        substitute = (
            end_time.dt.normalize()
            + pd.Timedelta(days=float(settings.unobserved_eol_days))
        )
        recorded = not_dead.reindex(substitute.index)
        effective = recorded.fillna(substitute)

        early_rate = float(settings.early_replacement_penalty_daily)
        late_rate = float(settings.late_replacement_penalty_daily)
        active = plan[plan["day"] <= horizon_end]
        served = set(active["battery"].astype(str))
        due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)

        for day, battery in zip(active["day"], active["battery"].astype(str)):
            offset = float((pd.Timestamp(day) - start) / pd.Timedelta(days=1))
            eol_offset = float(
                (pd.Timestamp(effective.loc[battery]) - start) / pd.Timedelta(days=1)
            )
            delta = eol_offset - offset
            swap_rows.append(
                {
                    "scenario_index": index,
                    "scenario": scenario["name"],
                    "battery": battery,
                    "day_offset": offset,
                    "probability": float(probability.get(battery, float("nan"))),
                    "due": battery in due,
                    "eol_offset": eol_offset,
                    "early": early_rate * max(delta, 0.0),
                    "late": late_rate * max(-delta, 0.0),
                    # What the same swap would cost placed as late as the window
                    # allows, and placed with hindsight on its own EOL.
                    "early_at_horizon": early_rate * max(eol_offset - horizon, 0.0),
                    "late_at_horizon": late_rate * max(horizon - eol_offset, 0.0),
                }
            )

        for battery in sorted(due - served):
            swap_rows.append(
                {
                    "scenario_index": index,
                    "scenario": scenario["name"],
                    "battery": str(battery),
                    "day_offset": float("nan"),
                    "probability": float(probability.get(str(battery), float("nan"))),
                    "due": True,
                    "eol_offset": float(
                        (pd.Timestamp(effective.loc[str(battery)]) - start)
                        / pd.Timedelta(days=1)
                    ),
                    "early": 0.0,
                    "late": float("nan"),  # billed by the evaluator, not here
                    "early_at_horizon": float("nan"),
                    "late_at_horizon": float("nan"),
                }
            )

        detail = replay_operational_cost(
            plan, locs, travel, settings, start, include_details=True
        )
        for record in detail["_daily_records"]:
            hours = float(record["hours"])
            if hours <= 1e-9:
                continue
            day_rows.append(
                {
                    "scenario_index": index,
                    "day_offset": float(
                        (pd.Timestamp(record["day"]) - start) / pd.Timedelta(days=1)
                    ),
                    "hours": hours,
                    "return_travel": float(record["return_travel"]),
                    "limit_hit": bool(record["limit_hit"]),
                    "overtime": 2.0 * max(hours - float(settings.overtime_start), 0.0),
                }
            )
        for record in detail["_weekly_records"]:
            week_rows.append(
                {
                    "scenario_index": index,
                    "hours": float(record["hours"]),
                    "limit_hit": bool(record["limit_hit"]),
                }
            )

        _, _, scores = evaluate_plan(
            plan, locs, travel, settings, eol_times=not_dead,
            start_time=start, verbose=0,
        )
        print(
            f"  {scenario['name']:>5} total={float(scores['total_cost']):8.1f} "
            f"early={float(scores['early_swap']):7.1f} late={float(scores['late_swap']):7.1f} "
            f"served={len(served):3d} due={len(due):3d}",
            flush=True,
        )

    swaps = pd.DataFrame(swap_rows)
    days = pd.DataFrame(day_rows)
    weeks = pd.DataFrame(week_rows)
    args.swaps_out.parent.mkdir(parents=True, exist_ok=True)
    swaps.to_csv(args.swaps_out, index=False)
    days.to_csv(args.days_out, index=False)

    n = swaps.scenario_index.nunique()
    planned = swaps[swaps.day_offset.notna()]
    wasted = planned[~planned.due]
    hits = planned[planned.due]

    print()
    print("=== per scenario ===")
    print(f"  planned swaps        {len(planned)/n:7.2f}")
    print(f"  of those due         {len(hits)/n:7.2f}   precision {len(hits)/max(len(planned),1):.3f}")
    print(f"  early paid           {planned.early.sum()/n:7.1f}")
    print(f"    on due             {hits.early.sum()/n:7.1f}   ({hits.early.sum()/max(len(hits),1):5.1f} per swap)")
    print(f"    on not-due         {wasted.early.sum()/n:7.1f}   ({wasted.early.sum()/max(len(wasted),1):5.1f} per swap)")
    print(f"  late paid on planned {planned.late.sum()/n:7.1f}")
    print()
    print("=== where in the window does the planner put swaps? ===")
    for label, block in [("due", hits), ("not due", wasted)]:
        q = np.quantile(block.day_offset, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  {label:8s} day offset  mean {block.day_offset.mean():5.1f}  "
              f"q10/25/50/75/90 {q[0]:5.1f} {q[1]:5.1f} {q[2]:5.1f} {q[3]:5.1f} {q[4]:5.1f}")
    print()
    print("=== counterfactual timing on the SAME chosen batteries ===")
    actual = planned.early.sum() + planned.late.sum()
    at_horizon = planned.early_at_horizon.sum() + planned.late_at_horizon.sum()
    # Perfect hindsight: swap a due battery on its EOL day, a not-due one on the
    # last day of the window.
    perfect = (
        wasted.early_at_horizon.sum()
        + 0.0  # a due battery swapped exactly on its EOL costs nothing
    )
    print(f"  as planned                        {actual/n:8.1f}")
    print(f"  everything on the final day       {at_horizon/n:8.1f}")
    print(f"  perfect hindsight, same set       {perfect/n:8.1f}")
    print(f"  headroom from timing alone        {(actual - perfect)/n:8.1f}")
    print()
    print("=== workday shape ===")
    print(f"  working days per scenario   {len(days)/n:6.2f}")
    print(f"  hours per working day       mean {days.hours.mean():5.2f}  "
          f"median {days.hours.median():5.2f}  p90 {days.hours.quantile(0.9):5.2f}  max {days.hours.max():5.2f}")
    print(f"  days over 8h                {(days.hours > 8).sum()/n:6.2f} per scenario")
    print(f"  days over 24h (penalty 100) {days.limit_hit.sum()/n:6.2f} per scenario")
    print(f"  weeks over 24h (penalty 100){weeks.limit_hit.sum()/n:6.2f} per scenario")
    print(f"  overtime charged            {days.overtime.sum()/n:6.1f} per scenario")
    print(f"  return travel               {days.return_travel.sum()/n:6.1f} per scenario")
    print()
    print("=== by scenario block: what does a wasted swap cost here? ===")
    print(f"{'block':>12}{'planned':>9}{'precision':>11}{'early':>9}{'per wasted':>12}{'eol_offset':>12}")
    for lo, hi, label in [(0,16,'early 0-15'), (16,32,'mid 16-31'), (32,48,'late 32-47')]:
        b = planned[(planned.scenario_index >= lo) & (planned.scenario_index < hi)]
        if b.empty:
            continue
        m = b.scenario_index.nunique()
        bw = b[~b.due]
        print(f"{label:>12}{len(b)/m:9.2f}{b.due.mean():11.3f}{b.early.sum()/m:9.1f}"
              f"{bw.early.sum()/max(len(bw),1):12.1f}{bw.eol_offset.median():12.1f}")

    print(f"\nwrote {args.swaps_out} and {args.days_out}")


if __name__ == "__main__":
    main()
