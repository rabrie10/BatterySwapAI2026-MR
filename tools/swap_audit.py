"""Which batteries does the planner swap that its own economics reject?

The out-of-fold model puts 11.21 batteries per scenario above the break-even
probability, and the planner swaps 16.5. This prices the difference exactly,
per battery, rather than by counts.

For every battery it reports the standalone decision the cost tables imply --
``min_d service_cost + work`` against ``defer_cost`` -- alongside what the
planner actually did and what the battery actually did. A swap that fails the
standalone test but happens anyway is the leak.

    python tools/swap_audit.py --limit 12
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

from batteryswap_solution.costs import build_expected_cost_tables, isolated_emergency_costs
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel

WORK_HOURS = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_swap_audit.json"))
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
        config=PlannerConfig(optimizer=OptimizationConfig(solver_seconds=1.0)),
    )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"])
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))

        plan = planner.plan(cut, locs, travel, settings)
        served = set(plan.loc[plan["day"] <= horizon_end, "battery"].astype(str))

        dates, _ = planner._planning_clock(start, settings)
        forecast = planner._forecast(cut, locs, start, dates)
        costs = build_expected_cost_tables(forecast, locs, settings, dates)
        emergency = isolated_emergency_costs(locs, travel, settings, costs.battery_ids)

        best_service = costs.service_cost.min(axis=1) + WORK_HOURS
        defer_timing = costs.defer_cost
        defer_full = defer_timing + costs.horizon_event_probability * emergency
        due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)

        for position, battery in enumerate(costs.battery_ids):
            rows.append(
                {
                    "scenario": scenario["name"],
                    "battery": battery,
                    "p": float(costs.horizon_event_probability[position]),
                    "best_service": float(best_service[position]),
                    "defer_timing": float(defer_timing[position]),
                    "defer_full": float(defer_full[position]),
                    "emergency_ops": float(emergency[position]),
                    "standalone_ok": bool(best_service[position] < defer_full[position]),
                    "standalone_ok_timing": bool(
                        best_service[position] < defer_timing[position]
                    ),
                    "swapped": battery in served,
                    "due": battery in due,
                }
            )
        print(f"  {scenario['name']:>5}  swapped={len(served):3d}", flush=True)

    frame = pd.DataFrame(rows)
    n = frame.scenario.nunique()
    swapped = frame[frame.swapped]
    print()
    print("=== per scenario ===")
    print(f"  planner swapped            {len(swapped)/n:6.2f}")
    print(f"  pass standalone (full)     {frame.standalone_ok.sum()/n:6.2f}")
    print(f"  pass standalone (timing)   {frame.standalone_ok_timing.sum()/n:6.2f}")
    print(f"  p > 0.26                   {(frame.p > 0.26).sum()/n:6.2f}")
    print(f"  actually due               {frame.due.sum()/n:6.2f}")
    print()

    leak = swapped[~swapped.standalone_ok]
    print("=== the leak: swapped although the standalone economics say defer ===")
    print(f"  count per scenario   {len(leak)/n:6.2f}")
    if len(leak):
        print(f"  of those, actually due  {leak.due.sum()/max(len(leak),1):6.1%}")
        print(f"  median p                {leak.p.median():6.3f}")
        print(f"  median loss per swap    {(leak.best_service - leak.defer_full).median():6.1f}")
        print(f"  total loss per scenario {(leak.best_service - leak.defer_full).sum()/n:6.1f}")
    print()
    passed = frame[frame.standalone_ok]
    print("=== batteries that pass standalone ===")
    print(f"  count per scenario   {len(passed)/n:6.2f}")
    print(f"  precision            {passed.due.sum()/max(len(passed),1):6.3f}")
    print(f"  recall               {passed.due.sum()/max(frame.due.sum(),1):6.3f}")
    print()
    print("=== how much of defer_cost is the emergency trip? ===")
    marginal = frame[(frame.p > 0.05) & (frame.p < 0.35)]
    print(f"  on marginal batteries (0.05<p<0.35), n={len(marginal)}")
    print(f"    median defer_timing  {marginal.defer_timing.median():7.2f}")
    print(f"    median emergency ops {(marginal.emergency_ops * marginal.p).median():7.2f}")
    print(f"    median best_service  {marginal.best_service.median():7.2f}")

    # Where in the calendar does the over-swapping happen? The closing scenarios
    # have a near substitute EOL, which makes a wasted swap look cheap.
    frame["index"] = frame.groupby("scenario", sort=False).ngroup()
    print()
    print("=== by scenario block ===")
    print(f"{'block':>12}{'swapped':>9}{'standalone':>12}{'p>0.26':>9}{'due':>7}{'precision':>11}{'wasted_cost':>13}")
    for lo, hi, label in [(0,16,'early 0-15'), (16,32,'mid 16-31'), (32,48,'late 32-47')]:
        block = frame[(frame["index"] >= lo) & (frame["index"] < hi)]
        if block.empty:
            continue
        m = block.scenario.nunique()
        sw = block[block.swapped]
        # what a wasted swap costs here: service cost of the ones that were not due
        wasted = sw[~sw.due]
        print(f"{label:>12}{len(sw)/m:9.2f}{block.standalone_ok.sum()/m:12.2f}"
              f"{(block.p>0.26).sum()/m:9.2f}{block.due.sum()/m:7.2f}"
              f"{sw.due.sum()/max(len(sw),1):11.3f}{wasted.best_service.median() if len(wasted) else 0:13.1f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "scenarios": int(n),
                "swapped_per_scenario": round(len(swapped) / n, 2),
                "standalone_pass_per_scenario": round(frame.standalone_ok.sum() / n, 2),
                "leak_per_scenario": round(len(leak) / n, 2),
                "leak_due_rate": round(float(leak.due.mean()) if len(leak) else 0.0, 4),
                "standalone_precision": round(
                    float(passed.due.sum() / max(len(passed), 1)), 4
                ),
                "standalone_recall": round(
                    float(passed.due.sum() / max(frame.due.sum(), 1)), 4
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
