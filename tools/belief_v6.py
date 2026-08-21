"""Does the optimizer's expected score match reality?

Everything the planner does is a consequence of ``_expected_score``. If that
number is close to the realised evaluator cost, the search is optimising the
right thing and any remaining gap is model error. If it is far off, the search
is confidently walking towards the wrong plan and no amount of model work will
help.

    python tools/belief_v6.py --limit 12
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

from batteryswap_public.evaluate import evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from batteryswap_solution.costs import build_expected_cost_tables, select_candidates
from batteryswap_solution.forecast import validate_forecast
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import build_replay_context
from batteryswap_solution.routing import order_assignments

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v6_folds.joblib"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    planner = CompetitionPlanner(forecaster=forecaster, config=PlannerConfig(
        local_search_evaluations=80,
        uncertain_local_search_evaluations=35,
        optimizer=OptimizationConfig(solver_seconds=1.0),
    ))

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows = []
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"])
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        plan = planner.plan(cut, locs, travel, settings)

        # Rebuild what the planner believed about this plan.
        dates, defer_day = planner._planning_clock(start, settings)
        forecast = planner._forecast(cut, locs, start, dates)
        full = build_expected_cost_tables(forecast, locs, settings, dates)
        keep = select_candidates(full)
        costs = full.take(keep)
        candidate_ids = set(costs.battery_ids)
        candidate_locations = locs[locs["battery"].astype(str).isin(candidate_ids)]
        context = build_replay_context(candidate_locations, travel, settings, start)
        trimmed = plan[plan["battery"].astype(str).isin(candidate_ids)].reset_index(drop=True)
        believed = planner._expected_score(
            trimmed,
            costs,
            np.empty((0, len(costs.battery_ids)), dtype=bool),
            candidate_locations,
            travel,
            settings,
            start,
            context,
        )

        _, _, scores = evaluate_plan(
            plan, locs, travel, settings, eol_times=not_dead, start_time=start, verbose=0
        )
        realised = float(scores["total_cost"])
        rows.append(
            {
                "scenario": scenario["name"],
                "believed": round(believed, 1),
                "realised": round(realised, 1),
                "gap": round(realised - believed, 1),
            }
        )
        print(
            f"  {scenario['name']:>5}  believed={believed:8.1f}  "
            f"realised={realised:8.1f}  gap={realised - believed:+8.1f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    print()
    print(
        json.dumps(
            {
                "mean_believed": round(float(frame["believed"].mean()), 1),
                "mean_realised": round(float(frame["realised"].mean()), 1),
                "mean_gap": round(float(frame["gap"].mean()), 1),
                "correlation": round(float(frame["believed"].corr(frame["realised"])), 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
