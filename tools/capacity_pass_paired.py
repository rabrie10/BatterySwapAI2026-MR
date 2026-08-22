"""Paired within-process A/B for the capacity post-pass.

CP-SAT's 1 s wall-clock termination re-rolls the search incumbent between
processes (measured: 20/48 scenarios drift between identically configured
runs, +-50/scenario-mean), so cross-run validation diffs cannot attribute a
+-10..20 effect. This runner kills the drift by construction: per scenario it
builds ONE incumbent (forecast -> costs -> CP-SAT -> local search), then
scores the official evaluate_plan on that incumbent both without and with
``CompetitionPlanner._capacity_repair``. The delta is exactly the pass.

    python tools/capacity_pass_paired.py --report outputs/capacity_paired_ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from batteryswap_solution.costs import build_expected_cost_tables, select_candidates
from batteryswap_solution.optimizer import OptimizationConfig, optimize_assignments
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v8_folds_cens.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/capacity_paired_ab.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    model = OofHazardModel(
        by_building=bundle["by_building"],
        building_of=building_of,
        climatology=bundle["climatology"],
    )
    forecaster = HazardForecaster(model, probability_scale=1.0)
    forecaster.rank_calibration = bundle.get("rank_calibration")

    config = PlannerConfig(
        local_search_evaluations=240,
        uncertain_local_search_evaluations=240,
        robust_emergency_samples=0,
        optimizer=OptimizationConfig(
            solver_seconds=1.0,
            expected_due_multiplier=1.6,
            expected_due_buffer=1.0,
            max_planned_count=15,
        ),
    )
    planner = CompetitionPlanner(forecaster=forecaster, config=config)

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if args.limit is not None and index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"]).normalize()
        settings = scenario["settings"]
        travel_costs = scenario["travel_costs"]
        dates, defer_day = planner._planning_clock(start, settings)
        forecast = planner._forecast(cut, locs, start, dates)
        full_costs = build_expected_cost_tables(
            forecast, locs, settings, dates,
            late_risk_multiplier=config.late_risk_multiplier,
            emergency_rank_scale=config.emergency_rank_scale,
        )
        keep = select_candidates(
            full_costs,
            margin_hours=config.candidate_margin_hours,
            max_candidates=config.max_candidates,
        )
        costs = full_costs.take(keep)
        candidate_ids = set(costs.battery_ids)
        excluded = [b for b in full_costs.battery_ids if b not in candidate_ids]
        id_column = "battery_id" if "battery_id" in locs else "battery"
        candidate_locations = locs[locs[id_column].astype(str).isin(candidate_ids)]
        seeds = [
            optimize_assignments(
                costs, candidate_locations, travel_costs, settings,
                config=config.optimizer,
            )
        ]
        base_plan = planner._local_search(
            seeds, costs, candidate_locations, travel_costs, settings, start, defer_day
        )
        began = time.time()
        repaired_plan = planner._capacity_repair(
            base_plan, costs, candidate_locations, travel_costs, settings, start, defer_day
        )
        pass_seconds = time.time() - began

        def official(plan: pd.DataFrame) -> dict[str, float]:
            full = planner._restore_excluded(plan, excluded, defer_day)
            _, _, scores = evaluate_plan(
                full, locs, travel_costs, settings,
                eol_times=not_dead, start_time=start, verbose=0,
            )
            return {c: float(scores[c]) for c in list(cost_components) + ["total_cost"]}

        before = official(base_plan)
        after = official(repaired_plan)
        base_days = base_plan.set_index("battery")["day"]
        after_days = repaired_plan.set_index("battery")["day"].reindex(base_days.index)
        moved = int((base_days != after_days).sum())
        entry = {
            "scenario": scenario["name"],
            "moved_batteries": moved,
            "pass_seconds": round(pass_seconds, 3),
            "before": before,
            "after": after,
            "delta_total": round(after["total_cost"] - before["total_cost"], 2),
        }
        rows.append(entry)
        print(
            f"  {scenario['name']:>5s} delta {entry['delta_total']:+9.1f} "
            f"moved {moved:2d} pass {pass_seconds:5.2f}s",
            flush=True,
        )

    frame = rows
    components = list(cost_components) + ["total_cost"]
    means = {
        c: round(
            sum(r["after"][c] - r["before"][c] for r in frame) / len(frame), 3
        )
        for c in components
    }
    summary = {
        "n_scenarios": len(frame),
        "mean_component_delta": means,
        "mean_pass_seconds": round(sum(r["pass_seconds"] for r in frame) / len(frame), 3),
        "max_pass_seconds": round(max(r["pass_seconds"] for r in frame), 3),
        "scenarios_changed": sum(1 for r in frame if r["moved_batteries"]),
        "scenarios_improved": sum(1 for r in frame if r["delta_total"] < -0.5),
        "scenarios_worsened": sum(1 for r in frame if r["delta_total"] > 0.5),
    }
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "scenarios": frame}, indent=2))


if __name__ == "__main__":
    main()
