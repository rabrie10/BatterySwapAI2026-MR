"""Screen V9 top-K policies while computing each expensive V7 forecast once."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from bsai.forecaster import HazardForecaster
from bsai.portfolio import annotate_portfolio
from bsai.validation import OofHazardModel


class FixedForecaster:
    def __init__(self, forecast):
        self.forecast = forecast

    def predict(self, *args, **kwargs):
        return self.forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_wiener_folds.joblib"))
    parser.add_argument("--incidence-model", type=Path, default=Path("outputs/v9_incidence_oof.joblib"))
    parser.add_argument("--indices", default="0,3,6,20,23,26,40,43,46")
    parser.add_argument(
        "--policies",
        default="1.25:0,1.25:6,1.5:4,1.5:6,1.75:4",
        help="comma-separated multiplier:buffer pairs",
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/v9_portfolio_screen.json"))
    args = parser.parse_args()

    wanted = {int(value) for value in args.indices.split(",") if value.strip()}
    policies = [
        (float(multiplier), float(buffer))
        for multiplier, buffer in (
            token.split(":", 1) for token in args.policies.split(",")
        )
    ]
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    base = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    incidence = joblib.load(args.incidence_model)
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows = []
    for index, (scenario, locs, cut, active_eol) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if index not in wanted:
            continue
        start = pd.Timestamp(scenario["start_time"])
        horizon = int(scenario["settings"].planning_window_days)
        forecast = base.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        horizon_end = start + pd.Timedelta(days=float(horizon))
        due = set(
            active_eol[(active_eol.notna()) & (active_eol <= horizon_end)].index
        )
        cases = [("base", forecast, float(base.last_expected_due), 0)]
        for multiplier, buffer in policies:
            policy = replace(
                incidence,
                service_multiplier=multiplier,
                service_buffer=buffer,
            )
            annotated, predicted_due, budget, _ = annotate_portfolio(forecast, policy)
            cases.append((f"{multiplier:g}:{buffer:g}", annotated, predicted_due, budget))

        for policy_name, policy_forecast, predicted_due, budget in cases:
            planner = CompetitionPlanner(
                FixedForecaster(policy_forecast),
                PlannerConfig(
                    local_search_evaluations=40,
                    uncertain_local_search_evaluations=20,
                    optimizer=OptimizationConfig(solver_seconds=0.5),
                ),
            )
            plan = planner.plan(cut, locs, scenario["travel_costs"], scenario["settings"])
            _, _, score = evaluate_plan(
                plan,
                locs,
                scenario["travel_costs"],
                scenario["settings"],
                eol_times=active_eol,
                start_time=start,
                verbose=0,
            )
            served = set(plan.loc[plan["day"] <= horizon_end, "battery"].astype(str))
            row = {
                "scenario": scenario["name"],
                "scenario_index": index,
                "policy": policy_name,
                "predicted_due": predicted_due,
                "budget": budget,
                "served": len(served),
                "due": len(due),
                "hits": len(served & due),
                "total_cost": float(score["total_cost"]),
            }
            row.update({component: float(score[component]) for component in cost_components})
            rows.append(row)
            print(
                f"{scenario['name']} policy={row['policy']} budget={budget:2d} "
                f"served={len(served):2d} due={len(due):2d} total={row['total_cost']:.1f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = []
    for policy, block in frame.groupby("policy", sort=False):
        summary.append(
            {
                "policy": policy,
                "mean_total_cost": round(float(block["total_cost"].mean()), 3),
                "mean_early": round(float(block["early_swap"].mean()), 3),
                "mean_late": round(float(block["late_swap"].mean()), 3),
                "mean_budget": round(float(block["budget"].mean()), 3),
                "mean_served": round(float(block["served"].mean()), 3),
                "precision": round(float(block["hits"].sum() / max(block["served"].sum(), 1)), 4),
                "recall": round(float(block["hits"].sum() / max(block["due"].sum(), 1)), 4),
            }
        )
    summary.sort(key=lambda row: row["mean_total_cost"])
    report = {"indices": sorted(wanted), "summary": summary, "scenarios": rows}
    print(json.dumps(report, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
