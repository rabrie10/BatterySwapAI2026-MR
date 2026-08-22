"""Measure forecast, scheduling, and route-order gaps on train scenarios.

This tool intentionally consumes the train-only EOL labels.  It is diagnostic
code and must never be imported by the submission entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset

from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.routing import order_assignments
from tools.benchmark_task2 import OracleForecaster, all_defer_plan


POLICIES = ("all_defer", "exact_eol_raw", "exact_eol_routed", "oracle_planner")


def exact_eol_assignments(
    locations: pd.DataFrame,
    active_eol: pd.Series,
    horizon_end: pd.Timestamp,
) -> dict[str, pd.Timestamp | None]:
    """Service observed in-window failures exactly on EOL; defer all others."""
    assignments: dict[str, pd.Timestamp | None] = {}
    for battery_id in locations["battery"].astype(str):
        eol = active_eol.get(battery_id, pd.NaT)
        assignments[battery_id] = (
            pd.Timestamp(eol).normalize()
            if pd.notna(eol) and pd.Timestamp(eol) <= horizon_end
            else None
        )
    return assignments


def raw_plan(
    assignments: dict[str, pd.Timestamp | None], defer_day: pd.Timestamp
) -> pd.DataFrame:
    """Build a valid plan with deliberately naive alphabetical within-day order."""
    rows = [
        {
            "day": pd.Timestamp(day).normalize() if day is not None else defer_day,
            "battery": battery_id,
        }
        for battery_id, day in assignments.items()
    ]
    return pd.DataFrame(rows).sort_values(["day", "battery"]).reset_index(drop=True)


def score_plan(plan, locs, scenario, active_eol) -> dict[str, float]:
    _, _, score = evaluate_plan(
        plan,
        locs,
        scenario["travel_costs"],
        scenario["settings"],
        eol_times=active_eol,
        start_time=pd.Timestamp(scenario["start_time"]),
        verbose=0,
    )
    row = {component: float(score[component]) for component in cost_components}
    row["total_cost"] = float(score["total_cost"])
    return row


def summarise(rows: list[dict]) -> dict[str, dict]:
    frame = pd.DataFrame(rows)
    result: dict[str, dict] = {}
    for policy in POLICIES:
        subset = frame[frame["policy"] == policy]
        result[policy] = {
            "mean_total_cost": round(float(subset["total_cost"].mean()), 4),
            "median_total_cost": round(float(subset["total_cost"].median()), 4),
            "p90_total_cost": round(float(subset["total_cost"].quantile(0.9)), 4),
            "components": {
                component: round(float(subset[component].mean()), 4)
                for component in cost_components
            },
        }
    means = {policy: result[policy]["mean_total_cost"] for policy in POLICIES}
    result["measured_gaps"] = {
        "route_order_gain_exact_schedule": round(
            means["exact_eol_raw"] - means["exact_eol_routed"], 4
        ),
        "day_selection_and_batching_gain": round(
            means["exact_eol_routed"] - means["oracle_planner"], 4
        ),
        "oracle_residual": means["oracle_planner"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--solver-seconds", type=float, default=1.0)
    parser.add_argument("--local-search", type=int, default=80)
    parser.add_argument("--report", type=Path, default=Path("outputs/oracle_gaps.json"))
    parser.add_argument(
        "--forecast-report",
        type=Path,
        default=Path("outputs/v8_w25_incidence_oof_full.json"),
        help="Comparable OOF report whose forecast gap should be calculated",
    )
    args = parser.parse_args()

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    started = time.perf_counter()
    processed = 0
    generator = iterate_scenarios(locations, timeseries, eol_times, scenarios)
    for index, (scenario, locs, cut, active_eol) in enumerate(generator):
        if index < args.start_index:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        start = pd.Timestamp(scenario["start_time"])
        horizon_end = start + pd.Timedelta(
            days=float(scenario["settings"].planning_window_days)
        )
        defer_day = horizon_end.normalize() + pd.Timedelta(days=1)
        assignments = exact_eol_assignments(locs, active_eol, horizon_end)
        plans = {
            "all_defer": all_defer_plan(locs, start, scenario["settings"]),
            "exact_eol_raw": raw_plan(assignments, defer_day),
            "exact_eol_routed": order_assignments(
                assignments,
                locs,
                scenario["travel_costs"],
                scenario["settings"].base_location,
                defer_day,
            ),
        }
        planner = CompetitionPlanner(
            forecaster=OracleForecaster(active_eol),
            config=PlannerConfig(
                local_search_evaluations=args.local_search,
                uncertain_local_search_evaluations=35,
                robust_emergency_samples=4,
                optimizer=OptimizationConfig(
                    solver_seconds=args.solver_seconds,
                    expected_due_multiplier=2.0,
                    expected_due_buffer=5.0,
                ),
            ),
        )
        plans["oracle_planner"] = planner.plan(
            cut, locs, scenario["travel_costs"], scenario["settings"]
        )

        totals = {}
        for policy, plan in plans.items():
            entry = score_plan(plan, locs, scenario, active_eol)
            entry.update(policy=policy, scenario=scenario["name"], scenario_index=index)
            rows.append(entry)
            totals[policy] = entry["total_cost"]
        print(
            f"{scenario['name']:>5}  "
            + "  ".join(f"{policy}={totals[policy]:8.2f}" for policy in POLICIES),
            flush=True,
        )

    if not rows:
        raise RuntimeError("No scenarios were evaluated")
    summary = summarise(rows)
    if args.forecast_report.exists():
        forecast_report = json.loads(args.forecast_report.read_text())
        forecast_summary = forecast_report["summary"]
        forecast_n = int(forecast_summary["n_scenarios"])
        if forecast_n == processed:
            forecast_total = float(forecast_summary["mean_total_cost"])
            oracle_total = float(summary["oracle_planner"]["mean_total_cost"])
            summary["forecast_reference"] = {
                "path": str(args.forecast_report),
                "mean_total_cost": forecast_total,
                "forecast_to_oracle_gap": round(forecast_total - oracle_total, 4),
                "fraction_of_forecast_score_above_oracle": round(
                    (forecast_total - oracle_total) / forecast_total, 4
                ),
            }
        else:
            summary["forecast_reference"] = {
                "path": str(args.forecast_report),
                "not_compared": (
                    f"scenario count mismatch: forecast={forecast_n}, oracle={processed}"
                ),
            }
    report = {
        "diagnostic_only": True,
        "n_scenarios": processed,
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scenarios": rows,
    }
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
