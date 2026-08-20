"""Benchmark Task 2 on train scenarios with either oracle or fallback risk.

Oracle mode is a development ceiling only. It must never be used in an official
submission because public/private EOL labels are not available to the planner.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset

from batteryswap_solution.forecast import (
    CONTRACT_VERSION,
    ForecastMetadata,
    RiskForecast,
)
from batteryswap_solution.costs import build_expected_cost_tables
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.routing import order_assignments
from batteryswap_solution.replay import replay_operational_cost


warnings.filterwarnings("ignore", category=DeprecationWarning)


class OracleForecaster:
    model_version = "train-only-oracle/v1"

    def __init__(self, eol_times: pd.Series):
        self.eol_times = eol_times.copy()

    def predict(
        self,
        battery_data,
        locations,
        *,
        prediction_origin,
        horizon_days,
        evaluation_observation_end,
    ) -> RiskForecast:
        origin = pd.Timestamp(prediction_origin).normalize()
        dates = pd.date_range(
            origin, origin + pd.Timedelta(days=int(horizon_days)), freq="D"
        )
        ids = locations["battery"].astype(str).tolist()
        curve_rows = []
        tail_rows = []
        summary_rows = []
        for battery_id in ids:
            eol = self.eol_times.loc[battery_id]
            if pd.isna(eol):
                cdf = np.zeros(len(dates), dtype=float)
                observed_tail = 0.0
                mean_excess = 0.0
                unobserved = 1.0
                q50 = np.nan
            else:
                eol = pd.Timestamp(eol)
                event_date = eol.normalize()
                cdf = (dates >= event_date).astype(float)
                if event_date > dates[-1]:
                    observed_tail = 1.0
                    mean_excess = float((eol - dates[-1]) / pd.Timedelta("1D"))
                else:
                    observed_tail = 0.0
                    mean_excess = 0.0
                unobserved = 0.0
                q50 = max(float((eol - origin) / pd.Timedelta("1D")), 0.0)
            curve_rows.extend(
                {
                    "battery_id": battery_id,
                    "forecast_date": day,
                    "failure_cdf": float(probability),
                }
                for day, probability in zip(dates, cdf)
            )
            tail_rows.append(
                {
                    "battery_id": battery_id,
                    "prob_observed_after_horizon": observed_tail,
                    "mean_excess_rul_days_given_observed_after_horizon": mean_excess,
                    "prob_unobserved_eol": unobserved,
                    "prob_no_observed_eol_by_horizon": observed_tail + unobserved,
                }
            )
            summary_rows.append(
                {
                    "battery_id": battery_id,
                    "q50_days": q50,
                    "data_quality": 1.0,
                    "cold_start": False,
                }
            )
        metadata = ForecastMetadata(
            contract_version=CONTRACT_VERSION,
            model_version=self.model_version,
            prediction_origin=origin,
            forecast_end_date=dates[-1],
            horizon_days=int(horizon_days),
            evaluation_observation_end=pd.Timestamp(evaluation_observation_end).normalize(),
        )
        return RiskForecast(
            metadata,
            pd.DataFrame(curve_rows),
            pd.DataFrame(tail_rows),
            pd.DataFrame(summary_rows),
        )


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


LOG_COLUMNS = [
    "timestamp_utc",
    "commit",
    "mode",
    "model_version",
    "n_scenarios",
    "total_cost",
    "all_defer",
    "battery_swap",
    "building_change",
    "room_change",
    "travel",
    "overtime",
    "daily_limit",
    "weekly_limit",
    "late_swap",
    "early_swap",
    "swaps",
    "elapsed_seconds",
    "forecasting_seconds",
    "planning_seconds",
    "label",
]


class TimedForecaster:
    """Transparent benchmark-only wrapper measuring Task 1 inference time."""

    def __init__(self, forecaster) -> None:
        self.forecaster = forecaster
        self.elapsed_seconds = 0.0
        self.model_version = getattr(forecaster, "model_version", "real")

    def predict(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.forecaster.predict(*args, **kwargs)
        finally:
            self.elapsed_seconds += time.perf_counter() - started


def record_benchmark(
    log_path: Path,
    mean_row: pd.Series,
    *,
    mode: str,
    model_version: str,
    n_scenarios: int,
    elapsed_seconds: float,
    label: str,
) -> None:
    """Append one comparable row to a local, human-readable benchmark log.

    Columns match the official leaderboard's cost-component breakdown
    exactly, so a local train-split run and a real public/private leaderboard
    entry can be compared column-for-column. Meant to let iteration happen
    against the local evaluator (unlimited runs) instead of the 5/day
    official submission limit.
    """

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git_commit_sha(),
        "mode": mode,
        "model_version": model_version,
        "n_scenarios": n_scenarios,
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "label": label,
    }
    for column in LOG_COLUMNS:
        if column in entry:
            continue
        entry[column] = round(float(mean_row[column]), 4) if column in mean_row else ""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(",".join(LOG_COLUMNS) + "\n")
        handle.write(",".join(str(entry[column]) for column in LOG_COLUMNS) + "\n")


def all_defer_plan(locations: pd.DataFrame, start: pd.Timestamp, settings) -> pd.DataFrame:
    day = start.normalize() + pd.Timedelta(
        f"{float(settings.planning_window_days) + 1} days"
    )
    return pd.DataFrame(
        {"day": [day] * len(locations), "battery": sorted(locations["battery"].astype(str))}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset/train"))
    parser.add_argument("--limit", type=int, default=3, help="Number of scenarios; 0 means all")
    parser.add_argument("--solver-seconds", type=float, default=2.0)
    parser.add_argument("--local-search", type=int, default=160)
    parser.add_argument("--robust-samples", type=int, default=4)
    parser.add_argument(
        "--late-risk-multiplier",
        type=float,
        default=1.0,
        help="Task 2 decision-risk policy: scales the late-swap penalty in the expected-cost "
        "model. <1 makes the planner more conservative about swapping (fewer, later swaps).",
    )
    parser.add_argument("--mode", choices=["oracle", "fallback", "real"], default="oracle")
    parser.add_argument(
        "--forecaster-path",
        type=Path,
        default=Path("models/risk_forecaster.pkl"),
        help="Task 1 artifact used only when --mode=real",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scenario-index", type=int)
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Append the mean-row result to this CSV log (e.g. docs/local_benchmark_log.csv) "
        "for tracking improvements across changes without spending official submissions.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Short free-text note stored alongside a --record entry (e.g. 'baseline', 'after X fix').",
    )
    args = parser.parse_args()

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    rows = []
    model_version = args.mode
    started = time.perf_counter()
    for scenario_index, (scenario, locs, cut, active_eol) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if args.scenario_index is not None and scenario_index < args.scenario_index:
            continue
        if args.scenario_index is not None and scenario_index > args.scenario_index:
            break
        if args.limit and scenario_index >= args.limit:
            break
        if args.mode == "oracle":
            forecaster = OracleForecaster(active_eol)
        elif args.mode == "real":
            with args.forecaster_path.open("rb") as handle:
                forecaster = TimedForecaster(pickle.load(handle))
            model_version = getattr(forecaster, "model_version", "real")
        else:
            forecaster = None
        config = PlannerConfig(
            late_risk_multiplier=args.late_risk_multiplier,
            local_search_evaluations=args.local_search,
            robust_emergency_samples=args.robust_samples,
            optimizer=OptimizationConfig(solver_seconds=args.solver_seconds),
        )
        planner = CompetitionPlanner(forecaster=forecaster, config=config)
        scenario_started = time.perf_counter()
        plan = planner.plan(cut, locs, scenario["travel_costs"], scenario["settings"])
        _, _, score = evaluate_plan(
            plan,
            locs,
            scenario["travel_costs"],
            scenario["settings"],
            eol_times=active_eol,
            start_time=pd.Timestamp(scenario["start_time"]),
            verbose=0,
        )
        baseline = all_defer_plan(
            locs, pd.Timestamp(scenario["start_time"]), scenario["settings"]
        )
        _, _, baseline_score = evaluate_plan(
            baseline,
            locs,
            scenario["travel_costs"],
            scenario["settings"],
            eol_times=active_eol,
            start_time=pd.Timestamp(scenario["start_time"]),
            verbose=0,
        )
        in_window = int(
            (
                plan["day"]
                <= pd.Timestamp(scenario["start_time"])
                + pd.Timedelta(
                    f"{float(scenario['settings'].planning_window_days)} days"
                )
            ).sum()
        )
        row = {
            "scenario": scenario["name"],
            "swaps": in_window,
            "seconds": time.perf_counter() - scenario_started,
            "all_defer": float(baseline_score["total_cost"]),
            "forecasting_seconds": float(getattr(forecaster, "elapsed_seconds", 0.0)),
        }
        row["planning_seconds"] = max(row["seconds"] - row["forecasting_seconds"], 0.0)
        row.update({component: float(score[component]) for component in cost_components})
        row["total_cost"] = float(score["total_cost"])
        rows.append(row)
        if args.show_plan:
            detail = plan.merge(
                locs[["battery", "building", "room"]], on="battery", how="left"
            )
            detail["eol"] = detail["battery"].map(active_eol)
            horizon_end = pd.Timestamp(scenario["start_time"]) + pd.Timedelta(
                f"{float(scenario['settings'].planning_window_days)} days"
            )
            detail = detail[detail["day"] <= horizon_end]
            dates, _ = planner._planning_clock(
                pd.Timestamp(scenario["start_time"]), scenario["settings"]
            )
            forecast = planner._forecast(
                cut,
                locs,
                pd.Timestamp(scenario["start_time"]),
                dates,
            )
            expected_costs = build_expected_cost_tables(
                forecast, locs, scenario["settings"], dates
            )
            positions = {
                battery_id: index
                for index, battery_id in enumerate(expected_costs.battery_ids)
            }
            date_positions = {
                day: index for index, day in enumerate(expected_costs.candidate_dates)
            }
            detail["best_day"] = detail["battery"].map(
                lambda battery_id: expected_costs.candidate_dates[
                    int(np.argmin(expected_costs.service_cost[positions[battery_id]]))
                ]
            )
            detail["expected_timing"] = detail.apply(
                lambda row: expected_costs.service_cost[
                    positions[row["battery"]], date_positions[row["day"]]
                ],
                axis=1,
            )
            print(detail.to_string(index=False), flush=True)
            due_sample = (
                expected_costs.horizon_event_probability[None, :] > 0.5
            )
            current_expected = planner._expected_score(
                plan,
                expected_costs,
                due_sample,
                locs,
                scenario["travel_costs"],
                scenario["settings"],
                pd.Timestamp(scenario["start_time"]),
            )
            print(f"internal_current={current_expected:.6f}", flush=True)
            replay = replay_operational_cost(
                plan,
                locs,
                scenario["travel_costs"],
                scenario["settings"],
                pd.Timestamp(scenario["start_time"]),
                include_details=True,
            )
            print(
                "daily_limit_hits="
                + repr(
                    [
                        record
                        for record in replay["_daily_records"]
                        if record["limit_hit"]
                    ]
                ),
                flush=True,
            )
            assignments = {
                row.battery: (
                    row.day if row.day <= horizon_end else None
                )
                for row in plan.itertuples()
            }
            for detail_row in detail.itertuples():
                if detail_row.expected_timing <= 0:
                    continue
                candidate_assignments = assignments.copy()
                candidate_assignments[detail_row.battery] = detail_row.best_day
                candidate = order_assignments(
                    candidate_assignments,
                    locs,
                    scenario["travel_costs"],
                    scenario["settings"].base_location,
                    horizon_end + pd.Timedelta("1D"),
                )
                candidate_expected = planner._expected_score(
                    candidate,
                    expected_costs,
                    due_sample,
                    locs,
                    scenario["travel_costs"],
                    scenario["settings"],
                    pd.Timestamp(scenario["start_time"]),
                )
                _, _, candidate_official = evaluate_plan(
                    candidate,
                    locs,
                    scenario["travel_costs"],
                    scenario["settings"],
                    eol_times=active_eol,
                    start_time=pd.Timestamp(scenario["start_time"]),
                    verbose=0,
                )
                print(
                    f"move={detail_row.battery} to={detail_row.best_day.date()} "
                    f"internal={candidate_expected:.6f} "
                    f"official={candidate_official['total_cost']:.6f}",
                    flush=True,
                )
        if not args.quiet:
            print(pd.Series(row).to_string(), flush=True)
            print("-" * 72, flush=True)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No scenarios were benchmarked")
    numeric = result.select_dtypes(include="number")
    total_elapsed = time.perf_counter() - started
    print("MEAN")
    print(numeric.mean().to_string())
    print(f"elapsed_seconds {total_elapsed:.3f}")

    if args.record is not None:
        record_benchmark(
            args.record,
            numeric.mean(),
            mode=args.mode,
            model_version=str(model_version),
            n_scenarios=len(result),
            elapsed_seconds=total_elapsed,
            label=args.label,
        )
        print(f"Recorded -> {args.record}")


if __name__ == "__main__":
    main()
