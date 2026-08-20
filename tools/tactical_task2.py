"""Run controlled Task 1 -> Task 2 experiments on labeled train scenarios.

This tool is deliberately separate from the submission entry point. It changes
one forecast sharpness parameter at a time and records enough detail to explain
score changes instead of treating aggregate total cost as a black box.
"""

from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
import json
import pickle
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset

from batteryswap_solution.forecast import RiskForecast
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import (
    CompetitionPlanner,
    PlannerConfig,
    infer_scenario_start,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)

FORECAST_OFFSETS = (0, 7, 14, 21, 28, 35, 42)


class FixedForecast:
    """Return one already-computed scenario forecast to avoid duplicate inference."""

    def __init__(self, forecast: RiskForecast) -> None:
        self.forecast = forecast
        self.model_version = forecast.metadata.model_version

    def predict(self, *args, **kwargs) -> RiskForecast:
        return self.forecast


def parse_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    indices = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not indices or min(indices) < 0:
        raise argparse.ArgumentTypeError("scenario indices must be non-negative integers")
    return indices


def all_defer_plan(
    locations: pd.DataFrame, start: pd.Timestamp, settings
) -> pd.DataFrame:
    defer_day = start.normalize() + pd.Timedelta(
        days=float(settings.planning_window_days) + 1.0
    )
    return pd.DataFrame(
        {
            "day": pd.DatetimeIndex([defer_day] * len(locations)),
            "battery": sorted(locations["battery"].astype(str)),
        }
    )


def override_physical_prior(
    forecaster,
    uncertainty_days: float,
    physical_risk_weight: float,
    physical_shape_min_remaining_days: float,
    horizon_rate_cap_multiplier: float,
    horizon_rate_activation_ratio: float,
    direct_horizon_weight: float,
):
    if not is_dataclass(forecaster) or not hasattr(
        forecaster, "physical_uncertainty_days"
    ):
        raise TypeError(
            "The selected forecaster does not expose physical_uncertainty_days"
        )
    horizon_rate_calibrator = getattr(forecaster, "horizon_rate_calibrator", None)
    if horizon_rate_calibrator is not None:
        horizon_rate_calibrator = replace(
            horizon_rate_calibrator,
            cap_multiplier=float(horizon_rate_cap_multiplier),
            activation_ratio=float(horizon_rate_activation_ratio),
        )
    return replace(
        forecaster,
        physical_uncertainty_days=float(uncertainty_days),
        physical_risk_weight=float(physical_risk_weight),
        physical_shape_min_remaining_days=float(physical_shape_min_remaining_days),
        horizon_rate_calibrator=horizon_rate_calibrator,
        direct_horizon_weight=float(direct_horizon_weight),
    )


def build_config(args: argparse.Namespace) -> PlannerConfig:
    return PlannerConfig(
        late_risk_multiplier=float(args.late_risk_multiplier),
        minimum_expected_improvement=float(args.minimum_expected_improvement),
        local_search_evaluations=int(args.local_search),
        uncertain_local_search_evaluations=int(args.uncertain_local_search),
        robust_emergency_samples=int(args.robust_samples),
        optimizer=OptimizationConfig(
            solver_seconds=float(args.solver_seconds),
            max_planned_rate=args.max_planned_rate,
        ),
    )


def compute_forecast(
    forecaster,
    cut: pd.DataFrame,
    locations: pd.DataFrame,
    scenario,
) -> RiskForecast:
    start = infer_scenario_start(cut)
    dates, _ = CompetitionPlanner._planning_clock(start, scenario["settings"])
    observation_end = pd.Timestamp(pd.to_datetime(locations["end_time"]).max())
    if observation_end.tzinfo is not None:
        observation_end = observation_end.tz_localize(None)
    return forecaster.predict(
        cut,
        locations,
        prediction_origin=start,
        horizon_days=len(dates) - 1,
        evaluation_observation_end=observation_end,
    )


def battery_diagnostics(
    *,
    experiment: str,
    uncertainty_days: float,
    physical_risk_weight: float,
    physical_shape_min_remaining_days: float,
    scenario_index: int,
    scenario,
    locations: pd.DataFrame,
    active_eol: pd.Series,
    plan: pd.DataFrame,
    forecast: RiskForecast,
) -> pd.DataFrame:
    settings = scenario["settings"]
    start = pd.Timestamp(scenario["start_time"])
    horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
    planned = plan.loc[plan["day"] <= horizon_end, ["battery", "day"]].copy()
    planned["battery"] = planned["battery"].astype(str)
    planned_days = planned.set_index("battery")["day"]

    loc = locations.copy()
    loc["battery"] = loc["battery"].astype(str)
    loc = loc.set_index("battery", drop=False)
    battery_ids = loc.index.tolist()
    observed_eol = pd.to_datetime(active_eol.reindex(battery_ids))
    proxy_eol = (
        pd.to_datetime(loc["end_time"])
        + pd.to_timedelta(float(settings.unobserved_eol_days), unit="D")
    ).dt.normalize()
    effective_eol = observed_eol.where(observed_eol.notna(), proxy_eol)

    curves = forecast.curves.pivot(
        index="battery_id", columns="forecast_date", values="failure_cdf"
    ).reindex(battery_ids)
    curves.columns = pd.DatetimeIndex(pd.to_datetime(curves.columns)).normalize()
    tail = forecast.tail.set_index("battery_id").reindex(battery_ids)
    summaries = forecast.summaries.set_index("battery_id").reindex(battery_ids)

    rows: list[dict] = []
    for battery_id in battery_ids:
        plan_day = planned_days.get(battery_id, pd.NaT)
        is_planned = pd.notna(plan_day)
        eol = observed_eol.loc[battery_id]
        due_in_horizon = bool(pd.notna(eol) and eol <= horizon_end)
        delta_days = np.nan
        planned_early = 0.0
        planned_late = 0.0
        if is_planned:
            delta_days = float(
                (effective_eol.loc[battery_id] - pd.Timestamp(plan_day))
                / pd.Timedelta(days=1)
            )
            planned_early = max(delta_days, 0.0) * float(
                settings.early_replacement_penalty_daily
            )
            planned_late = max(-delta_days, 0.0) * float(
                settings.late_replacement_penalty_daily
            )

        row = {
            "experiment": experiment,
            "physical_uncertainty_days": float(uncertainty_days),
            "physical_risk_weight": float(physical_risk_weight),
            "physical_shape_min_remaining_days": float(
                physical_shape_min_remaining_days
            ),
            "scenario_index": scenario_index,
            "scenario": scenario["name"],
            "scenario_start": start,
            "horizon_end": horizon_end,
            "battery": battery_id,
            "building": loc.loc[battery_id, "building"],
            "room": loc.loc[battery_id, "room"],
            "planned": bool(is_planned),
            "plan_day": plan_day,
            "observed_eol": eol,
            "effective_eol": effective_eol.loc[battery_id],
            "due_in_horizon": due_in_horizon,
            "planned_due": bool(is_planned and due_in_horizon),
            "planned_not_due": bool(is_planned and not due_in_horizon),
            "emergency": bool(due_in_horizon and not is_planned),
            "plan_to_effective_eol_days": delta_days,
            "planned_early_cost": planned_early,
            "planned_late_cost": planned_late,
            "forecast_observed_after_horizon": float(
                tail.loc[battery_id, "prob_observed_after_horizon"]
            ),
            "forecast_unobserved_eol": float(
                tail.loc[battery_id, "prob_unobserved_eol"]
            ),
            "forecast_q50_days": float(summaries.loc[battery_id, "q50_days"]),
            "forecast_cold_start": bool(summaries.loc[battery_id, "cold_start"]),
        }
        for offset in FORECAST_OFFSETS:
            forecast_day = start.normalize() + pd.Timedelta(days=offset)
            value = curves.loc[battery_id, forecast_day]
            row[f"failure_cdf_d{offset}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_diagnostics(
    *,
    experiment: str,
    uncertainty_days: float,
    physical_risk_weight: float,
    physical_shape_min_remaining_days: float,
    scenario_index: int,
    scenario,
    battery_rows: pd.DataFrame,
    score: pd.Series,
    all_defer_score: pd.Series,
    runtime_seconds: float,
    expected_improvement: float,
) -> dict:
    planned_swaps = int(battery_rows["planned"].sum())
    actual_due = int(battery_rows["due_in_horizon"].sum())
    planned_due = int(battery_rows["planned_due"].sum())
    emergency_swaps = int(battery_rows["emergency"].sum())
    precision = planned_due / planned_swaps if planned_swaps else np.nan
    recall = planned_due / actual_due if actual_due else np.nan
    row = {
        "experiment": experiment,
        "physical_uncertainty_days": float(uncertainty_days),
        "physical_risk_weight": float(physical_risk_weight),
        "physical_shape_min_remaining_days": float(
            physical_shape_min_remaining_days
        ),
        "scenario_index": scenario_index,
        "scenario": scenario["name"],
        "scenario_start": pd.Timestamp(scenario["start_time"]),
        "active_batteries": int(len(battery_rows)),
        "actual_due": actual_due,
        "planned_swaps": planned_swaps,
        "planned_due": planned_due,
        "planned_not_due": int(battery_rows["planned_not_due"].sum()),
        "emergency_swaps": emergency_swaps,
        "due_recall": recall,
        "planned_precision": precision,
        "runtime_seconds": float(runtime_seconds),
        "expected_improvement": float(expected_improvement),
        "all_defer_total_cost": float(all_defer_score["total_cost"]),
    }
    row.update({component: float(score[component]) for component in cost_components})
    row["timing_cost"] = row["late_swap"] + row["early_swap"]
    row["capacity_cost"] = (
        row["overtime"] + row["daily_limit"] + row["weekly_limit"]
    )
    row["operational_cost"] = float(score["total_cost"]) - row["timing_cost"]
    row["total_cost"] = float(score["total_cost"])
    return row


def aggregate_summary(scenarios: pd.DataFrame) -> dict:
    metrics = [
        "total_cost",
        "timing_cost",
        "operational_cost",
        "early_swap",
        "late_swap",
        "daily_limit",
        "weekly_limit",
        "planned_swaps",
        "emergency_swaps",
        "due_recall",
        "planned_precision",
        "runtime_seconds",
    ]
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = scenarios[metric].astype(float)
        summary[metric] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p90": float(values.quantile(0.9)),
            "max": float(values.max()),
        }
    return summary


def run_experiment(
    *,
    args: argparse.Namespace,
    uncertainty_days: float,
    physical_risk_weight: float,
    locations: pd.DataFrame,
    timeseries: pd.DataFrame,
    eol_times: pd.Series,
    scenarios: list,
    base_forecaster,
    selected_indices: set[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cap_label = "none" if args.max_planned_rate is None else f"{args.max_planned_rate:g}"
    experiment = (
        f"{args.run_name}-u{uncertainty_days:g}-w{physical_risk_weight:g}"
        f"-h{args.horizon_rate_cap_multiplier:g}"
        f"-a{args.horizon_rate_activation_ratio:g}"
        f"-d{args.direct_horizon_weight:g}"
        f"-cap{cap_label}"
    )
    forecaster = override_physical_prior(
        base_forecaster,
        uncertainty_days,
        physical_risk_weight,
        args.physical_shape_min_remaining_days,
        args.horizon_rate_cap_multiplier,
        args.horizon_rate_activation_ratio,
        args.direct_horizon_weight,
    )
    config = build_config(args)
    scenario_rows: list[dict] = []
    battery_frames: list[pd.DataFrame] = []
    experiment_started = time.perf_counter()
    last_selected_index = max(selected_indices) if selected_indices is not None else None

    generator = iterate_scenarios(locations, timeseries, eol_times, scenarios)
    for scenario_index, (scenario, locs, cut, active_eol) in enumerate(generator):
        if last_selected_index is not None and scenario_index > last_selected_index:
            break
        if selected_indices is not None and scenario_index not in selected_indices:
            continue

        scenario_started = time.perf_counter()
        forecast = compute_forecast(forecaster, cut, locs, scenario)
        planner = CompetitionPlanner(
            forecaster=FixedForecast(forecast),
            config=config,
        )
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
        defer = all_defer_plan(
            locs, pd.Timestamp(scenario["start_time"]), scenario["settings"]
        )
        _, _, defer_score = evaluate_plan(
            defer,
            locs,
            scenario["travel_costs"],
            scenario["settings"],
            eol_times=active_eol,
            start_time=pd.Timestamp(scenario["start_time"]),
            verbose=0,
        )
        battery_rows = battery_diagnostics(
            experiment=experiment,
            uncertainty_days=uncertainty_days,
            physical_risk_weight=physical_risk_weight,
            physical_shape_min_remaining_days=args.physical_shape_min_remaining_days,
            scenario_index=scenario_index,
            scenario=scenario,
            locations=locs,
            active_eol=active_eol,
            plan=plan,
            forecast=forecast,
        )
        runtime_seconds = time.perf_counter() - scenario_started
        row = scenario_diagnostics(
            experiment=experiment,
            uncertainty_days=uncertainty_days,
            physical_risk_weight=physical_risk_weight,
            physical_shape_min_remaining_days=args.physical_shape_min_remaining_days,
            scenario_index=scenario_index,
            scenario=scenario,
            battery_rows=battery_rows,
            score=score,
            all_defer_score=defer_score,
            runtime_seconds=runtime_seconds,
            expected_improvement=planner.last_expected_improvement,
        )
        scenario_rows.append(row)
        battery_frames.append(battery_rows)
        if not args.quiet:
            print(
                f"{experiment} {scenario['name']} total={row['total_cost']:.3f} "
                f"planned={row['planned_swaps']} emergency={row['emergency_swaps']} "
                f"early={row['early_swap']:.3f} late={row['late_swap']:.3f} "
                f"seconds={runtime_seconds:.2f}",
                flush=True,
            )

    if not scenario_rows:
        raise RuntimeError("No scenarios matched --scenario-indices")
    scenario_frame = pd.DataFrame(scenario_rows).sort_values("scenario_index")
    battery_frame = pd.concat(battery_frames, ignore_index=True)
    summary = {
        "experiment": experiment,
        "model_version": str(forecaster.model_version),
        "incidence_weighting": str(
            getattr(forecaster, "incidence_weighting", "legacy")
        ),
        "physical_uncertainty_days": float(uncertainty_days),
        "physical_risk_weight": float(physical_risk_weight),
        "physical_shape_min_remaining_days": float(
            args.physical_shape_min_remaining_days
        ),
        "scenario_indices": scenario_frame["scenario_index"].astype(int).tolist(),
        "scenario_count": int(len(scenario_frame)),
        "elapsed_seconds": float(time.perf_counter() - experiment_started),
        "planner_config": {
            "late_risk_multiplier": float(args.late_risk_multiplier),
            "minimum_expected_improvement": float(args.minimum_expected_improvement),
            "solver_seconds": float(args.solver_seconds),
            "local_search": int(args.local_search),
            "uncertain_local_search": int(args.uncertain_local_search),
            "robust_samples": int(args.robust_samples),
            "horizon_rate_cap_multiplier": float(
                args.horizon_rate_cap_multiplier
            ),
            "horizon_rate_activation_ratio": float(
                args.horizon_rate_activation_ratio
            ),
            "max_planned_rate": args.max_planned_rate,
            "direct_horizon_weight": float(args.direct_horizon_weight),
        },
        "metrics": aggregate_summary(scenario_frame),
    }
    return scenario_frame, battery_frame, summary


def write_results(
    output_root: Path,
    experiment: str,
    scenarios: pd.DataFrame,
    batteries: pd.DataFrame,
    summary: dict,
) -> Path:
    run_dir = output_root / experiment
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Experiment output already exists: {run_dir}. Use a new --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(run_dir / "scenarios.csv", index=False)
    batteries.to_csv(run_dir / "batteries.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset/train"))
    parser.add_argument(
        "--forecaster-path",
        type=Path,
        default=Path("models/risk_forecaster.pkl"),
    )
    parser.add_argument(
        "--physical-uncertainty-days",
        type=float,
        nargs="+",
        default=[20.0],
    )
    parser.add_argument(
        "--physical-risk-weight",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument(
        "--physical-shape-min-remaining-days", type=float, default=0.0
    )
    parser.add_argument("--horizon-rate-cap-multiplier", type=float, default=1.0)
    parser.add_argument("--horizon-rate-activation-ratio", type=float, default=1.9)
    parser.add_argument("--direct-horizon-weight", type=float, default=0.0)
    parser.add_argument("--max-planned-rate", type=float)
    parser.add_argument(
        "--scenario-indices",
        help="Comma-separated zero-based indices. Omit to run all scenarios.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tactical"))
    parser.add_argument("--solver-seconds", type=float, default=2.0)
    parser.add_argument("--local-search", type=int, default=160)
    parser.add_argument("--uncertain-local-search", type=int, default=70)
    parser.add_argument("--robust-samples", type=int, default=4)
    parser.add_argument("--late-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--minimum-expected-improvement", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if any(value <= 0 for value in args.physical_uncertainty_days):
        parser.error("--physical-uncertainty-days values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.physical_risk_weight):
        parser.error("--physical-risk-weight values must be between 0 and 1")
    if args.horizon_rate_cap_multiplier <= 0.0:
        parser.error("--horizon-rate-cap-multiplier must be positive")
    if args.horizon_rate_activation_ratio < 1.0:
        parser.error("--horizon-rate-activation-ratio must be at least 1.0")
    if not 0.0 <= args.direct_horizon_weight <= 1.0:
        parser.error("--direct-horizon-weight must be between 0 and 1")
    if args.max_planned_rate is not None and not 0.0 < args.max_planned_rate <= 1.0:
        parser.error("--max-planned-rate must be in (0, 1]")
    selected_indices = parse_indices(args.scenario_indices)
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    with args.forecaster_path.open("rb") as handle:
        base_forecaster = pickle.load(handle)

    for uncertainty_days in args.physical_uncertainty_days:
        for physical_risk_weight in args.physical_risk_weight:
            scenarios_frame, batteries_frame, summary = run_experiment(
                args=args,
                uncertainty_days=uncertainty_days,
                physical_risk_weight=physical_risk_weight,
                locations=locations,
                timeseries=timeseries,
                eol_times=eol_times,
                scenarios=scenarios,
                base_forecaster=base_forecaster,
                selected_indices=selected_indices,
            )
            experiment = summary["experiment"]
            run_dir = write_results(
                args.output_dir,
                experiment,
                scenarios_frame,
                batteries_frame,
                summary,
            )
            mean = summary["metrics"]["total_cost"]["mean"]
            p90 = summary["metrics"]["total_cost"]["p90"]
            print(
                f"completed {experiment}: mean={mean:.3f} p90={p90:.3f} "
                f"output={run_dir}",
                flush=True,
            )


if __name__ == "__main__":
    main()
