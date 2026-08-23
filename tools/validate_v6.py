"""Score the V6 pipeline end to end on the train scenarios.

The only number allowed to justify a submission is this one: the production
planner, driven by predictions from a model that never saw the device's own
building, scored by the official ``evaluate_plan``.

Anchors are printed on every run so a result is never read in isolation:

    all-defer          3324.7   servicing nothing at all
    naive oracle        205.2   perfect labels, greedy router
    planner oracle       77.8   perfect labels, this planner (scenarios 0-11)

Differences under about 100 on the 48-scenario mean are noise. The scenarios
start a week apart and each covers six weeks, so adjacent windows overlap by
roughly 85% and the effective sample size is nearer eight than forty-eight.
``--blocks`` reports non-overlapping blocks for that reason.

    python tools/validate_v6.py
    python tools/validate_v6.py --production   # in-fold, diagnostic only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from bsai.simple_planner import PerBatteryPlanner, SimplePlannerConfig
from bsai.validation import OofHazardModel

ALL_DEFER_ANCHOR = 3324.7
NAIVE_ORACLE_ANCHOR = 205.2


class FallbackCounter(logging.Handler):
    """Count planner fallbacks.

    ``CompetitionPlanner.plan`` catches everything and returns an all-defer
    plan, which is the right behaviour in a submission and a terrible one in
    validation: a genuine crash then reads as a merely mediocre score. A
    renamed parameter once turned every scenario into all-defer this way, and
    the run still finished and reported a number.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        print(f"  !! planner fallback: {record.getMessage()}", flush=True)


def build_forecaster(args, building_of: dict[str, str]) -> HazardForecaster:
    if args.production:
        model = joblib.load(args.model)
        return HazardForecaster(model, probability_scale=args.probability_scale)
    bundle = joblib.load(args.folds)
    if args.volatility_scale is not None:
        for fold_model in bundle["by_building"].values():
            fold_model.volatility_scale = args.volatility_scale
    model = OofHazardModel(
        by_building=bundle["by_building"],
        building_of=building_of,
        climatology=bundle["climatology"],
    )
    return HazardForecaster(model, probability_scale=args.probability_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--model", type=Path, default=Path("models/v6_hazard.joblib"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v6_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/v6_validation.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver-seconds", type=float, default=1.0)
    parser.add_argument("--local-search", type=int, default=80)
    parser.add_argument("--uncertain-search", type=int, default=35)
    parser.add_argument(
        "--production",
        action="store_true",
        help="use the shipped model instead of out-of-fold; in-fold and optimistic",
    )
    parser.add_argument("--late-multiplier", type=float, default=1.0)
    parser.add_argument("--candidate-margin", type=float, default=24.0)
    parser.add_argument("--emergency-rank-scale", type=float, default=1.0)
    parser.add_argument("--prune", type=int, default=0,
                        help="evaluations spent offering the weakest swaps for removal")
    parser.add_argument(
        "--move-order",
        choices=("legacy", "interleaved"),
        default="interleaved",
        help="how the local search spends its evaluation budget; see PlannerConfig",
    )
    parser.add_argument("--max-planned-rate", type=float, default=None)
    parser.add_argument("--probability-scale", type=float, default=1.0)
    parser.add_argument(
        "--volatility-scale",
        type=float,
        default=None,
        help="override the Wiener volatility scale; the level must be calibrated "
        "on the scenario population, not the training cutoffs",
    )
    parser.add_argument("--per-battery", action="store_true",
                        help="decide each battery on its own cost; no joint search")
    parser.add_argument("--work-cost", type=float, default=0.25)
    parser.add_argument("--no-emergency-ops", action="store_true",
                        help="price deferral on lateness alone, as the V5 prototype did")
    parser.add_argument("--capacity-roundtrip", type=float, default=1.0)
    parser.add_argument("--max-daily-factor", type=float, default=2.0,
                        help="hard bound on one day's work, as a multiple of the "
                             "24-hour daily limit; 1.0 forbids two far round trips")
    parser.add_argument("--greedy", action="store_true", help="skip CP-SAT")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="record the predicted probability of every genuinely due battery",
    )
    parser.add_argument("--blocks", type=int, default=6, help="non-overlapping blocks")
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))

    forecaster = build_forecaster(args, building_of)
    if args.per_battery:
        planner = PerBatteryPlanner(
            forecaster=forecaster,
            config=SimplePlannerConfig(
                late_risk_multiplier=args.late_multiplier,
                emergency_rank_scale=args.emergency_rank_scale,
                work_cost_hours=args.work_cost,
                include_emergency_operations=not args.no_emergency_ops,
            ),
        )
    else:
        planner = CompetitionPlanner(
            forecaster=forecaster,
            config=PlannerConfig(
                late_risk_multiplier=args.late_multiplier,
                local_search_evaluations=args.local_search,
                uncertain_local_search_evaluations=args.uncertain_search,
                candidate_margin_hours=args.candidate_margin,
                emergency_rank_scale=args.emergency_rank_scale,
                move_order=args.move_order,
                prune_evaluations=args.prune,
                optimizer=OptimizationConfig(
                    solver_seconds=args.solver_seconds,
                    capacity_roundtrip_fraction=args.capacity_roundtrip,
                    max_daily_hours_factor=args.max_daily_factor,
                    max_planned_rate=args.max_planned_rate,
                    use_cp_sat=not args.greedy,
                ),
            ),
        )

    fallbacks = FallbackCounter()
    logging.getLogger("batteryswap_solution.planner").addHandler(fallbacks)
    logging.getLogger("bsai.simple_planner").addHandler(fallbacks)

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    audit: list[dict] = []
    started = time.time()

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if args.limit is not None and index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"])
        settings = scenario["settings"]
        horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
        began = time.time()
        plan = planner.plan(cut, locs, scenario["travel_costs"], settings)
        elapsed = time.time() - began

        _, _, scores = evaluate_plan(
            plan,
            locs,
            scenario["travel_costs"],
            settings,
            eol_times=not_dead,
            start_time=start,
            verbose=0,
        )

        served = set(plan.loc[plan["day"] <= horizon_end, "battery"].astype(str))
        due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)
        if args.audit:
            probabilities = forecaster.last_probabilities
            for battery in sorted(due):
                audit.append(
                    {
                        "scenario": scenario["name"],
                        "battery": battery,
                        "probability": round(
                            float(probabilities.get(battery, float("nan"))), 4
                        ),
                        "days_to_eol": int((not_dead[battery] - start).days),
                        "served": battery in served,
                    }
                )
        entry = {component: float(scores[component]) for component in cost_components}
        entry.update(
            scenario=scenario["name"],
            total_cost=float(scores["total_cost"]),
            served=len(served),
            due=len(due),
            hit=len(served & due),
            missed=len(due - served),
            cold_start=int(forecaster.last_cold_start),
            expected_due=round(float(forecaster.last_expected_due), 2),
            seconds=round(elapsed, 2),
        )
        rows.append(entry)
        print(
            f"  {scenario['name']:>5}  total={entry['total_cost']:9.1f}  "
            f"served={entry['served']:3d}  due={entry['due']:3d}  "
            f"missed={entry['missed']:3d}  cold={entry['cold_start']:3d}  "
            f"{elapsed:5.1f}s",
            flush=True,
        )

    if fallbacks.count:
        raise SystemExit(
            f"{fallbacks.count} scenarios fell back to all-defer; "
            "fix the planner before reading any score from this run"
        )

    frame = pd.DataFrame(rows)
    summary = _summarise(frame, args.blocks)
    if audit:
        summary["due_battery_audit"] = _audit_summary(pd.DataFrame(audit))
    print()
    print(json.dumps(summary, indent=2))
    print(
        f"\nanchors: all-defer {ALL_DEFER_ANCHOR}, naive oracle {NAIVE_ORACLE_ANCHOR}, "
        f"planner oracle 77.8 (scenarios 0-11 only)"
    )
    print(f"total wall time {time.time() - started:.0f}s")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "mode": "production-in-fold" if args.production else "out-of-fold",
                "summary": summary,
                "scenarios": rows,
                "audit": audit,
            },
            indent=2,
        )
    )


def _summarise(frame: pd.DataFrame, blocks: int) -> dict:
    total = frame["total_cost"]
    summary = {
        "n_scenarios": int(len(frame)),
        "mean_total_cost": round(float(total.mean()), 2),
        "median_total_cost": round(float(total.median()), 2),
        "p90_total_cost": round(float(total.quantile(0.9)), 2),
        "max_total_cost": round(float(total.max()), 2),
        "components": {
            component: round(float(frame[component].mean()), 2)
            for component in cost_components
        },
        "decisions": {
            "served_per_scenario": round(float(frame["served"].mean()), 2),
            "due_per_scenario": round(float(frame["due"].mean()), 2),
            "missed_per_scenario": round(float(frame["missed"].mean()), 2),
            "cold_start_per_scenario": round(float(frame["cold_start"].mean()), 1),
            "expected_due_per_scenario": round(float(frame["expected_due"].mean()), 2),
            "recall": round(float(frame["hit"].sum() / max(frame["due"].sum(), 1)), 3),
            "precision": round(float(frame["hit"].sum() / max(frame["served"].sum(), 1)), 3),
        },
        "runtime": {
            "mean_seconds_per_scenario": round(float(frame["seconds"].mean()), 2),
            "max_seconds_per_scenario": round(float(frame["seconds"].max()), 2),
            # Public plus private is 96 scenarios; the harness itself costs a
            # further 68 seconds per split before our code runs.
            "projected_minutes_for_96": round(
                float(frame["seconds"].mean() * 96 + 136) / 60, 1
            ),
        },
    }
    if blocks > 1 and len(frame) >= blocks:
        edges = np.array_split(np.arange(len(frame)), blocks)
        means = [float(total.iloc[block].mean()) for block in edges]
        summary["block_means"] = [round(value, 1) for value in means]
        summary["block_mean_sd"] = round(float(np.std(means, ddof=1)), 1)
    return summary


def _audit_summary(frame: pd.DataFrame) -> dict:
    """Were the misses low-probability, or did the optimizer skip them?

    If missed batteries carry decent predicted probability, the decision layer
    is at fault. If they sit near zero, the model never saw them coming and no
    scheduling knob can help.
    """
    served, missed = frame[frame["served"]], frame[~frame["served"]]
    out = {
        "n_due": int(len(frame)),
        "n_missed": int(len(missed)),
        "median_probability_served": round(float(served["probability"].median()), 4)
        if len(served)
        else None,
        "median_probability_missed": round(float(missed["probability"].median()), 4)
        if len(missed)
        else None,
    }
    if len(missed):
        for threshold in (0.02, 0.05, 0.2, 0.5):
            out[f"missed_with_probability_below_{threshold}"] = int(
                (missed["probability"] < threshold).sum()
            )
    return out


if __name__ == "__main__":
    main()
