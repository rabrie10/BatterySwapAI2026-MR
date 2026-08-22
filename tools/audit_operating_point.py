"""Per-battery error ledger at the shipping operating point.

Operating point: censored model (outputs/v8_folds_cens.joblib), robust-samples 0,
local-search 240, due-multiplier 1.6 buffer 1, max-planned 15, late-multiplier 1.8,
capacity post-pass on. One planner run per scenario; everything else is exact
arithmetic on the captured cost tables and the evaluator's own transition log.

Produces, per (scenario, battery):
  * realized early/late cost, swap day, due/served status (exact, from
    ``evaluate_plan`` transitions -- includes the emergency-queue lateness);
  * the planner's economics at late-multiplier 1.8 AND 1.0 from the same
    forecast: standalone gain, candidate-filter membership, economics rank,
    the binding slot limit, best service day;
  * exclusion reason for every missed due battery (invisible / uneconomic /
    outranked-by-the-cap / dropped-by-search) and early-cost class for every
    swap (due-early / post-window-due / never-due).

The late-multiplier mechanism is decomposed by rebuilding the identical greedy
slot-filling under both multipliers from the captured tables: which batteries
enter the capped set, and how the timing-optimal day moves for the common ones.

    python tools/audit_operating_point.py --limit 2   # smoke
    python tools/audit_operating_point.py             # full 48, ~10 min
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

from batteryswap_solution import planner as planner_module
from batteryswap_solution.costs import (
    build_expected_cost_tables,
    isolated_emergency_costs,
    select_candidates,
)
from batteryswap_solution.optimizer import OptimizationConfig, scenario_planned_swap_limit
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel

WORK_HOURS = 0.25
INVISIBLE_P = 0.02


class FallbackCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1
        print(f"  !! planner fallback: {record.getMessage()}", flush=True)


def economics(table, probability: np.ndarray, emergency: np.ndarray) -> dict[str, np.ndarray]:
    """The standalone swap-vs-defer arithmetic the optimizer sees."""
    best_service = table.service_cost.min(axis=1) + WORK_HOURS
    best_day = table.service_cost.argmin(axis=1)
    defer_full = table.defer_cost + probability * emergency
    gain = defer_full - best_service
    # select_candidates uses the timing-only defer cost and no work hours.
    filter_gain = table.defer_cost - table.service_cost.min(axis=1)
    return {
        "best_service": best_service,
        "best_day": best_day.astype(float),
        "defer_full": defer_full,
        "gain": gain,
        "filter_gain": filter_gain,
    }


def greedy_set(gain: np.ndarray, keep: np.ndarray, limit: int | None) -> list[int]:
    """Positive-gain candidates, best first, truncated at the slot limit."""
    candidates = [int(i) for i in keep if gain[i] > 0.0]
    candidates.sort(key=lambda i: -gain[i])
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v8_folds_cens.joblib"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--late-multiplier", type=float, default=1.8)
    parser.add_argument("--alt-late-multiplier", type=float, default=1.0)
    parser.add_argument("--local-search", type=int, default=240)
    parser.add_argument("--robust-samples", type=int, default=0)
    parser.add_argument("--due-multiplier", type=float, default=1.6)
    parser.add_argument("--due-buffer", type=float, default=1.0)
    parser.add_argument("--max-planned", type=int, default=15)
    parser.add_argument("--solver-seconds", type=float, default=1.0)
    parser.add_argument("--ledger", type=Path, default=Path("outputs/audit_ledger.csv"))
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/audit_operating_point.json")
    )
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
    forecaster.rank_calibration = bundle.get("rank_calibration")

    optimizer_config = OptimizationConfig(
        solver_seconds=args.solver_seconds,
        expected_due_multiplier=args.due_multiplier,
        expected_due_buffer=args.due_buffer,
        max_planned_count=args.max_planned,
    )
    planner = CompetitionPlanner(
        forecaster=forecaster,
        config=PlannerConfig(
            late_risk_multiplier=args.late_multiplier,
            local_search_evaluations=args.local_search,
            robust_emergency_samples=args.robust_samples,
            optimizer=optimizer_config,
        ),
    )

    fallbacks = FallbackCounter()
    logging.getLogger("batteryswap_solution.planner").addHandler(fallbacks)

    # Capture the cost tables plan() builds so the alternative-multiplier
    # tables can be computed from the identical forecast without re-planning.
    captured: dict[str, object] = {}
    real_build = planner_module.build_expected_cost_tables

    def capturing_build(forecast, locations, settings, dates, **kwargs):
        table = real_build(forecast, locations, settings, dates, **kwargs)
        captured["forecast"] = forecast
        captured["dates"] = dates
        captured["table"] = table
        return table

    planner_module.build_expected_cost_tables = capturing_build

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    ledger_rows: list[dict] = []
    scenario_rows: list[dict] = []
    started = time.time()

    try:
        for index, (scenario, locs, cut, not_dead) in enumerate(
            iterate_scenarios(locations, timeseries, eol_times, scenarios)
        ):
            if args.limit is not None and index >= args.limit:
                break
            start = pd.Timestamp(scenario["start_time"]).normalize()
            settings = scenario["settings"]
            travel = scenario["travel_costs"]
            horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))

            began = time.time()
            plan = planner.plan(cut, locs, travel, settings)
            plan_seconds = time.time() - began

            table_hi = captured["table"]
            forecast = captured["forecast"]
            dates = captured["dates"]
            table_lo = real_build(
                forecast,
                locs,
                settings,
                dates,
                late_risk_multiplier=args.alt_late_multiplier,
                emergency_rank_scale=1.0,
            )
            battery_ids = list(table_hi.battery_ids)
            position_of = {battery: i for i, battery in enumerate(battery_ids)}
            probability = table_hi.horizon_event_probability

            emergency = isolated_emergency_costs(locs, travel, settings, table_hi.battery_ids)
            econ_hi = economics(table_hi, probability, emergency)
            econ_lo = economics(table_lo, probability, emergency)

            keep_hi = select_candidates(table_hi)
            keep_lo = select_candidates(table_lo)
            limit_hi = scenario_planned_swap_limit(table_hi.take(keep_hi), optimizer_config)
            limit_lo = scenario_planned_swap_limit(table_lo.take(keep_lo), optimizer_config)
            due_budget = int(
                np.ceil(
                    args.due_multiplier
                    * float(table_hi.take(keep_hi).horizon_event_probability.sum())
                    + args.due_buffer
                )
            )

            rank_hi = np.full(len(battery_ids), np.nan)
            order = sorted(
                (int(i) for i in keep_hi), key=lambda i: -econ_hi["gain"][i]
            )
            for rank, i in enumerate(order, start=1):
                rank_hi[i] = rank

            greedy_hi = greedy_set(econ_hi["gain"], keep_hi, limit_hi)
            greedy_lo = greedy_set(econ_lo["gain"], keep_lo, limit_lo)

            # Exact realized costs per battery from the evaluator's own log.
            transitions, _, overall = evaluate_plan(
                plan,
                locs,
                travel,
                settings,
                eol_times=not_dead,
                start_time=start,
                verbose=0,
            )
            realized: dict[str, dict] = {}
            for transition in transitions:
                action = transition["action"]
                if action["name"] != "swap-battery":
                    continue
                battery = str(action["battery"])
                day = pd.Timestamp(transition["state"]["day"])
                realized[battery] = {
                    "swap_day": day,
                    "late": float(transition["costs"].get("late_swap", 0.0)),
                    "early": float(transition["costs"].get("early_swap", 0.0)),
                }

            plan_days = plan.set_index(plan["battery"].astype(str))["day"]
            served = set(
                plan.loc[pd.to_datetime(plan["day"]) <= horizon_end, "battery"].astype(str)
            )
            planned_count = len(served)
            due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)
            weakest_planned_gain = min(
                (
                    float(econ_hi["gain"][position_of[b]])
                    for b in served
                    if b in position_of
                ),
                default=float("nan"),
            )
            cap_bound = limit_hi is not None and planned_count >= limit_hi

            end_times = pd.to_datetime(locs.set_index(locs["battery"].astype(str))["end_time"])
            if getattr(end_times.dt, "tz", None) is not None:
                end_times = end_times.dt.tz_localize(None)

            greedy_hi_set = {battery_ids[i] for i in greedy_hi}
            greedy_lo_set = {battery_ids[i] for i in greedy_lo}
            keep_hi_set = {battery_ids[int(i)] for i in keep_hi}
            keep_lo_set = {battery_ids[int(i)] for i in keep_lo}

            for i, battery in enumerate(battery_ids):
                eol = not_dead.get(battery, pd.NaT)
                observed = pd.notna(eol)
                is_due = battery in due
                is_served = battery in served
                record = realized.get(battery)
                interesting = (
                    is_due
                    or record is not None
                    or battery in keep_hi_set
                    or battery in greedy_lo_set
                )
                if not interesting:
                    continue
                end_time = end_times.loc[battery]
                x_days = float(
                    (end_time.normalize() + pd.Timedelta(days=30) - horizon_end)
                    / pd.Timedelta(days=1)
                )
                swap_day = record["swap_day"] if record else pd.NaT
                late_cost = record["late"] if record else 0.0
                early_cost = record["early"] if record else 0.0

                miss_class = ""
                if is_due:
                    if is_served:
                        miss_class = "TIMING" if late_cost > 0 else "HIT-ON-TIME"
                    elif float(probability[i]) < INVISIBLE_P:
                        miss_class = "INVISIBLE"
                    elif float(econ_hi["gain"][i]) <= 0.0:
                        miss_class = "VISIBLE-UNECONOMIC"
                    elif limit_hi is not None and rank_hi[i] > limit_hi:
                        miss_class = "VISIBLE-OUTRANKED"
                    else:
                        miss_class = "VISIBLE-DROPPED"

                early_class = ""
                if record is not None and early_cost > 0:
                    if not observed:
                        early_class = "NEVER-DUE"
                    elif is_due:
                        early_class = "DUE-EARLY"
                    else:
                        early_class = "POST-WINDOW-DUE"

                ledger_rows.append(
                    {
                        "scenario": scenario["name"],
                        "scenario_index": index,
                        "battery": battery,
                        "building": building_of.get(battery, ""),
                        "p": float(probability[i]),
                        "x_days": x_days,
                        "eol_observed": bool(observed),
                        "days_to_eol": float((eol - start) / pd.Timedelta(days=1))
                        if observed
                        else np.nan,
                        "due": is_due,
                        "served": is_served,
                        "swap_offset": float((swap_day - start) / pd.Timedelta(days=1))
                        if record
                        else np.nan,
                        "late_cost": late_cost,
                        "early_cost": early_cost,
                        "emergency": record is not None and not is_served,
                        "gain_hi": float(econ_hi["gain"][i]),
                        "gain_lo": float(econ_lo["gain"][i]),
                        "best_day_hi": float(econ_hi["best_day"][i]),
                        "best_day_lo": float(econ_lo["best_day"][i]),
                        "econ_rank_hi": float(rank_hi[i]),
                        "in_filter_hi": battery in keep_hi_set,
                        "in_filter_lo": battery in keep_lo_set,
                        "in_greedy_hi": battery in greedy_hi_set,
                        "in_greedy_lo": battery in greedy_lo_set,
                        "slot_limit": limit_hi if limit_hi is not None else -1,
                        "due_budget": due_budget,
                        "planned_count": planned_count,
                        "cap_bound": cap_bound,
                        "weakest_planned_gain": weakest_planned_gain,
                        "miss_class": miss_class,
                        "early_class": early_class,
                    }
                )

            common = greedy_hi_set & greedy_lo_set
            day_shift = [
                float(
                    econ_lo["best_day"][position_of[b]]
                    - econ_hi["best_day"][position_of[b]]
                )
                for b in common
            ]
            entry = {component: float(overall[component]) for component in cost_components}
            entry.update(
                scenario=scenario["name"],
                total_cost=float(overall["total_cost"]),
                served=planned_count,
                due=len(due),
                hit=len(served & due),
                missed=len(due - served),
                slot_limit=limit_hi,
                slot_limit_lo=limit_lo,
                due_budget=due_budget,
                cap_bound=bool(cap_bound),
                greedy_hi=len(greedy_hi),
                greedy_lo=len(greedy_lo),
                greedy_common=len(common),
                greedy_hi_due=sum(1 for b in greedy_hi_set if b in due),
                greedy_lo_due=sum(1 for b in greedy_lo_set if b in due),
                greedy_enter=sorted(greedy_hi_set - greedy_lo_set),
                greedy_leave=sorted(greedy_lo_set - greedy_hi_set),
                mean_best_day_shift=float(np.mean(day_shift)) if day_shift else 0.0,
                served_in_greedy_hi=len(served & greedy_hi_set),
                seconds=round(plan_seconds, 2),
            )
            scenario_rows.append(entry)
            print(
                f"  {scenario['name']:>5}  total={entry['total_cost']:9.1f}  served={planned_count:3d}"
                f"  hit={entry['hit']:2d}/{entry['due']:2d}  limit={limit_hi}"
                f"  greedy {len(greedy_lo)}->{len(greedy_hi)} (common {len(common)},"
                f" due {entry['greedy_lo_due']}->{entry['greedy_hi_due']})"
                f"  shift={entry['mean_best_day_shift']:+.1f}d  {plan_seconds:5.1f}s",
                flush=True,
            )
    finally:
        planner_module.build_expected_cost_tables = real_build

    if fallbacks.count:
        raise SystemExit(f"{fallbacks.count} planner fallbacks; ledger is invalid")

    ledger = pd.DataFrame(ledger_rows)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.ledger, index=False)

    frame = pd.DataFrame(scenario_rows)
    summary = {
        "n_scenarios": int(len(frame)),
        "mean_total_cost": round(float(frame["total_cost"].mean()), 2),
        "components": {
            component: round(float(frame[component].mean()), 2)
            for component in cost_components
        },
        "served_per_scenario": round(float(frame["served"].mean()), 2),
        "missed_per_scenario": round(float(frame["missed"].mean()), 2),
        "recall": round(float(frame["hit"].sum() / max(frame["due"].sum(), 1)), 3),
        "cap_bound_scenarios": int(frame["cap_bound"].sum()),
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "scenarios": scenario_rows}, indent=2)
    )
    print(f"ledger: {args.ledger} ({len(ledger)} rows)")


if __name__ == "__main__":
    main()
