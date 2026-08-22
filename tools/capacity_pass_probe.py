"""One-off diagnostic: what does the capacity post-pass see on one scenario?

Runs the production pipeline up to the local search for a single scenario,
then replays the incumbent with details and prints per-day hours, week-bucket
hours, and the delta of every candidate move the post-pass would evaluate.

    python tools/capacity_pass_probe.py --scenario s_4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from batteryswap_solution.costs import build_expected_cost_tables, select_candidates
from batteryswap_solution.optimizer import OptimizationConfig, optimize_assignments
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import build_replay_context, replay_operational_cost

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v8_folds_cens.joblib"))
    parser.add_argument("--scenario", type=str, default="s_4")
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
    for scenario, locs, cut, not_dead in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        if scenario["name"] != args.scenario:
            continue
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
        id_column = "battery_id" if "battery_id" in locs else "battery"
        candidate_locations = locs[locs[id_column].astype(str).isin(candidate_ids)]
        seeds = [
            optimize_assignments(
                costs, candidate_locations, travel_costs, settings,
                config=config.optimizer,
            )
        ]
        plan = planner._local_search(
            seeds, costs, candidate_locations, travel_costs, settings, start, defer_day
        )

        context = build_replay_context(candidate_locations, travel_costs, settings, start)
        details = replay_operational_cost(
            plan, candidate_locations, travel_costs, settings, start,
            include_details=True, context=context,
        )
        loc = candidate_locations.copy().set_index(
            candidate_locations["battery"].astype(str)
        )
        building_column = "building" if "building" in loc else "building_id"
        print(f"== incumbent after local search: operational {details['total_cost']:.2f}")
        by_day: dict[pd.Timestamp, list[str]] = {}
        for row in plan.itertuples(index=False):
            day = pd.Timestamp(row.day).normalize()
            if day in set(costs.candidate_dates):
                by_day.setdefault(day, []).append(str(row.battery))
        for record in details["_daily_records"]:
            day = pd.Timestamp(record["day"])
            batteries = by_day.get(day, [])
            buildings = sorted({str(loc.loc[b, building_column]) for b in batteries})
            flag = " HIT" if record["limit_hit"] else ""
            print(
                f"  day {day.date()} hours {record['hours']:6.2f}"
                f" return {record['return_travel']:5.2f}{flag}  "
                f"{len(batteries)} batt {buildings}"
            )
        for record in details["_weekly_records"]:
            flag = " HIT" if record["limit_hit"] else ""
            print(
                f"  week {pd.Timestamp(record['week_start']).date()}"
                f" hours {record['hours']:6.2f}{flag}"
            )

        # Round-1 delta trace: mirror the post-pass enumeration for the first
        # hit day and print every candidate's operational and timing delta.
        from batteryswap_solution.routing import order_assignments

        assignments: dict[str, pd.Timestamp | None] = {}
        date_index = {date: i for i, date in enumerate(costs.candidate_dates)}
        for row in plan.itertuples(index=False):
            day = pd.Timestamp(row.day).normalize()
            assignments[str(row.battery)] = day if day in date_index else None
        battery_positions = {b: i for i, b in enumerate(costs.battery_ids)}
        priority = {
            b: float(costs.horizon_event_probability[i])
            for i, b in enumerate(costs.battery_ids)
        }
        incumbent_op = float(details["total_cost"])
        by_day2: dict[pd.Timestamp, list[str]] = {}
        for battery_id in sorted(assignments):
            day = assignments[battery_id]
            if day is not None:
                by_day2.setdefault(day, []).append(battery_id)
        worked_days = sorted(by_day2)
        hit_days = [
            pd.Timestamp(r["day"]) for r in details["_daily_records"]
            if r["limit_hit"] and pd.Timestamp(r["day"]) in by_day2
        ]
        day_rows_order: dict[pd.Timestamp, list[str]] = {}
        for row in plan.itertuples(index=False):
            day = pd.Timestamp(row.day).normalize()
            if day in date_index:
                day_rows_order.setdefault(day, []).append(str(row.battery))
        rows_out = []
        for hit in hit_days:
            batteries = day_rows_order[hit]
            groups: dict[str, list[str]] = {}
            sequence: list[str] = []
            for b in batteries:
                building = str(loc.loc[b, building_column])
                if building not in groups:
                    groups[building] = []
                    sequence.append(building)
                groups[building].append(b)
            group_list = []
            for split in range(1, len(sequence)):
                prefix, suffix = [], []
                for building in sequence[:split]:
                    prefix.extend(groups[building])
                for building in sequence[split:]:
                    suffix.extend(groups[building])
                group_list.append(prefix)
                group_list.append(suffix)
            group_list.extend(groups[k] for k in sorted(groups))
            group_list.append(list(batteries))
            group_list.extend([b] for b in batteries)
            group_list = [sorted(g) for g in group_list]
            seen = set()
            for group in group_list:
                positions = [battery_positions[b] for b in group]
                best_day = costs.candidate_dates[
                    int(np.argmin(costs.service_cost[positions].sum(axis=0)))
                ]
                targets = [hit + pd.Timedelta(days=o) for o in (-1, 1, -2, 2, -3, 3, -7, 7, -14, 14)]
                targets.append(best_day)
                targets.extend(sorted((d for d in worked_days if d != hit), key=lambda d: (abs((d - hit).days), d))[:4])
                for target in targets:
                    if target not in date_index:
                        continue
                    if target == costs.candidate_dates[-1] and target.weekday() == 6:
                        continue
                    key = (tuple(group), target)
                    if key in seen:
                        continue
                    seen.add(key)
                    if all(assignments[b] == target for b in group):
                        continue
                    cand = dict(assignments)
                    for b in group:
                        cand[b] = target
                    cand_plan = order_assignments(
                        cand, candidate_locations, travel_costs,
                        str(settings.base_location), defer_day, priority=priority,
                    )
                    cand_op = float(replay_operational_cost(
                        cand_plan, candidate_locations, travel_costs, settings,
                        start, context=context,
                    )["total_cost"])
                    op_delta = cand_op - incumbent_op
                    t_delta = float(sum(
                        costs.service_cost[battery_positions[b], date_index[target]]
                        - costs.service_cost[battery_positions[b], date_index[assignments[b]]]
                        for b in group
                    ))
                    rows_out.append((op_delta + t_delta, op_delta, t_delta, len(group), str(target.date()), group[0]))
        print("== raw hit diagnostics")
        for r in details["_daily_records"]:
            if r["limit_hit"]:
                day_val = pd.Timestamp(r["day"])
                print(
                    f"  hit record day={day_val!r} in_by_day={day_val in by_day2} "
                    f"in_date_index={day_val in date_index}"
                )
        print(f"  by_day keys: {[str(k.date()) for k in worked_days]}")
        print(f"  hit_days list: {[str(h.date()) for h in hit_days]}")
        rows_out.sort(key=lambda r: r[0])
        print(f"== round-1 candidates: {len(rows_out)} evaluated; 25 best by total delta")
        for total, op_d, t_d, size, target, first in rows_out[:25]:
            print(f"  delta {total:9.2f} = op {op_d:9.2f} + timing {t_d:9.2f}  n={size}  -> {target}  [{first}]")

        # Anatomy of the hit day and one split candidate.
        for hit in hit_days:
            print(f"== anatomy of hit day {hit.date()}")
            for b in day_rows_order[hit]:
                building = str(loc.loc[b, building_column])
                out = travel_costs.set_index(["from", "to"])["hours"]
                print(
                    f"  {b} building {building} room {loc.loc[b, 'room' if 'room' in loc else 'room_id']}"
                    f" base->b {float(out.loc[(str(settings.base_location), building)]):5.2f}"
                    f" b->base {float(out.loc[(building, str(settings.base_location))]):5.2f}"
                    f" p {priority[b]:.3f}"
                )
            batteries = day_rows_order[hit]
            groups2: dict[str, list[str]] = {}
            sequence2: list[str] = []
            for b in batteries:
                building = str(loc.loc[b, building_column])
                if building not in groups2:
                    groups2[building] = []
                    sequence2.append(building)
                groups2[building].append(b)
            if len(sequence2) >= 2:
                half = len(sequence2) // 2
                seg = []
                for building in sequence2[half:]:
                    seg.extend(groups2[building])
                target = hit + pd.Timedelta(days=1)
                cand = dict(assignments)
                for b in seg:
                    cand[b] = target
                cand_plan = order_assignments(
                    cand, candidate_locations, travel_costs,
                    str(settings.base_location), defer_day, priority=priority,
                )
                cand_score = replay_operational_cost(
                    cand_plan, candidate_locations, travel_costs, settings,
                    start, include_details=True, context=context,
                )
                print(f"== split suffix {len(seg)} batteries -> {target.date()}: component diff")
                for comp in ("travel", "overtime", "daily_limit", "weekly_limit", "building_change", "room_change"):
                    print(f"  {comp:16s} {float(details[comp]):8.2f} -> {float(cand_score[comp]):8.2f}")
                for record in cand_score["_daily_records"]:
                    flag = " HIT" if record["limit_hit"] else ""
                    print(f"  day {pd.Timestamp(record['day']).date()} hours {record['hours']:6.2f}{flag}")
                for record in cand_score["_weekly_records"]:
                    flag = " HIT" if record["limit_hit"] else ""
                    print(f"  week {pd.Timestamp(record['week_start']).date()} hours {record['hours']:6.2f}{flag}")

        repaired = planner._capacity_repair(
            plan, costs, candidate_locations, travel_costs, settings, start, defer_day
        )
        after = replay_operational_cost(
            repaired, candidate_locations, travel_costs, settings, start,
            include_details=True, context=context,
        )
        print(f"== after capacity pass: operational {after['total_cost']:.2f}")
        for record in after["_daily_records"]:
            flag = " HIT" if record["limit_hit"] else ""
            print(f"  day {pd.Timestamp(record['day']).date()} hours {record['hours']:6.2f}{flag}")
        for record in after["_weekly_records"]:
            flag = " HIT" if record["limit_hit"] else ""
            print(f"  week {pd.Timestamp(record['week_start']).date()} hours {record['hours']:6.2f}{flag}")

        moved = 0
        before_days = plan.set_index("battery")["day"]
        after_days = repaired.set_index("battery")["day"]
        for battery in before_days.index:
            if before_days.loc[battery] != after_days.loc[battery]:
                moved += 1
                print(
                    f"  moved {battery}: {pd.Timestamp(before_days.loc[battery]).date()}"
                    f" -> {pd.Timestamp(after_days.loc[battery]).date()}"
                )
        print(f"== moved {moved} batteries")
        return
    raise SystemExit(f"scenario {args.scenario} not found")


if __name__ == "__main__":
    main()
