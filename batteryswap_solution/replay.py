"""Fast evaluator-equivalent replay for operational Task 2 costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReplayContext:
    locations: pd.DataFrame
    travel: pd.Series
    settings: object
    start: pd.Timestamp
    horizon_end: pd.Timestamp
    end_day: pd.Timestamp


def build_replay_context(
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    settings,
    start_time: pd.Timestamp,
) -> ReplayContext:
    start = pd.Timestamp(start_time).normalize()
    horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
    end_day = horizon_end.normalize() + pd.Timedelta(days=(6 - horizon_end.weekday()))
    loc = locations.copy()
    loc["battery"] = loc["battery"].astype(str)
    loc = loc.set_index("battery")
    travel = travel_costs.set_index(["from", "to"])["hours"]
    return ReplayContext(loc, travel, settings, start, horizon_end, end_day)


def replay_operational_cost(
    plan: pd.DataFrame,
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    settings,
    start_time: pd.Timestamp,
    *,
    emergency_batteries: list[str] | tuple[str, ...] = (),
    include_details: bool = False,
    context: ReplayContext | None = None,
) -> dict[str, object]:
    """Replay all non-timing costs using batteryswap_public 0.3.4 semantics."""

    context = context or build_replay_context(
        locations, travel_costs, settings, start_time
    )
    loc = context.locations
    travel = context.travel
    settings = context.settings
    start = context.start
    horizon_end = context.horizon_end
    end_day = context.end_day
    active_plan = plan[plan["day"] <= horizon_end]
    by_day = {
        pd.Timestamp(day).normalize(): group["battery"].astype(str).tolist()
        for day, group in active_plan.groupby("day", sort=False)
    }

    base = str(settings.base_location)

    state_building = base
    state_room = str(settings.base_room)
    state_day = start
    daily_work = 0.0
    weekly_work = 0.0
    last_week_transition = start
    score = {
        "battery_swap": 0.0,
        "building_change": 0.0,
        "room_change": 0.0,
        "travel": 0.0,
        "overtime": 0.0,
        "daily_limit": 0.0,
        "weekly_limit": 0.0,
    }
    daily_records: list[dict[str, object]] = []
    weekly_records: list[dict[str, object]] = []

    def add_work(component: str, amount: float) -> None:
        nonlocal daily_work, weekly_work
        amount = float(amount)
        score[component] += amount
        daily_work += amount
        weekly_work += amount

    def check_week(day: pd.Timestamp, force: bool = False) -> None:
        nonlocal weekly_work, last_week_transition
        if force or day >= last_week_transition + pd.Timedelta(days=7):
            limit_hit = weekly_work >= float(settings.worker_limit_weekly_hours)
            if limit_hit:
                score["weekly_limit"] += float(settings.worker_limit_weekly_penalty)
            weekly_records.append(
                {
                    "week_start": last_week_transition,
                    "closed_on": day,
                    "hours": weekly_work,
                    "limit_hit": limit_hit,
                }
            )
            weekly_work = 0.0
            last_week_transition = day

    def end_previous_day(new_day: pd.Timestamp) -> None:
        nonlocal state_building, state_day, daily_work, weekly_work
        return_travel = float(travel.loc[(state_building, base)])
        total_daily = daily_work + return_travel
        score["travel"] += return_travel
        weekly_work += return_travel
        limit_hit = total_daily > float(settings.worker_limit_daily_hours)
        if limit_hit:
            score["daily_limit"] += float(settings.worker_limit_daily_penalty)
        overtime = max(total_daily - float(settings.overtime_start), 0.0)
        score["overtime"] += overtime * float(settings.overtime_penalty_factor)
        daily_records.append(
            {
                "day": state_day,
                "closed_on": new_day,
                "hours": total_daily,
                "return_travel": return_travel,
                "limit_hit": limit_hit,
            }
        )
        state_building = base
        state_day = new_day
        # evaluate.py changes day (resetting time_of_day) before applying the
        # return-travel transition. The return is therefore also carried into
        # the next workday's overtime calculation.
        daily_work = return_travel

    def visit_battery(battery_id: str) -> None:
        nonlocal state_building, state_room
        building = str(loc.loc[battery_id, "building"])
        room = str(loc.loc[battery_id, "room"])
        if state_building != building:
            add_work("building_change", float(settings.time_per_building_change_hours))
            add_work("travel", float(travel.loc[(state_building, building)]))
            state_building = building
        if state_room != room:
            add_work("room_change", float(settings.time_per_room_change_hours))
            state_room = room
        add_work("battery_swap", float(settings.time_per_battery_hours))

    for calendar_day in pd.date_range(start, end_day, freq="D", inclusive="both"):
        check_week(calendar_day)
        batteries = by_day.get(calendar_day, [])
        if batteries and state_day != calendar_day:
            end_previous_day(calendar_day)
        for battery_id in batteries:
            visit_battery(battery_id)

    if state_day == end_day:
        raise ValueError("Evaluator cannot close a plan with work on its final Sunday")
    end_previous_day(end_day)
    for battery_id in sorted(str(value) for value in emergency_batteries):
        check_week(state_day)
        visit_battery(battery_id)
        end_previous_day(state_day + pd.Timedelta(days=1))
    check_week(state_day, force=True)
    score["total_cost"] = float(sum(score.values()))
    if include_details:
        score["_daily_records"] = daily_records
        score["_weekly_records"] = weekly_records
    return score
