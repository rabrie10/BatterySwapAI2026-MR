"""Day assignment optimizer with deterministic CP-SAT and greedy fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostTables, isolated_emergency_costs

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - exercised only in reduced local envs
    cp_model = None


@dataclass(frozen=True)
class OptimizationConfig:
    solver_seconds: float = 2.0
    random_seed: int = 20260818
    cost_scale: int = 1000
    time_scale: int = 1000
    capacity_margin_hours: float = 0.05
    objective_roundtrip_fraction: float = 0.55
    # A day's route is one tour from base and back, not one round trip per
    # building, so charging every building a full round trip overstates how full
    # a day is and pushes the model towards one building per day -- which costs
    # more active days, more returns to base, and more of the evaluator's
    # double-counted return travel.
    capacity_roundtrip_fraction: float = 1.0
    use_cp_sat: bool = True
    max_planned_rate: float | None = None


def planned_swap_limit(battery_count: int, rate: float | None) -> int | None:
    if rate is None:
        return None
    if not 0.0 < float(rate) <= 1.0:
        raise ValueError("max_planned_rate must be in (0, 1]")
    return max(1, int(np.ceil(float(rate) * int(battery_count))))


def _columns(locations: pd.DataFrame) -> tuple[str, str, str]:
    id_column = "battery_id" if "battery_id" in locations else "battery"
    building_column = "building_id" if "building_id" in locations else "building"
    room_column = "room_id" if "room_id" in locations else "room"
    return id_column, building_column, room_column


def _greedy_assign(
    costs: CostTables,
    defer_cost: np.ndarray,
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    settings,
    config: OptimizationConfig,
) -> dict[str, pd.Timestamp | None]:
    """Known-valid deterministic fallback when CP-SAT is unavailable."""

    id_column, building_column, room_column = _columns(locations)
    loc = locations.copy()
    loc[id_column] = loc[id_column].astype(str)
    loc = loc.set_index(id_column)
    travel = travel_costs.set_index(["from", "to"])["hours"]
    base = str(settings.base_location)
    assignments: dict[str, pd.Timestamp | None] = {
        battery_id: None for battery_id in costs.battery_ids
    }
    beneficial: list[tuple[float, str, pd.Timestamp]] = []

    for index, battery_id in enumerate(costs.battery_ids):
        building = str(loc.loc[battery_id, building_column])
        room = str(loc.loc[battery_id, room_column])
        standalone = float(settings.time_per_battery_hours)
        if building != base:
            standalone += float(settings.time_per_building_change_hours)
            standalone += float(travel.loc[(base, building)]) + float(travel.loc[(building, base)])
        if room != str(settings.base_room):
            standalone += float(settings.time_per_room_change_hours)
        service_scores = costs.service_cost[index] + standalone
        best_day = int(np.argmin(service_scores))
        if service_scores[best_day] + 1e-9 < defer_cost[index]:
            beneficial.append(
                (
                    float(defer_cost[index] - service_scores[best_day]),
                    battery_id,
                    costs.candidate_dates[best_day],
                )
            )
    limit = planned_swap_limit(len(costs.battery_ids), config.max_planned_rate)
    selected = sorted(beneficial, key=lambda item: (-item[0], item[1]))
    if limit is not None:
        selected = selected[:limit]
    for _, battery_id, best_day in selected:
        assignments[battery_id] = best_day
    return assignments


def optimize_assignments(
    costs: CostTables,
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    settings,
    *,
    config: OptimizationConfig,
) -> dict[str, pd.Timestamp | None]:
    """Jointly choose service/defer and service day for every battery."""

    emergency_operations = isolated_emergency_costs(
        locations, travel_costs, settings, costs.battery_ids
    )
    defer_cost = costs.defer_cost + costs.horizon_event_probability * emergency_operations
    if cp_model is None or not config.use_cp_sat:
        return _greedy_assign(
            costs, defer_cost, locations, travel_costs, settings, config
        )

    id_column, building_column, room_column = _columns(locations)
    loc = locations.copy()
    loc[id_column] = loc[id_column].astype(str)
    loc = loc.set_index(id_column)
    dates = costs.candidate_dates
    battery_count = len(costs.battery_ids)
    day_count = len(dates)
    base = str(settings.base_location)
    travel = travel_costs.set_index(["from", "to"])["hours"]

    model = cp_model.CpModel()
    service = [
        [model.new_bool_var(f"x_{battery}_{day}") for day in range(day_count)]
        for battery in range(battery_count)
    ]
    deferred = [model.new_bool_var(f"z_{battery}") for battery in range(battery_count)]
    for battery in range(battery_count):
        model.add(sum(service[battery]) + deferred[battery] == 1)
        model.add_hint(deferred[battery], 1)
    limit = planned_swap_limit(battery_count, config.max_planned_rate)
    if limit is not None:
        model.add(
            sum(
                service[battery][day]
                for battery in range(battery_count)
                for day in range(day_count)
            )
            <= limit
        )

    # The evaluator crashes if its horizon end is also Sunday and work is
    # scheduled exactly on that final calendar day. Keep that pathological day
    # available in forecasts but never emit an action there.
    if dates[-1].weekday() == 6:
        for battery in range(battery_count):
            model.add(service[battery][-1] == 0)

    batteries_by_building: dict[str, list[int]] = {}
    batteries_by_room: dict[str, list[int]] = {}
    for battery, battery_id in enumerate(costs.battery_ids):
        building = str(loc.loc[battery_id, building_column])
        room = str(loc.loc[battery_id, room_column])
        batteries_by_building.setdefault(building, []).append(battery)
        batteries_by_room.setdefault(room, []).append(battery)

    building_active = {
        (day, building): model.new_bool_var(f"building_{day}_{building}")
        for day in range(day_count)
        for building in sorted(batteries_by_building)
    }
    room_active = {
        (day, room): model.new_bool_var(f"room_{day}_{room}")
        for day in range(day_count)
        for room in sorted(batteries_by_room)
    }
    day_active = [model.new_bool_var(f"day_{day}") for day in range(day_count)]

    for day in range(day_count):
        for building, batteries in batteries_by_building.items():
            active = building_active[(day, building)]
            expressions = [service[battery][day] for battery in batteries]
            for expression in expressions:
                model.add(expression <= active)
            model.add(active <= sum(expressions))
            model.add(active <= day_active[day])
        for room, batteries in batteries_by_room.items():
            active = room_active[(day, room)]
            expressions = [service[battery][day] for battery in batteries]
            for expression in expressions:
                model.add(expression <= active)
            model.add(active <= sum(expressions))
        model.add(day_active[day] <= sum(building_active[(day, b)] for b in batteries_by_building))

    time_scale = int(config.time_scale)
    battery_units = round(float(settings.time_per_battery_hours) * time_scale)
    room_units = round(float(settings.time_per_room_change_hours) * time_scale)
    building_units = round(float(settings.time_per_building_change_hours) * time_scale)
    diagonal_units = round(float(travel.loc[(base, base)]) * time_scale)
    daily_work = []
    overtime = []
    daily_breach: list = []
    weekly_breach: list = []
    for day in range(day_count):
        work_terms = [
            battery_units * service[battery][day]
            for battery in range(battery_count)
        ]
        work_terms.extend(
            room_units * room_active[(day, room)] for room in batteries_by_room
        )
        for building in batteries_by_building:
            if building == base:
                continue
            roundtrip = float(travel.loc[(base, building)]) + float(travel.loc[(building, base)])
            capacity_units = building_units + round(
                config.capacity_roundtrip_fraction * roundtrip * time_scale
            )
            work_terms.append(capacity_units * building_active[(day, building)])
        work_terms.append(diagonal_units * day_active[day])
        work = model.new_int_var(0, 10**8, f"work_{day}")
        model.add(work == sum(work_terms))
        daily_work.append(work)

        # The evaluator does not forbid a long day or a long week -- it charges a
        # flat penalty for one. Modelling either as a hard constraint makes the
        # optimizer defer a due battery, and a deferred due battery costs
        # 10 per day of lateness from day 48 onwards plus a dedicated emergency
        # trip: 200 to 400, against a penalty of 100. Both limits are therefore
        # priced in the objective, with a loose hard bound kept only to stop the
        # solver exploring physically absurd days.
        daily_limit = round(float(settings.worker_limit_daily_hours) * time_scale)
        model.add(work <= 2 * daily_limit)
        over_daily = model.new_bool_var(f"over_daily_{day}")
        model.add(work >= daily_limit + 1).only_enforce_if(over_daily)
        model.add(work <= daily_limit).only_enforce_if(over_daily.negated())
        daily_breach.append(over_daily)
        overtime_var = model.new_int_var(0, 2 * daily_limit, f"overtime_{day}")
        model.add(overtime_var >= work - round(float(settings.overtime_start) * time_scale))
        overtime.append(overtime_var)

    # The weekly penalty fires at >= the limit, not above it.
    weekly_limit = round(float(settings.worker_limit_weekly_hours) * time_scale)
    for first_day in range(0, day_count, 7):
        week_total = sum(daily_work[first_day : first_day + 7])
        over_weekly = model.new_bool_var(f"over_weekly_{first_day}")
        model.add(week_total >= weekly_limit).only_enforce_if(over_weekly)
        model.add(week_total <= weekly_limit - 1).only_enforce_if(over_weekly.negated())
        weekly_breach.append(over_weekly)

    objective: list = []
    cost_scale = int(config.cost_scale)
    for battery in range(battery_count):
        for day in range(day_count):
            coefficient = round(float(costs.service_cost[battery, day]) * cost_scale)
            coefficient += round(float(settings.time_per_battery_hours) * cost_scale)
            objective.append(coefficient * service[battery][day])
        objective.append(round(float(defer_cost[battery]) * cost_scale) * deferred[battery])

    for day in range(day_count):
        for room in batteries_by_room:
            objective.append(
                round(float(settings.time_per_room_change_hours) * cost_scale)
                * room_active[(day, room)]
            )
        for building in batteries_by_building:
            if building == base:
                continue
            roundtrip = float(travel.loc[(base, building)]) + float(travel.loc[(building, base)])
            proxy = float(settings.time_per_building_change_hours)
            proxy += config.objective_roundtrip_fraction * roundtrip
            objective.append(round(proxy * cost_scale) * building_active[(day, building)])
        objective.append(round(float(travel.loc[(base, base)]) * cost_scale) * day_active[day])
        objective.append(
            round(float(settings.overtime_penalty_factor) * cost_scale / time_scale)
            * overtime[day]
        )

    daily_penalty = round(float(settings.worker_limit_daily_penalty) * cost_scale)
    weekly_penalty = round(float(settings.worker_limit_weekly_penalty) * cost_scale)
    objective.extend(daily_penalty * breach for breach in daily_breach)
    objective.extend(weekly_penalty * breach for breach in weekly_breach)
    model.minimize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.solver_seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(config.random_seed)
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _greedy_assign(
            costs, defer_cost, locations, travel_costs, settings, config
        )

    assignments: dict[str, pd.Timestamp | None] = {}
    for battery, battery_id in enumerate(costs.battery_ids):
        chosen = None
        for day in range(day_count):
            if solver.value(service[battery][day]):
                chosen = dates[day]
                break
        assignments[battery_id] = chosen
    return assignments
