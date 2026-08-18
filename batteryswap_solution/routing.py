"""Deterministic daily route optimization for the evaluator's row-order API."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd


def travel_lookup(travel_costs: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row["from"]), str(row["to"])): float(row["hours"])
        for _, row in travel_costs.iterrows()
    }


def _route_cost(route: list[str], base: str, travel: dict[tuple[str, str], float]) -> float:
    if not route:
        return float(travel[(base, base)])
    total = float(travel[(base, route[0])])
    total += sum(float(travel[(left, right)]) for left, right in zip(route, route[1:]))
    total += float(travel[(route[-1], base)])
    return total


def _held_karp(buildings: tuple[str, ...], base: str, travel: dict[tuple[str, str], float]) -> list[str]:
    count = len(buildings)

    @lru_cache(maxsize=None)
    def solve(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
        if mask == (1 << last):
            return float(travel[(base, buildings[last])]), (last,)
        previous_mask = mask ^ (1 << last)
        best: tuple[float, tuple[int, ...]] | None = None
        for previous in range(count):
            if not previous_mask & (1 << previous):
                continue
            previous_cost, previous_path = solve(previous_mask, previous)
            candidate = (
                previous_cost + float(travel[(buildings[previous], buildings[last])]),
                previous_path + (last,),
            )
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best

    full_mask = (1 << count) - 1
    finalists = []
    for last in range(count):
        partial_cost, path = solve(full_mask, last)
        finalists.append((partial_cost + float(travel[(buildings[last], base)]), path))
    _, best_path = min(finalists)
    return [buildings[index] for index in best_path]


def _insertion_two_opt(
    buildings: tuple[str, ...],
    base: str,
    travel: dict[tuple[str, str], float],
) -> list[str]:
    route: list[str] = []
    remaining = set(buildings)
    while remaining:
        best = None
        for building in sorted(remaining):
            for position in range(len(route) + 1):
                candidate = route[:position] + [building] + route[position:]
                choice = (_route_cost(candidate, base, travel), building, position)
                if best is None or choice < best:
                    best = choice
        assert best is not None
        _, building, position = best
        route.insert(position, building)
        remaining.remove(building)

    improved = True
    while improved:
        improved = False
        incumbent_cost = _route_cost(route, base, travel)
        for left in range(len(route) - 1):
            for right in range(left + 2, len(route) + 1):
                candidate = route[:left] + list(reversed(route[left:right])) + route[right:]
                candidate_cost = _route_cost(candidate, base, travel)
                if candidate_cost + 1e-12 < incumbent_cost:
                    route = candidate
                    incumbent_cost = candidate_cost
                    improved = True
    return route


def route_buildings(
    buildings: list[str] | tuple[str, ...],
    base: str,
    travel: dict[tuple[str, str], float],
    *,
    exact_limit: int = 10,
) -> list[str]:
    """Return the shortest deterministic base-to-base building route found."""

    unique = tuple(sorted(set(str(building) for building in buildings if str(building) != base)))
    if not unique:
        return []
    if len(unique) <= exact_limit:
        return _held_karp(unique, base, travel)
    return _insertion_two_opt(unique, base, travel)


def order_assignments(
    assignments: dict[str, pd.Timestamp | None],
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    base: str,
    defer_day: pd.Timestamp,
    *,
    priority: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert battery/date assignments to evaluator execution order."""

    id_column = "battery_id" if "battery_id" in locations else "battery"
    building_column = "building_id" if "building_id" in locations else "building"
    room_column = "room_id" if "room_id" in locations else "room"
    loc = locations.copy()
    loc[id_column] = loc[id_column].astype(str)
    loc = loc.set_index(id_column)
    travel = travel_lookup(travel_costs)
    priority = priority or {}

    by_day: dict[pd.Timestamp, list[str]] = defaultdict(list)
    deferred: list[str] = []
    for battery_id, day in assignments.items():
        if day is None:
            deferred.append(str(battery_id))
        else:
            by_day[pd.Timestamp(day).normalize()].append(str(battery_id))

    rows: list[dict[str, object]] = []
    for day in sorted(by_day):
        batteries = by_day[day]
        by_building: dict[str, list[str]] = defaultdict(list)
        for battery_id in batteries:
            by_building[str(loc.loc[battery_id, building_column])].append(battery_id)

        building_order = []
        if base in by_building:
            building_order.append(base)
        building_order.extend(route_buildings(list(by_building), base, travel))

        for building in building_order:
            by_room: dict[str, list[str]] = defaultdict(list)
            for battery_id in by_building[building]:
                by_room[str(loc.loc[battery_id, room_column])].append(battery_id)
            room_order = sorted(
                by_room,
                key=lambda room: (
                    -max(priority.get(battery, 0.0) for battery in by_room[room]),
                    room,
                ),
            )
            for room in room_order:
                room_batteries = sorted(
                    by_room[room],
                    key=lambda battery: (-priority.get(battery, 0.0), battery),
                )
                rows.extend({"day": day, "battery": battery} for battery in room_batteries)

    rows.extend(
        {"day": pd.Timestamp(defer_day).normalize(), "battery": battery}
        for battery in sorted(deferred)
    )
    return pd.DataFrame(rows, columns=["day", "battery"]).reset_index(drop=True)
