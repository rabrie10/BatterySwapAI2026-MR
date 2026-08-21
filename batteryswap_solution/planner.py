"""Official BatterySwapAI Planner adapter and score-aligned Task 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from batteryswap_public.evaluate import check_plan_valid, evaluate_plan
from batteryswap_public.interfaces import Planner

from .costs import (
    CostTables,
    build_expected_cost_tables,
    isolated_emergency_costs,
    select_candidates,
)
from .forecast import RiskForecaster, VoltageTrendForecaster, validate_forecast
from .optimizer import OptimizationConfig, optimize_assignments, planned_swap_limit
from .replay import ReplayContext, build_replay_context, replay_operational_cost
from .routing import order_assignments


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerConfig:
    late_risk_multiplier: float = 1.0
    minimum_expected_improvement: float = 0.0
    local_search_evaluations: int = 160
    uncertain_local_search_evaluations: int = 70
    robust_emergency_samples: int = 4
    random_seed: int = 20260818
    emergency_rank_scale: float = 1.0
    candidate_margin_hours: float = 24.0
    max_candidates: int = 150
    optimizer: OptimizationConfig = field(default_factory=OptimizationConfig)


def infer_scenario_start(battery_data: pd.DataFrame) -> pd.Timestamp:
    if isinstance(battery_data.index, pd.MultiIndex) and "end_time" in battery_data.index.names:
        value = battery_data.index.get_level_values("end_time").max()
    elif "end_time" in battery_data:
        value = pd.to_datetime(battery_data["end_time"]).max()
    else:
        raise ValueError("Cannot infer scenario start from battery_data")
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


class CompetitionPlanner(Planner):
    """Production Task 2 planner with deterministic safety fallbacks."""

    def __init__(
        self,
        forecaster: RiskForecaster | None = None,
        config: PlannerConfig | None = None,
    ) -> None:
        self.forecaster = forecaster or VoltageTrendForecaster()
        self.fallback_forecaster = VoltageTrendForecaster()
        self.config = config or PlannerConfig()
        self.last_expected_improvement = float("nan")

    @staticmethod
    def _planning_clock(
        start: pd.Timestamp, settings
    ) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
        end = (start + pd.Timedelta(days=float(settings.planning_window_days))).normalize()
        dates = pd.date_range(start, end, freq="D", inclusive="both")
        return dates, end + pd.Timedelta(days=1)

    @staticmethod
    def _all_defer(
        locations: pd.DataFrame, start: pd.Timestamp, settings
    ) -> pd.DataFrame:
        id_column = "battery_id" if "battery_id" in locations else "battery"
        defer_day = (
            start + pd.Timedelta(days=float(settings.planning_window_days) + 1.0)
        ).normalize()
        return pd.DataFrame(
            {
                "day": pd.DatetimeIndex([defer_day] * len(locations)),
                "battery": sorted(locations[id_column].astype(str)),
            }
        ).reset_index(drop=True)

    @staticmethod
    def _restore_excluded(
        plan: pd.DataFrame, excluded: list[str], defer_day: pd.Timestamp
    ) -> pd.DataFrame:
        """Put the batteries the search never considered back, all deferred.

        A plan must name every battery exactly once, and anything outside the
        candidate set was excluded precisely because deferring dominates.
        """
        if not excluded:
            return plan
        tail = pd.DataFrame(
            {
                "day": pd.DatetimeIndex([pd.Timestamp(defer_day).normalize()] * len(excluded)),
                "battery": sorted(str(value) for value in excluded),
            }
        )
        combined = pd.concat([plan, tail], ignore_index=True)
        return combined.sort_values("day", kind="stable").reset_index(drop=True)

    @staticmethod
    def _operational_score(
        plan: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
        start: pd.Timestamp,
    ) -> float:
        return replay_operational_cost(
            plan, locations, travel_costs, settings, start
        )["total_cost"]

    def _forecast(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        start: pd.Timestamp,
        dates: pd.DatetimeIndex,
    ):
        observation_end = pd.to_datetime(locations["end_time"]).max()
        observation_end = pd.Timestamp(observation_end)
        if observation_end.tzinfo is not None:
            observation_end = observation_end.tz_localize(None)
        battery_ids = locations["battery"].astype(str).tolist()
        kwargs = dict(
            prediction_origin=start,
            horizon_days=len(dates) - 1,
            evaluation_observation_end=observation_end,
        )
        try:
            forecast = self.forecaster.predict(battery_data, locations, **kwargs)
            return validate_forecast(forecast, battery_ids, dates)
        except Exception:
            LOGGER.exception("Primary forecaster failed; using voltage trend fallback")
            forecast = self.fallback_forecaster.predict(battery_data, locations, **kwargs)
            return validate_forecast(forecast, battery_ids, dates)

    def _expected_score(
        self,
        plan: pd.DataFrame,
        costs: CostTables,
        due_samples: np.ndarray,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
        start: pd.Timestamp,
        replay_context: ReplayContext | None = None,
    ) -> float:
        limit = planned_swap_limit(
            len(costs.battery_ids), self.config.optimizer.max_planned_rate
        )
        if limit is not None:
            planned_count = int(
                pd.to_datetime(plan["day"])
                .dt.normalize()
                .isin(costs.candidate_dates)
                .sum()
            )
            if planned_count > limit:
                return float("inf")
        date_to_index = {date: index for index, date in enumerate(costs.candidate_dates)}
        plan_days = plan.set_index("battery")["day"]
        expected_timing = 0.0
        planned_batteries: set[str] = set()
        for battery_index, battery_id in enumerate(costs.battery_ids):
            day = pd.Timestamp(plan_days.loc[battery_id]).normalize()
            if day in date_to_index:
                expected_timing += float(
                    costs.service_cost[battery_index, date_to_index[day]]
                )
                planned_batteries.add(battery_id)
            else:
                expected_timing += float(costs.defer_cost[battery_index])

        if due_samples.size == 0:
            emergency = isolated_emergency_costs(
                locations, travel_costs, settings, costs.battery_ids
            )
            deferred_operation = sum(
                costs.horizon_event_probability[index] * emergency[index]
                for index, battery_id in enumerate(costs.battery_ids)
                if battery_id not in planned_batteries
            )
            return expected_timing + deferred_operation + self._operational_score(
                plan, locations, travel_costs, settings, start
            )

        operational_scores = []
        for sample in due_samples:
            emergency_batteries = [
                battery_id
                for index, battery_id in enumerate(costs.battery_ids)
                if sample[index] and battery_id not in planned_batteries
            ]
            operational_scores.append(
                replay_operational_cost(
                    plan,
                    locations,
                    travel_costs,
                    settings,
                    start,
                    emergency_batteries=emergency_batteries,
                    context=replay_context,
                )["total_cost"]
            )
        return expected_timing + float(np.mean(operational_scores))

    def _local_search(
        self,
        seeds: list[dict[str, pd.Timestamp | None]],
        costs: CostTables,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
        start: pd.Timestamp,
        defer_day: pd.Timestamp,
    ) -> pd.DataFrame:
        loc = locations.copy().set_index(locations["battery"].astype(str))
        building_column = "building" if "building" in loc else "building_id"
        priority = {
            battery_id: float(costs.horizon_event_probability[index])
            for index, battery_id in enumerate(costs.battery_ids)
        }
        replay_context = build_replay_context(
            locations, travel_costs, settings, start
        )
        sample_count = max(int(self.config.robust_emergency_samples), 0)
        if sample_count:
            rng = np.random.default_rng(self.config.random_seed)
            uniforms = np.empty((sample_count, len(costs.battery_ids)), dtype=float)
            strata = (np.arange(sample_count, dtype=float) + 0.5) / sample_count
            for battery_index in range(len(costs.battery_ids)):
                uniforms[:, battery_index] = rng.permutation(strata)
            due_samples = uniforms < costs.horizon_event_probability[None, :]
            due_samples = np.unique(due_samples, axis=0)
        else:
            due_samples = np.empty((0, len(costs.battery_ids)), dtype=bool)
        search_budget = int(self.config.local_search_evaluations)
        if len(due_samples) > 1:
            search_budget = min(
                search_budget,
                int(self.config.uncertain_local_search_evaluations),
            )
        repair_reserve = min(60, max(20, search_budget // 3))
        general_budget = min(100, max(search_budget - repair_reserve, 0))

        def build(candidate_assignments):
            return order_assignments(
                candidate_assignments,
                locations,
                travel_costs,
                str(settings.base_location),
                defer_day,
                priority=priority,
            )

        # Start from whichever construction scores best. Measured on all 48
        # train scenarios a CP-SAT seed and a per-battery-optimum seed converge
        # to the same plan in every cost component, so this is insurance for an
        # unfamiliar split rather than a source of gain today.
        assignments = None
        incumbent = None
        incumbent_score = float("inf")
        for seed in seeds:
            candidate = build(seed)
            score = self._expected_score(
                candidate,
                costs,
                due_samples,
                locations,
                travel_costs,
                settings,
                start,
                replay_context,
            )
            if score < incumbent_score:
                assignments, incumbent, incumbent_score = dict(seed), candidate, score
        all_defer_assignments = {battery_id: None for battery_id in costs.battery_ids}
        all_defer = build(all_defer_assignments)
        all_defer_score = self._expected_score(
            all_defer,
            costs,
            due_samples,
            locations,
            travel_costs,
            settings,
            start,
            replay_context,
        )
        if all_defer_score + 1e-9 < incumbent_score:
            assignments = all_defer_assignments
            incumbent = all_defer
            incumbent_score = all_defer_score

        groups: dict[tuple[pd.Timestamp, str], list[str]] = {}
        for battery_id, day in assignments.items():
            if day is None:
                continue
            building = str(loc.loc[battery_id, building_column])
            groups.setdefault((pd.Timestamp(day).normalize(), building), []).append(battery_id)

        selected_days = sorted({day for day in assignments.values() if day is not None})
        battery_positions = {
            battery_id: index for index, battery_id in enumerate(costs.battery_ids)
        }
        moves: list[tuple[list[str], pd.Timestamp | None]] = []

        # CP-SAT can defer a very expensive battery when conservative capacity
        # bounds make all dates look infeasible. Exact reinsertion gets first
        # claim on the repair budget because it can also remove an emergency
        # queue and its discontinuous capacity penalties.
        deferred = [
            battery_id for battery_id, day in assignments.items() if day is None
        ]
        deferred.sort(
            key=lambda battery_id: (
                -priority[battery_id],
                battery_id,
            )
        )
        for battery_id in deferred:
            if priority[battery_id] <= 0.0:
                continue
            position = battery_positions[battery_id]
            best_index = int(np.argmin(costs.service_cost[position]))
            best_day = costs.candidate_dates[best_index]
            building = str(loc.loc[battery_id, building_column])
            same_building_days = sorted(
                {
                    pd.Timestamp(day).normalize()
                    for other_id, day in assignments.items()
                    if day is not None
                    and str(loc.loc[other_id, building_column]) == building
                },
                key=lambda day: (abs((day - best_day).days), day),
            )
            for target_day in list(dict.fromkeys([best_day] + same_building_days[:3])):
                moves.append(([battery_id], target_day))

        group_items = sorted(groups.items())
        selected_by_building: dict[str, list[str]] = {}
        for battery_id, day in assignments.items():
            if day is None:
                continue
            building = str(loc.loc[battery_id, building_column])
            selected_by_building.setdefault(building, []).append(battery_id)
        building_bundle_moves = []
        for building, batteries in selected_by_building.items():
            visit_days = sorted(
                {pd.Timestamp(assignments[battery_id]).normalize() for battery_id in batteries}
            )
            if len(visit_days) <= 1:
                continue
            positions = [battery_positions[battery_id] for battery_id in batteries]
            aggregate_timing = costs.service_cost[positions].sum(axis=0)
            best_day = costs.candidate_dates[int(np.argmin(aggregate_timing))]
            roundtrip = (
                float(
                    replay_context.travel.loc[
                        (str(settings.base_location), building)
                    ]
                )
                + float(
                    replay_context.travel.loc[
                        (building, str(settings.base_location))
                    ]
                )
            )
            for target_day in list(dict.fromkeys([best_day] + visit_days)):
                building_bundle_moves.append(
                    (roundtrip, batteries, target_day)
                )
        for _, batteries, target_day in sorted(
            building_bundle_moves,
            key=lambda item: (-item[0], item[2], tuple(sorted(item[1]))),
        ):
            moves.append((batteries, target_day))

        selected_by_day: dict[pd.Timestamp, list[str]] = {}
        for battery_id, day in assignments.items():
            if day is not None:
                selected_by_day.setdefault(
                    pd.Timestamp(day).normalize(), []
                ).append(battery_id)
        worked_days = sorted(selected_by_day)
        for left_day, right_day in zip(worked_days, worked_days[1:]):
            moves.append((selected_by_day[left_day], right_day))
            moves.append((selected_by_day[right_day], left_day))

        individual_best_moves = []
        for battery_id, current_day in assignments.items():
            if current_day is None:
                continue
            position = battery_positions[battery_id]
            best_index = int(np.argmin(costs.service_cost[position]))
            best_day = costs.candidate_dates[best_index]
            current_index = costs.candidate_dates.get_loc(
                pd.Timestamp(current_day).normalize()
            )
            regret = float(
                costs.service_cost[position, current_index]
                - costs.service_cost[position, best_index]
            )
            individual_best_moves.append((regret, battery_id, best_day))
        for _, battery_id, best_day in sorted(
            individual_best_moves,
            key=lambda item: (-item[0], item[1]),
        ):
            moves.append(([battery_id], best_day))

        for (_, _), batteries in group_items:
            positions = [battery_positions[battery_id] for battery_id in batteries]
            aggregate_timing = costs.service_cost[positions].sum(axis=0)
            best_day = costs.candidate_dates[int(np.argmin(aggregate_timing))]
            moves.append((batteries, best_day))
        for offset in (-7, 7, -14, 14, -1, 1, -2, 2):
            for (source_day, _), batteries in group_items:
                moves.append(
                    (batteries, source_day + pd.Timedelta(days=offset))
                )
        for (source_day, _), batteries in group_items:
            existing_days = sorted(
                selected_days,
                key=lambda day: (abs((day - source_day).days), day),
            )
            moves.extend((batteries, target_day) for target_day in existing_days)
            moves.append((batteries, None))

        evaluations = 0
        for batteries, target_day in moves:
            if evaluations >= general_budget:
                break
            if target_day is not None and target_day not in costs.candidate_dates:
                continue
            if (
                target_day is not None
                and target_day == costs.candidate_dates[-1]
                and target_day.weekday() == 6
            ):
                continue
            if all(assignments[battery_id] == target_day for battery_id in batteries):
                continue
            candidate_assignments = assignments.copy()
            for battery_id in batteries:
                candidate_assignments[battery_id] = target_day
            candidate = build(candidate_assignments)
            candidate_score = self._expected_score(
                candidate,
                costs,
                due_samples,
                locations,
                travel_costs,
                settings,
                start,
                replay_context,
            )
            evaluations += 1
            if candidate_score + 1e-9 < incumbent_score:
                assignments = candidate_assignments
                incumbent = candidate
                incumbent_score = candidate_score

        # Accepted merges change which batteries share a day. Rebuild those
        # groups and use exact replay records to repair discontinuous limits.
        while evaluations < search_budget:
            replay = replay_operational_cost(
                incumbent,
                locations,
                travel_costs,
                settings,
                start,
                include_details=True,
                context=replay_context,
            )
            hit_days = [
                record["day"]
                for record in replay["_daily_records"]
                if record["limit_hit"]
            ]
            if not hit_days:
                break

            current_worked_days = sorted(
                {
                    pd.Timestamp(day).normalize()
                    for day in incumbent["day"]
                    if pd.Timestamp(day).normalize() in costs.candidate_dates
                }
            )
            repair_moves: list[tuple[list[str], pd.Timestamp]] = []
            repair_by_hit: dict[
                pd.Timestamp, list[tuple[float, list[str], pd.Timestamp]]
            ] = {}
            for hit_day in hit_days:
                day_batteries = incumbent.loc[
                    incumbent["day"] == hit_day, "battery"
                ].astype(str).tolist()
                by_building: dict[str, list[str]] = {}
                for battery_id in day_batteries:
                    building = str(loc.loc[battery_id, building_column])
                    by_building.setdefault(building, []).append(battery_id)
                repair_groups = [[battery_id] for battery_id in day_batteries]
                repair_groups.extend(by_building.values())
                repair_groups.append(day_batteries)
                for batteries in repair_groups:
                    positions = [battery_positions[battery_id] for battery_id in batteries]
                    aggregate_timing = costs.service_cost[positions].sum(axis=0)
                    best_day = costs.candidate_dates[int(np.argmin(aggregate_timing))]
                    targets = [
                        best_day,
                        hit_day - pd.Timedelta(days=7),
                        hit_day + pd.Timedelta(days=7),
                        hit_day - pd.Timedelta(days=1),
                        hit_day + pd.Timedelta(days=1),
                    ]
                    targets.extend(
                        sorted(
                            current_worked_days,
                            key=lambda day: (abs((day - hit_day).days), day),
                        )
                    )
                    for target_day in dict.fromkeys(targets):
                        if target_day in costs.candidate_dates and target_day != hit_day:
                            repair_moves.append((batteries, target_day))
                            target_index = costs.candidate_dates.get_loc(target_day)
                            current_index = costs.candidate_dates.get_loc(hit_day)
                            timing_delta = float(
                                costs.service_cost[positions, target_index].sum()
                                - costs.service_cost[positions, current_index].sum()
                            )
                            repair_by_hit.setdefault(hit_day, []).append(
                                (timing_delta, batteries, target_day)
                            )

            accepted = False
            compound_moves: list[dict[str, pd.Timestamp]] = []
            if len(repair_by_hit) >= 2:
                ranked_by_hit = {
                    day: sorted(
                        candidates,
                        key=lambda item: (
                            item[0], item[2], tuple(sorted(item[1]))
                        ),
                    )[:5]
                    for day, candidates in repair_by_hit.items()
                }
                ranked_days = sorted(ranked_by_hit)
                for left_day, right_day in zip(ranked_days, ranked_days[1:]):
                    for _, left_batteries, left_target in ranked_by_hit[left_day]:
                        for _, right_batteries, right_target in ranked_by_hit[right_day]:
                            updates = {
                                battery_id: left_target
                                for battery_id in left_batteries
                            }
                            updates.update(
                                {
                                    battery_id: right_target
                                    for battery_id in right_batteries
                                }
                            )
                            compound_moves.append(updates)

            for updates in compound_moves:
                if evaluations >= search_budget:
                    break
                candidate_assignments = assignments.copy()
                candidate_assignments.update(updates)
                candidate = build(candidate_assignments)
                candidate_score = self._expected_score(
                    candidate,
                    costs,
                    due_samples,
                    locations,
                    travel_costs,
                    settings,
                    start,
                    replay_context,
                )
                evaluations += 1
                if candidate_score + 1e-9 < incumbent_score:
                    assignments = candidate_assignments
                    incumbent = candidate
                    incumbent_score = candidate_score
                    accepted = True
                    break

            if accepted:
                continue
            for batteries, target_day in repair_moves:
                if evaluations >= search_budget:
                    break
                if (
                    target_day == costs.candidate_dates[-1]
                    and target_day.weekday() == 6
                ):
                    continue
                candidate_assignments = assignments.copy()
                for battery_id in batteries:
                    candidate_assignments[battery_id] = target_day
                candidate = build(candidate_assignments)
                candidate_score = self._expected_score(
                    candidate,
                    costs,
                    due_samples,
                    locations,
                    travel_costs,
                    settings,
                    start,
                    replay_context,
                )
                evaluations += 1
                if candidate_score + 1e-9 < incumbent_score:
                    assignments = candidate_assignments
                    incumbent = candidate
                    incumbent_score = candidate_score
                    accepted = True
                    break
            if not accepted:
                break
        self.last_expected_improvement = max(all_defer_score - incumbent_score, 0.0)
        if (
            self.last_expected_improvement + 1e-9
            < float(self.config.minimum_expected_improvement)
        ):
            return all_defer
        return incumbent

    def plan(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
    ) -> pd.DataFrame:
        start = infer_scenario_start(battery_data)
        fallback = self._all_defer(locations, start, settings)
        try:
            dates, defer_day = self._planning_clock(start, settings)
            forecast = self._forecast(battery_data, locations, start, dates)
            full_costs = build_expected_cost_tables(
                forecast,
                locations,
                settings,
                dates,
                late_risk_multiplier=self.config.late_risk_multiplier,
                emergency_rank_scale=self.config.emergency_rank_scale,
            )
            # Search only over batteries servicing could plausibly help. Every
            # search evaluation is linear in the battery count, and roughly
            # nine of some four hundred are ever due, so this is the difference
            # between fitting the evaluation budget and not.
            keep = select_candidates(
                full_costs,
                margin_hours=self.config.candidate_margin_hours,
                max_candidates=self.config.max_candidates,
            )
            costs = full_costs.take(keep)
            candidate_ids = set(costs.battery_ids)
            excluded = [
                battery_id
                for battery_id in full_costs.battery_ids
                if battery_id not in candidate_ids
            ]
            id_column = "battery_id" if "battery_id" in locations else "battery"
            candidate_locations = locations[
                locations[id_column].astype(str).isin(candidate_ids)
            ]

            # Measured on all 48 train scenarios, seeding the search from the
            # per-battery optimum instead of the CP-SAT assignment produces an
            # identical plan in every component -- the local search dominates
            # the construction. One seed it is.
            seeds = [
                optimize_assignments(
                    costs,
                    candidate_locations,
                    travel_costs,
                    settings,
                    config=self.config.optimizer,
                )
            ]
            plan = self._local_search(
                seeds,
                costs,
                candidate_locations,
                travel_costs,
                settings,
                start,
                defer_day,
            )
            plan = self._restore_excluded(plan, excluded, defer_day)
            check_plan_valid(plan, locations, start_time=start)
            fast_score = self._operational_score(
                plan, locations, travel_costs, settings, start
            )
            _, _, official = evaluate_plan(
                plan,
                locations,
                travel_costs,
                settings,
                eol_times=None,
                start_time=start,
                verbose=0,
            )
            if not np.isclose(fast_score, float(official["total_cost"]), atol=1e-8):
                raise RuntimeError(
                    f"Operational replay mismatch: fast={fast_score}, official={official['total_cost']}"
                )
            return plan[["day", "battery"]].reset_index(drop=True)
        except Exception:
            LOGGER.exception("Task 2 optimization failed; returning valid all-defer plan")
            check_plan_valid(fallback, locations, start_time=start)
            return fallback
