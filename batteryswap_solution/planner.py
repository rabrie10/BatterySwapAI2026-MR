"""Official BatterySwapAI Planner adapter and score-aligned Task 2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from .optimizer import OptimizationConfig, optimize_assignments, scenario_planned_swap_limit
from .replay import ReplayContext, build_replay_context, replay_operational_cost
from .routing import order_assignments, route_buildings, travel_lookup


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
    # Planned-swap ceiling as a function of the scenario's median wasted-swap
    # price X = (observation end + 30 d) - window end, which is known at plan
    # time. The ranking's top-12 realised rate is 0.60 in high-X scenarios and
    # 0.21 in the mid range (and no amount of foresight fixes the mid range:
    # a 21-day peek only reaches 0.15 on the still-alive dues), so volume is
    # spent where the ranking earns it and withheld where it cannot.
    # Bands: (x_upper_bound, cap), evaluated in order; None disables.
    planned_cap_by_median_x: tuple[tuple[float, int], ...] | None = None
    # Act on the forecast's slot_demote fingerprint (see bsai/forecaster.py).
    # Measured 2026-08-22: demotion-only arm cost ~+60-100 (misses 4.25->4.48)
    # -- the fingerprint sweeps real dues; under a binding cap the swap set is
    # substitution-saturated. Off by default; kept for future tighter rules.
    slot_demotion: bool = False
    # The JOINT selection exchange: defer zombie-fingerprint planned batteries
    # AND add dark-channel gate batteries AND refill to the cap, as one pass.
    # Measured on the paired-incumbent harness (exact deltas, official scorer,
    # all 48 scenarios): -79.2/scenario, 33 wins / 13 losses, sign-test
    # p = 0.0045. Either side alone is neutral-to-harmful; the value is the
    # directed exchange (see outputs/paired_selection.md).
    selection_exchange: bool = True
    # Deterministic capacity post-pass that runs after the local search and
    # repairs the daily/weekly limit hits the budgeted search left behind,
    # then merges adjacent light days (CompetitionPlanner._capacity_repair).
    # Kill switch: construct the planner with capacity_repair=False.
    capacity_repair: bool = True
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
        limit = scenario_planned_swap_limit(costs, self.config.optimizer)
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

    # ------------------------------------------------------------------
    # Capacity post-pass. Deterministic; runs after the local search and
    # before _restore_excluded. Enabled by PlannerConfig.capacity_repair.
    # ------------------------------------------------------------------
    def _capacity_repair(
        self,
        plan: pd.DataFrame,
        costs: CostTables,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
        start: pd.Timestamp,
        defer_day: pd.Timestamp,
    ) -> pd.DataFrame:
        """Repair the capacity-limit hits the budgeted local search left behind.

        The evaluator charges a flat 100 per day whose hours exceed the daily
        limit (strict >) and per Monday-anchored week bucket at or above the
        weekly limit, plus 2x(hours - 8) overtime per day. The local search
        repairs these with whatever budget survives its general moves; this
        pass is the dedicated finisher with exact acceptance. Two measured
        caveats bound what finishing can earn (tools/capacity_pass_probe.py on
        s_4/s_21): the historical 25-35h mega-days are far-cluster geometry --
        every building ~8h from base, so any split buys a second ~16h-travel
        day that inherits the first day's return carry (evaluate.py carries a
        day's return travel into the next workday) and trips its own limit --
        and a large share of the validation's daily/weekly component is
        emergency-day cost from missed forecasts, which no plan repair can
        touch. On such days this pass proves irreducibility by finding no
        improving move and leaves them alone. What it does fix:

        * for every day over the daily limit and every worked day inside a hit
          week bucket, try moving each building group, each contiguous
          route segment (the split that actually fixes a chained multi-building
          day: single-building removals leave it over the limit), the whole
          day, and each single battery to nearby days (+-1..3, +-7, +-14), the
          group's timing-optimal day, and the nearest existing workdays --
          week fixes only consider targets in a different week bucket;
        * once nothing is over a limit, try merging adjacent light days when
          the merge saves operational cost outright (a return trip against the
          overtime it buys) without degrading the combined objective.

        A move is accepted iff exact-replay operational delta plus the timing
        delta from ``costs.service_cost`` is strictly negative (merges must
        also save operational cost on their own: a merge funded purely by
        expected timing is the local search's job, not capacity repair). The
        planned set never changes, so the deferred-emergency term of the
        search objective cancels and acceptance is exactly the deterministic
        objective delta. Robust emergency sampling
        (``robust_emergency_samples > 0``) averages emergency replays into the
        search objective; this pass deliberately ignores that coupling -- the
        shipped configuration runs with sampling off.

        Determinism: hit days and weeks are visited in sorted order, groups
        and targets are enumerated in sorted order, budgets truncate that
        deterministic enumeration, and each round accepts the strict minimum
        of (delta, target day, sorted batteries). Every accepted move lowers
        the objective by more than 1e-9, so the loop cannot cycle; the caps
        below only bound the worst case. Candidate plans re-route only the
        touched days (day routes are independent in the evaluator), which
        keeps a candidate at one small routing call plus one ~ms replay.
        """

        max_rounds = 40         # accepted moves; each costs one detailed replay
        round_replays = 120     # candidate replays per round
        total_replays = 600     # candidate replays for the whole pass (~ms each)
        merge_headroom = 6.0    # pre-filter slack over the daily limit for merges

        loc = locations.copy().set_index(locations["battery"].astype(str))
        building_column = "building" if "building" in loc else "building_id"
        room_column = "room" if "room" in loc else "room_id"
        base = str(settings.base_location)
        building_of = {
            battery_id: str(loc.loc[battery_id, building_column])
            for battery_id in loc.index
        }
        room_of = {
            battery_id: str(loc.loc[battery_id, room_column])
            for battery_id in loc.index
        }
        battery_positions = {
            battery_id: index for index, battery_id in enumerate(costs.battery_ids)
        }
        priority = {
            battery_id: float(costs.horizon_event_probability[index])
            for index, battery_id in enumerate(costs.battery_ids)
        }
        travel = travel_lookup(travel_costs)
        replay_context = build_replay_context(locations, travel_costs, settings, start)
        date_index = {date: index for index, date in enumerate(costs.candidate_dates)}
        last_date = costs.candidate_dates[-1]
        defer_norm = pd.Timestamp(defer_day).normalize()

        def route_day(batteries: list[str]) -> list[str]:
            """One day's evaluator row order, exactly as order_assignments emits it."""
            by_building: dict[str, list[str]] = {}
            for battery_id in sorted(batteries):
                by_building.setdefault(building_of[battery_id], []).append(battery_id)
            building_order = [base] if base in by_building else []
            building_order.extend(route_buildings(list(by_building), base, travel))
            rows: list[str] = []
            for building in building_order:
                by_room: dict[str, list[str]] = {}
                for battery_id in by_building[building]:
                    by_room.setdefault(room_of[battery_id], []).append(battery_id)
                for room in sorted(
                    by_room,
                    key=lambda room: (
                        -max(priority.get(battery, 0.0) for battery in by_room[room]),
                        room,
                    ),
                ):
                    rows.extend(
                        sorted(
                            by_room[room],
                            key=lambda battery: (-priority.get(battery, 0.0), battery),
                        )
                    )
            return rows

        # The plan at this point names exactly the candidate batteries; days
        # outside the cost tables' candidate dates are the deferred tail. The
        # incoming per-day row order is kept verbatim (it is the local
        # search's routed order); only days a move touches are re-routed.
        day_rows: dict[pd.Timestamp, list[str]] = {}
        deferred: list[str] = []
        assignments: dict[str, pd.Timestamp | None] = {}
        for row in plan.itertuples(index=False):
            day = pd.Timestamp(row.day).normalize()
            battery_id = str(row.battery)
            if day in date_index:
                assignments[battery_id] = day
                day_rows.setdefault(day, []).append(battery_id)
            else:
                assignments[battery_id] = None
                deferred.append(battery_id)
        deferred = sorted(deferred)

        def frame_of(rows_by_day: dict[pd.Timestamp, list[str]]) -> pd.DataFrame:
            days: list[pd.Timestamp] = []
            batteries: list[str] = []
            for day in sorted(rows_by_day):
                for battery_id in rows_by_day[day]:
                    days.append(day)
                    batteries.append(battery_id)
            days.extend([defer_norm] * len(deferred))
            batteries.extend(deferred)
            return pd.DataFrame(
                {"day": pd.DatetimeIndex(days), "battery": batteries}
            )

        def moved_rows(
            group: list[str], target: pd.Timestamp
        ) -> dict[pd.Timestamp, list[str]]:
            group_set = set(group)
            rows = dict(day_rows)
            for source in sorted({assignments[battery_id] for battery_id in group}):
                remaining = [b for b in rows[source] if b not in group_set]
                if remaining:
                    rows[source] = route_day(remaining)
                else:
                    del rows[source]
            rows[target] = route_day(rows.get(target, []) + group)
            return rows

        def operational(current: pd.DataFrame, with_details: bool = False):
            return replay_operational_cost(
                current,
                locations,
                travel_costs,
                settings,
                start,
                include_details=with_details,
                context=replay_context,
            )

        def timing_delta(batteries: list[str], target: pd.Timestamp) -> float:
            target_index = date_index[target]
            return float(
                sum(
                    costs.service_cost[battery_positions[battery_id], target_index]
                    - costs.service_cost[
                        battery_positions[battery_id],
                        date_index[assignments[battery_id]],
                    ]
                    for battery_id in batteries
                )
            )

        def week_bucket(day: pd.Timestamp) -> int:
            return int((day - start).days) // 7

        incumbent = frame_of(day_rows)
        details = operational(incumbent, with_details=True)
        incumbent_operational = float(details["total_cost"])
        spent = 0

        for _ in range(max_rounds):
            if not day_rows:
                break
            worked_days = sorted(day_rows)
            day_hours = {
                pd.Timestamp(record["day"]): float(record["hours"])
                for record in details["_daily_records"]
            }
            hit_days = sorted(
                pd.Timestamp(record["day"])
                for record in details["_daily_records"]
                if record["limit_hit"] and pd.Timestamp(record["day"]) in day_rows
            )
            hit_buckets = sorted(
                {
                    week_bucket(pd.Timestamp(record["week_start"]))
                    for record in details["_weekly_records"]
                    if record["limit_hit"]
                }
            )

            def day_groups(day: pd.Timestamp) -> tuple[list[list[str]], list[list[str]]]:
                """(structural groups, single batteries) for one worked day.

                Structural groups -- contiguous route segments, per-building
                groups, the whole day -- come first in the enumeration:
                chained multi-building days are only fixable by shedding
                several buildings at once, and the cheap subsets to shed are
                prefixes/suffixes of the route. Singles are enumerated last so
                a tight budget is spent on moves that can actually clear a hit.
                """
                batteries = day_rows[day]
                by_building: dict[str, list[str]] = {}
                building_sequence: list[str] = []
                for battery_id in batteries:  # row order == route order
                    building = building_of[battery_id]
                    if building not in by_building:
                        by_building[building] = []
                        building_sequence.append(building)
                    by_building[building].append(battery_id)
                groups: list[list[str]] = []
                for split in range(1, len(building_sequence)):
                    prefix: list[str] = []
                    for building in building_sequence[:split]:
                        prefix.extend(by_building[building])
                    suffix: list[str] = []
                    for building in building_sequence[split:]:
                        suffix.extend(by_building[building])
                    groups.append(prefix)
                    groups.append(suffix)
                groups.extend(
                    by_building[building] for building in sorted(by_building)
                )
                if len(by_building) > 1:
                    groups.append(list(batteries))
                singles = [[battery_id] for battery_id in sorted(batteries)]
                return [sorted(group) for group in groups], singles

            def group_targets(day: pd.Timestamp, group: list[str]) -> list[pd.Timestamp]:
                positions = [battery_positions[battery_id] for battery_id in group]
                best_day = costs.candidate_dates[
                    int(np.argmin(costs.service_cost[positions].sum(axis=0)))
                ]
                targets = [
                    day + pd.Timedelta(days=offset)
                    for offset in (1, -1, 2, -2, 3, -3, 7, -7, 14, -14)
                ]
                targets.append(best_day)
                targets.extend(
                    sorted(
                        (other for other in worked_days if other != day),
                        key=lambda other: (abs((other - day).days), other),
                    )[:4]
                )
                return targets

            # Targets iterate on the outside so every structural group gets
            # its nearby-day shot before the budget can run out.
            candidates: list[tuple[list[str], pd.Timestamp]] = []
            fix_days: list[tuple[pd.Timestamp, int | None]] = [
                (day, None) for day in hit_days
            ]
            for bucket in hit_buckets:
                fix_days.extend(
                    (day, bucket)
                    for day in worked_days
                    if week_bucket(day) == bucket
                )
            deferred_singles: list[tuple[list[str], pd.Timestamp]] = []
            for day, bucket in fix_days:
                structural, singles = day_groups(day)
                target_lists = {
                    tuple(group): group_targets(day, group)
                    for group in structural + singles
                }
                longest = max((len(t) for t in target_lists.values()), default=0)
                for rank in range(longest):
                    for group in structural:
                        targets = target_lists[tuple(group)]
                        if rank < len(targets):
                            target = targets[rank]
                            if bucket is None or week_bucket(target) != bucket:
                                candidates.append((group, target))
                    for group in singles:
                        targets = target_lists[tuple(group)]
                        if rank < len(targets):
                            target = targets[rank]
                            if bucket is None or week_bucket(target) != bucket:
                                deferred_singles.append((group, target))
            candidates.extend(deferred_singles)
            if not candidates:
                # Nothing over a limit: try merging adjacent light days. The
                # replay delta prices the saved return trip against overtime
                # and any freshly tripped limit exactly.
                daily_limit_hours = float(settings.worker_limit_daily_hours)
                for left, right in zip(worked_days, worked_days[1:]):
                    combined = day_hours.get(left, 0.0) + day_hours.get(right, 0.0)
                    if combined > daily_limit_hours + merge_headroom:
                        continue
                    candidates.append((sorted(day_rows[left]), right))
                    candidates.append((sorted(day_rows[right]), left))

            best_key: tuple[float, pd.Timestamp, tuple[str, ...]] | None = None
            best_rows: dict[pd.Timestamp, list[str]] | None = None
            best_group: list[str] | None = None
            best_target: pd.Timestamp | None = None
            seen: set[tuple[tuple[str, ...], pd.Timestamp]] = set()
            round_spent = 0
            for group, target in candidates:
                if round_spent >= round_replays or spent >= total_replays:
                    break
                if target not in date_index:
                    continue
                if target == last_date and target.weekday() == 6:
                    # evaluate.py cannot close a plan with work on its final Sunday.
                    continue
                key = (tuple(group), target)
                if key in seen:
                    continue
                seen.add(key)
                if all(assignments[battery_id] == target for battery_id in group):
                    continue
                candidate_rows = moved_rows(group, target)
                candidate_operational = float(
                    operational(frame_of(candidate_rows))["total_cost"]
                )
                round_spent += 1
                spent += 1
                operational_delta = candidate_operational - incumbent_operational
                # Every accepted move must save operational cost outright.
                # Timing only guards (the combined delta below): a move funded
                # by expected timing is the local search's jurisdiction, and
                # measured on the 48-scenario A/B such accepts realize badly
                # (s_33 +225, s_37 +171 from two timing-funded "fixes" that
                # bought fresh limit hits against an expected-timing credit).
                if operational_delta >= -1e-9:
                    continue
                delta = operational_delta + timing_delta(group, target)
                candidate_key = (delta, target, tuple(group))
                if delta < -1e-9 and (best_key is None or candidate_key < best_key):
                    best_key = candidate_key
                    best_rows = candidate_rows
                    best_group = group
                    best_target = target
            if best_key is None:
                break
            day_rows = best_rows
            for battery_id in best_group:
                assignments[battery_id] = best_target
            incumbent = frame_of(day_rows)
            details = operational(incumbent, with_details=True)
            incumbent_operational = float(details["total_cost"])
        return incumbent

    def _selection_exchange(
        self,
        plan: pd.DataFrame,
        full_costs: CostTables,
        forecast,
        dates: pd.DatetimeIndex,
        defer_day: pd.Timestamp,
        locations: pd.DataFrame,
        candidate_ids: set[str],
        limit: int | None,
    ) -> pd.DataFrame:
        """The measured joint exchange: zombies out, gate batteries in, refill.

        Implements the paired-harness arm 'A+B refilled to limit' with the
        fidelity corrections from outputs/paired_selection.md: row order is
        preserved (row order within a day IS the route -- re-sorting cost
        +77.8/scenario on its own), adds displace the lowest-probability
        planned battery when at the cap, placement is nearest-planned-visit
        first for adds and refills alike, the refill pool is the candidate
        set, and the regime gate skips only the closing scenarios (median
        X < 50) where the arms measured the exchange harmful.
        """
        summaries = getattr(forecast, "summaries", None)
        if summaries is None or summaries.empty or "gate_include" not in summaries:
            return plan
        end_times = pd.to_datetime(locations["end_time"])
        if getattr(end_times.dt, "tz", None) is not None:
            end_times = end_times.dt.tz_localize(None)
        median_x = float(
            (end_times.dt.normalize() + pd.Timedelta(days=30.0) - dates[-1]).dt.days.median()
        )
        if median_x < 50.0:
            return plan
        flags = summaries.set_index(summaries["battery_id"].astype(str))
        zombies = set(flags.index[flags["slot_demote"].astype(bool)])
        gates = set(flags.index[flags["gate_include"].astype(bool)])
        if not zombies and not gates:
            return plan

        horizon_end = dates[-1]
        rows: list[tuple[pd.Timestamp, str]] = [
            (pd.Timestamp(day).normalize(), str(battery))
            for day, battery in zip(plan["day"], plan["battery"])
        ]
        position_of = {b: i for i, b in enumerate(full_costs.battery_ids)}
        probability = full_costs.horizon_event_probability
        id_column = "battery_id" if "battery_id" in locations else "battery"
        building_column = "building_id" if "building_id" in locations else "building"
        loc = locations.copy()
        loc[id_column] = loc[id_column].astype(str)
        building_of = dict(zip(loc[id_column], loc[building_column].astype(str)))

        def planned_items() -> list[tuple[int, pd.Timestamp, str]]:
            return [
                (i, day, battery)
                for i, (day, battery) in enumerate(rows)
                if day <= horizon_end
            ]

        def best_day(battery: str) -> pd.Timestamp | None:
            index = position_of.get(battery)
            if index is None:
                return None
            order = np.argsort(full_costs.service_cost[index])
            for j in order:
                day = full_costs.candidate_dates[j]
                if day == full_costs.candidate_dates[-1] and day.weekday() == 6:
                    continue
                return day
            return None

        def placement_day(battery: str) -> pd.Timestamp | None:
            anchor = best_day(battery)
            building = building_of.get(battery)
            visits = [
                day
                for _, day, other in planned_items()
                if other != battery and building_of.get(other) == building
            ]
            if not visits or anchor is None:
                return anchor
            return min(visits, key=lambda day: (abs((day - anchor).days), day))

        def move_to_defer(battery: str) -> None:
            for i, (day, other) in enumerate(rows):
                if other == battery:
                    rows.pop(i)
                    rows.append((pd.Timestamp(defer_day).normalize(), battery))
                    return

        def insert_planned(battery: str, day: pd.Timestamp) -> None:
            # Remove the battery's existing (deferred) row, then insert at the
            # END of the target day's group so existing routes are untouched.
            for i, (_, other) in enumerate(rows):
                if other == battery:
                    rows.pop(i)
                    break
            insert_at = len(rows)
            seen_day = False
            for i, (row_day, _) in enumerate(rows):
                if row_day == day:
                    seen_day = True
                    insert_at = i + 1
                elif seen_day and row_day != day:
                    insert_at = i
                    break
                elif not seen_day and row_day > day:
                    insert_at = i
                    break
            rows.insert(insert_at, (day, battery))

        planned = {b for _, _, b in planned_items()}
        cap = int(limit) if limit is not None else len(planned) + len(gates)

        # 1. Zombies out.
        for battery in sorted(zombies & planned):
            move_to_defer(battery)
            planned.discard(battery)

        def displace_weakest() -> bool:
            victims = [
                (float(probability[position_of[b]]), b)
                for _, _, b in planned_items()
                if b not in gates and position_of.get(b) is not None
            ]
            if not victims:
                return False
            _, victim = min(victims)
            move_to_defer(victim)
            planned.discard(victim)
            return True

        # 2. Gate batteries in; displace the weakest planned battery at the cap.
        for battery in sorted(gates - planned):
            day = placement_day(battery)
            if day is None:
                continue
            if len(planned) >= cap and not displace_weakest():
                break
            insert_planned(battery, day)
            planned.add(battery)

        # 3. Refill headroom from the candidate pool by probability.
        if len(planned) < cap:
            pool = sorted(
                (
                    (float(-probability[position_of[b]]), b)
                    for b in candidate_ids
                    if b not in planned and b not in zombies and b not in gates
                    and position_of.get(b) is not None
                ),
            )
            for negative_p, battery in pool:
                if len(planned) >= cap:
                    break
                if -negative_p <= 0.05:
                    break
                day = placement_day(battery)
                if day is None:
                    continue
                insert_planned(battery, day)
                planned.add(battery)

        return pd.DataFrame(
            {
                "day": pd.DatetimeIndex([day for day, _ in rows]),
                "battery": [battery for _, battery in rows],
            }
        ).reset_index(drop=True)

    def plan(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
    ) -> pd.DataFrame:
        start = infer_scenario_start(battery_data)
        fallback = self._all_defer(locations, start, settings)
        # Any config mutation below (X-banded cap, full-fleet budget) is
        # scenario-local: a reused planner instance must not inherit the
        # previous scenario's frozen slot limit. Found by the paired harness:
        # the budget mutation froze the cap at scenario 0's value.
        entry_config = self.config
        try:
            dates, defer_day = self._planning_clock(start, settings)
            if self.config.planned_cap_by_median_x:
                end_times = pd.to_datetime(locations["end_time"])
                if getattr(end_times.dt, "tz", None) is not None:
                    end_times = end_times.dt.tz_localize(None)
                median_x = float(
                    (
                        end_times.dt.normalize()
                        + pd.Timedelta(days=float(settings.unobserved_eol_days))
                        - dates[-1]
                    ).dt.days.median()
                )
                cap = None
                for upper, value in self.config.planned_cap_by_median_x:
                    if median_x <= float(upper):
                        cap = int(value)
                        break
                if cap is not None:
                    self.config = replace(
                        self.config,
                        optimizer=replace(self.config.optimizer, max_planned_count=cap),
                    )
            forecast = self._forecast(battery_data, locations, start, dates)
            full_costs = build_expected_cost_tables(
                forecast,
                locations,
                settings,
                dates,
                late_risk_multiplier=self.config.late_risk_multiplier,
                emergency_rank_scale=self.config.emergency_rank_scale,
            )
            # The expected-due budget is computed on the FULL fleet before any
            # exclusion. Demoting a battery from the slot allocation must not
            # shrink the budget its probability justified -- collapsing both at
            # once is the mechanism that killed every probability-knockdown
            # attempt before this.
            full_limit = scenario_planned_swap_limit(full_costs, self.config.optimizer)
            if full_limit is not None:
                self.config = replace(
                    self.config,
                    optimizer=replace(
                        self.config.optimizer,
                        max_planned_count=full_limit,
                        expected_due_multiplier=None,
                        max_planned_rate=None,
                    ),
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
            demoted: set[str] = set()
            summaries = getattr(forecast, "summaries", None)
            if (
                self.config.slot_demotion
                and summaries is not None
                and not summaries.empty
                and "slot_demote" in summaries.columns
            ):
                flagged = summaries.loc[
                    summaries["slot_demote"].astype(bool), "battery_id"
                ].astype(str)
                demoted = set(flagged)
            if demoted:
                keep = np.asarray(
                    [
                        row
                        for row in keep
                        if full_costs.battery_ids[row] not in demoted
                    ],
                    dtype=int,
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
            if self.config.capacity_repair:
                # Deterministic capacity post-pass: exact-replay repair of
                # daily/weekly limit hits, then light-day merges.
                plan = self._capacity_repair(
                    plan,
                    costs,
                    candidate_locations,
                    travel_costs,
                    settings,
                    start,
                    defer_day,
                )
            plan = self._restore_excluded(plan, excluded, defer_day)
            if self.config.selection_exchange:
                plan = self._selection_exchange(
                    plan,
                    full_costs,
                    forecast,
                    dates,
                    defer_day,
                    locations,
                    candidate_ids,
                    self.config.optimizer.max_planned_count,
                )
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
        finally:
            self.config = entry_config
