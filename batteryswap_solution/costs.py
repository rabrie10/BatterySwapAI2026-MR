"""Evaluator-aligned expected timing costs for planning decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import RiskForecast


@dataclass(frozen=True)
class CostTables:
    battery_ids: tuple[str, ...]
    candidate_dates: pd.DatetimeIndex
    service_cost: np.ndarray
    defer_cost: np.ndarray
    event_pmf: np.ndarray
    horizon_event_probability: np.ndarray

    def take(self, indices: np.ndarray) -> "CostTables":
        indices = np.asarray(indices, dtype=int)
        return CostTables(
            battery_ids=tuple(self.battery_ids[index] for index in indices),
            candidate_dates=self.candidate_dates,
            service_cost=self.service_cost[indices],
            defer_cost=self.defer_cost[indices],
            event_pmf=self.event_pmf[indices],
            horizon_event_probability=self.horizon_event_probability[indices],
        )


def select_candidates(
    costs: CostTables,
    *,
    margin_hours: float = 24.0,
    max_candidates: int = 150,
) -> np.ndarray:
    """Batteries the optimizer could plausibly want to service.

    About 9.5 batteries per scenario actually reach EOL inside the window, out
    of roughly 420 alive, yet every search evaluation used to walk the whole
    fleet. Servicing helps only when the expected timing cost of a swap is not
    far above the expected cost of deferring, so everything well below that line
    can be deferred up front without changing the answer.

    ``margin_hours`` is deliberately generous -- 24 hours of expected timing cost
    is about 48 days of earliness -- so the reduction is a speedup rather than a
    decision. Ties are broken by the size of the gain, and the cap bounds the
    worst case on an unfamiliar split.
    """
    gain = costs.defer_cost - costs.service_cost.min(axis=1)
    keep = np.flatnonzero(gain > -abs(float(margin_hours)))
    if keep.size > max_candidates:
        order = np.argsort(-gain[keep])[:max_candidates]
        keep = np.sort(keep[order])
    return keep


def _setting(settings, name: str) -> float:
    return float(getattr(settings, name))


def build_expected_cost_tables(
    forecast: RiskForecast,
    locations: pd.DataFrame,
    settings,
    candidate_dates: pd.DatetimeIndex,
    *,
    late_risk_multiplier: float = 1.0,
    emergency_rank_scale: float = 1.0,
) -> CostTables:
    """Compute expected score contributions for every battery/service date.

    Deferred cost includes the evaluator's deterministic sorted emergency queue
    through each battery's expected rank. This captures the dominant coupling
    while keeping the assignment model deterministic and compact.
    """

    id_column = "battery_id" if "battery_id" in locations else "battery"
    loc = locations.copy()
    loc[id_column] = loc[id_column].astype(str)
    loc = loc.set_index(id_column, drop=False)
    battery_ids = tuple(loc.index)
    dates = pd.DatetimeIndex(pd.to_datetime(candidate_dates)).normalize()

    cdf = (
        forecast.curves.pivot(
            index="battery_id", columns="forecast_date", values="failure_cdf"
        )
        .reindex(index=battery_ids, columns=dates)
        .to_numpy(dtype=float)
    )
    pmf = np.diff(np.concatenate([np.zeros((len(battery_ids), 1)), cdf], axis=1), axis=1)
    pmf = np.clip(pmf, 0.0, 1.0)

    tail = forecast.tail.set_index("battery_id").reindex(battery_ids)
    observed_tail = tail["prob_observed_after_horizon"].to_numpy(dtype=float)
    mean_excess = tail[
        "mean_excess_rul_days_given_observed_after_horizon"
    ].to_numpy(dtype=float)
    unobserved = tail["prob_unobserved_eol"].to_numpy(dtype=float)

    early = _setting(settings, "early_replacement_penalty_daily")
    late = _setting(settings, "late_replacement_penalty_daily") * float(late_risk_multiplier)
    event_day = np.arange(len(dates), dtype=float)[None, :, None]
    service_day = np.arange(len(dates), dtype=float)[None, None, :]
    delta = event_day - service_day
    event_loss = early * np.maximum(delta, 0.0) + late * np.maximum(-delta, 0.0)
    service_cost = np.sum(pmf[:, :, None] * event_loss, axis=1)

    horizon_index = float(len(dates) - 1)
    service_offsets = np.arange(len(dates), dtype=float)[None, :]
    service_cost += observed_tail[:, None] * early * (
        horizon_index - service_offsets + mean_excess[:, None]
    )

    end_times = pd.to_datetime(loc["end_time"])
    proxy_dates = (end_times + pd.to_timedelta(_setting(settings, "unobserved_eol_days"), unit="D")).dt.normalize()
    proxy_offsets = (
        (proxy_dates - dates[0]) / pd.Timedelta(days=1)
    ).to_numpy(dtype=float)[:, None]
    proxy_delta = proxy_offsets - service_offsets
    proxy_loss = early * np.maximum(proxy_delta, 0.0) + late * np.maximum(-proxy_delta, 0.0)
    service_cost += unobserved[:, None] * proxy_loss

    horizon_probability = cdf[:, -1]
    horizon_end = dates[-1]
    emergency_start = horizon_end + pd.Timedelta(days=(6 - horizon_end.weekday()))
    emergency_start_offset = float((emergency_start - dates[0]) / pd.Timedelta(days=1))

    # Emergency visits occur in sorted battery-ID order, one per day. Under an
    # independent marginal approximation, expected rank is the sum of earlier
    # batteries' probabilities of being due within the horizon.
    #
    # That sum runs over every battery, but the queue only ever contains the
    # ones the plan misses. Summed over the fleet it comes to about 6.4 on train
    # against a realised 3.6 misses per scenario, so the raw rank inflates the
    # cost of deferring -- by 10 hours per day of rank -- and quietly biases
    # every decision towards servicing. The scale makes that assumption
    # explicit and tunable.
    sorted_positions = np.argsort(np.asarray(battery_ids, dtype=str), kind="stable")
    expected_rank = np.zeros(len(battery_ids), dtype=float)
    cumulative = 0.0
    for position in sorted_positions:
        expected_rank[position] = cumulative
        cumulative += horizon_probability[position]
    emergency_offsets = emergency_start_offset + emergency_rank_scale * expected_rank
    late_days = np.maximum(emergency_offsets[:, None] - np.arange(len(dates))[None, :], 0.0)
    expected_emergency_lateness = np.sum(pmf * late_days, axis=1) * late

    # Isolated emergency operation time is added by the optimizer, where the
    # scenario travel matrix is available. Keeping timing and operations
    # separate also avoids double-counting planned logistics.
    defer_cost = expected_emergency_lateness

    return CostTables(
        battery_ids=battery_ids,
        candidate_dates=dates,
        service_cost=service_cost,
        defer_cost=defer_cost,
        event_pmf=pmf,
        horizon_event_probability=horizon_probability,
    )


def isolated_emergency_costs(
    locations: pd.DataFrame,
    travel_costs: pd.DataFrame,
    settings,
    battery_ids: tuple[str, ...],
) -> np.ndarray:
    """Operational cost of each battery as a one-stop emergency workday."""

    id_column = "battery_id" if "battery_id" in locations else "battery"
    building_column = "building_id" if "building_id" in locations else "building"
    room_column = "room_id" if "room_id" in locations else "room"
    loc = locations.copy()
    loc[id_column] = loc[id_column].astype(str)
    loc = loc.set_index(id_column)
    travel = travel_costs.set_index(["from", "to"])["hours"]
    base = str(settings.base_location)
    base_room = str(settings.base_room)
    costs = np.zeros(len(battery_ids), dtype=float)
    for index, battery_id in enumerate(battery_ids):
        building = str(loc.loc[battery_id, building_column])
        room = str(loc.loc[battery_id, room_column])
        work = _setting(settings, "time_per_battery_hours")
        if building != base:
            work += _setting(settings, "time_per_building_change_hours")
            work += float(travel.loc[(base, building)])
        if room != base_room:
            work += _setting(settings, "time_per_room_change_hours")
        work += float(travel.loc[(building, base)])
        overtime = max(work - _setting(settings, "overtime_start"), 0.0)
        costs[index] = work + overtime * _setting(settings, "overtime_penalty_factor")
        if work > _setting(settings, "worker_limit_daily_hours"):
            costs[index] += _setting(settings, "worker_limit_daily_penalty")
    return costs
