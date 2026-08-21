"""A per-battery decision layer, as an alternative to the joint search.

`batteryswap_solution.CompetitionPlanner` runs a large-neighbourhood search over
the whole plan against an expected-cost objective. Measured on the first twelve
train scenarios, that objective believes 1176.8 where the evaluator charges
2328.2, with a correlation of only 0.613. A search that good at optimising a
number that wrong compounds the error: it confidently assembles plans whose
believed savings do not exist, and it services batteries a per-battery rule
would leave alone.

So this planner deliberately does less. Each battery gets the day that minimises
its own expected timing cost, or is deferred if deferring is cheaper; the
routing layer then orders whatever survives. No joint objective, nothing to
compound. It is the decision rule that scored 1567.6 out-of-fold while the joint
search scored 2526.0 on the same forecasts.

Which one ships is decided by `tools/validate_v6.py`, not by preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from batteryswap_public.evaluate import check_plan_valid
from batteryswap_public.interfaces import Planner

from batteryswap_solution.costs import build_expected_cost_tables
from batteryswap_solution.forecast import (
    RiskForecaster,
    VoltageTrendForecaster,
    validate_forecast,
)
from batteryswap_solution.routing import order_assignments

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimplePlannerConfig:
    late_risk_multiplier: float = 1.0
    emergency_rank_scale: float = 1.0
    # Charged against every swap to represent the visit it implies. The routing
    # layer recovers the sharing between co-located batteries, so pricing a full
    # standalone trip here would refuse swaps that are nearly free in practice.
    work_cost_hours: float = 0.25
    include_emergency_operations: bool = True


class PerBatteryPlanner(Planner):
    """Decide each battery on its own expected cost, then route."""

    def __init__(
        self,
        forecaster: RiskForecaster | None = None,
        config: SimplePlannerConfig | None = None,
    ) -> None:
        self.forecaster = forecaster or VoltageTrendForecaster()
        self.fallback_forecaster = VoltageTrendForecaster()
        self.config = config or SimplePlannerConfig()

    @staticmethod
    def _clock(start: pd.Timestamp, settings) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
        end = (start + pd.Timedelta(days=float(settings.planning_window_days))).normalize()
        return pd.date_range(start, end, freq="D", inclusive="both"), end + pd.Timedelta(days=1)

    @staticmethod
    def _all_defer(locations: pd.DataFrame, defer_day: pd.Timestamp) -> pd.DataFrame:
        id_column = "battery_id" if "battery_id" in locations else "battery"
        return pd.DataFrame(
            {
                "day": pd.DatetimeIndex([defer_day] * len(locations)),
                "battery": sorted(locations[id_column].astype(str)),
            }
        ).reset_index(drop=True)

    def _forecast(self, battery_data, locations, start, dates):
        observation_end = pd.Timestamp(pd.to_datetime(locations["end_time"]).max())
        if observation_end.tzinfo is not None:
            observation_end = observation_end.tz_localize(None)
        battery_ids = locations["battery"].astype(str).tolist()
        kwargs = dict(
            prediction_origin=start,
            horizon_days=len(dates) - 1,
            evaluation_observation_end=observation_end,
        )
        try:
            return validate_forecast(
                self.forecaster.predict(battery_data, locations, **kwargs),
                battery_ids,
                dates,
            )
        except Exception:
            LOGGER.exception("Primary forecaster failed; using voltage trend fallback")
            return validate_forecast(
                self.fallback_forecaster.predict(battery_data, locations, **kwargs),
                battery_ids,
                dates,
            )

    def plan(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
    ) -> pd.DataFrame:
        from batteryswap_solution.costs import isolated_emergency_costs
        from batteryswap_solution.planner import infer_scenario_start

        start = infer_scenario_start(battery_data)
        dates, defer_day = self._clock(start, settings)
        fallback = self._all_defer(locations, defer_day)
        try:
            forecast = self._forecast(battery_data, locations, start, dates)
            costs = build_expected_cost_tables(
                forecast,
                locations,
                settings,
                dates,
                late_risk_multiplier=self.config.late_risk_multiplier,
                emergency_rank_scale=self.config.emergency_rank_scale,
            )
            defer = costs.defer_cost.copy()
            if self.config.include_emergency_operations:
                defer = defer + costs.horizon_event_probability * isolated_emergency_costs(
                    locations, travel_costs, settings, costs.battery_ids
                )
            service = costs.service_cost + float(self.config.work_cost_hours)

            best_day = service.argmin(axis=1)
            best_cost = service[np.arange(len(best_day)), best_day]
            assignments: dict[str, pd.Timestamp | None] = {}
            for index, battery_id in enumerate(costs.battery_ids):
                assignments[battery_id] = (
                    None
                    if defer[index] <= best_cost[index]
                    else costs.candidate_dates[int(best_day[index])]
                )

            # The evaluator asserts when work lands on the final calendar day of
            # its padded week; never emit that day.
            if dates[-1].weekday() == 6:
                for battery_id, day in assignments.items():
                    if day is not None and day == dates[-1]:
                        assignments[battery_id] = dates[-2]

            priority = {
                battery_id: float(costs.horizon_event_probability[index])
                for index, battery_id in enumerate(costs.battery_ids)
            }
            plan = order_assignments(
                assignments,
                locations,
                travel_costs,
                str(settings.base_location),
                defer_day,
                priority=priority,
            )
            check_plan_valid(plan, locations, start_time=start)
            return plan[["day", "battery"]].reset_index(drop=True)
        except Exception:
            LOGGER.exception("Per-battery planning failed; returning all-defer")
            check_plan_valid(fallback, locations, start_time=start)
            return fallback
