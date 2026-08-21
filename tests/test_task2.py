from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from batteryswap_public.evaluate import EvaluationSettings, check_plan_valid, evaluate_plan

from batteryswap_solution.costs import build_expected_cost_tables, portfolio_keep_indices
from batteryswap_solution.forecast import (
    CONTRACT_VERSION,
    ForecastContractError,
    ForecastMetadata,
    RiskForecast,
    validate_forecast,
)
from batteryswap_solution.optimizer import OptimizationConfig, planned_swap_limit
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import replay_operational_cost
from batteryswap_solution.routing import route_buildings


START = pd.Timestamp("2026-03-02")


def settings(**updates) -> EvaluationSettings:
    values = {
        "base_location": "b_base",
        "base_room": "r_base",
        "planning_window_days": 6,
        "worker_limit_weekly_hours": 23.9,
    }
    values.update(updates)
    return EvaluationSettings(**values)


def locations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery": ["d_urgent", "d_defer", "d_bundle"],
            "building": ["b_far", "b_base", "b_far"],
            "room": ["r_a", "r_base", "r_a"],
            "start_time": [START - pd.Timedelta(days=100)] * 3,
            "end_time": [START + pd.Timedelta(days=30)] * 3,
        }
    )


def travel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"from": "b_base", "to": "b_base", "hours": 0.03},
            {"from": "b_base", "to": "b_far", "hours": 1.0},
            {"from": "b_far", "to": "b_base", "hours": 1.0},
            {"from": "b_far", "to": "b_far", "hours": 0.03},
        ]
    )


def battery_data() -> pd.DataFrame:
    rows = []
    for battery, voltage in [("d_urgent", 2.42), ("d_defer", 2.95), ("d_bundle", 2.45)]:
        for hour in range(48):
            rows.append(
                {
                    "device_id": battery,
                    "end_time": START - pd.Timedelta(hours=47 - hour),
                    "voltage": voltage,
                    "temperature": 21.0,
                }
            )
    return pd.DataFrame(rows).set_index(["device_id", "end_time"]).sort_index()


def fixed_forecast() -> RiskForecast:
    dates = pd.date_range(START, START + pd.Timedelta(days=6), freq="D")
    profiles = {
        "d_urgent": np.array([0.0, 0.20, 0.55, 0.80, 0.90, 0.95, 0.97]),
        "d_bundle": np.array([0.0, 0.10, 0.35, 0.60, 0.78, 0.88, 0.92]),
        "d_defer": np.array([0.0, 0.0, 0.0, 0.002, 0.004, 0.006, 0.01]),
    }
    curve_rows = []
    tail_rows = []
    for battery, cdf in profiles.items():
        curve_rows.extend(
            {"battery_id": battery, "forecast_date": day, "failure_cdf": value}
            for day, value in zip(dates, cdf)
        )
        tail_rows.append(
            {
                "battery_id": battery,
                "prob_observed_after_horizon": 1.0 - cdf[-1],
                "mean_excess_rul_days_given_observed_after_horizon": 10.0,
                "prob_unobserved_eol": 0.0,
                "prob_no_observed_eol_by_horizon": 1.0 - cdf[-1],
            }
        )
    metadata = ForecastMetadata(
        contract_version=CONTRACT_VERSION,
        model_version="test/v1",
        prediction_origin=START,
        forecast_end_date=dates[-1],
        horizon_days=6,
        evaluation_observation_end=START + pd.Timedelta(days=30),
    )
    return RiskForecast(metadata, pd.DataFrame(curve_rows), pd.DataFrame(tail_rows), pd.DataFrame())


class FixedForecaster:
    def predict(self, *args, **kwargs):
        return fixed_forecast()


class ForecastContractTests(unittest.TestCase):
    def test_contract_canonicalizes_tiny_monotonicity_error(self):
        forecast = fixed_forecast()
        forecast.curves.loc[forecast.curves.index[3], "failure_cdf"] -= 1e-9
        dates = pd.date_range(START, START + pd.Timedelta(days=6), freq="D")
        result = validate_forecast(
            forecast, ["d_urgent", "d_defer", "d_bundle"], dates
        )
        urgent = result.curves[result.curves["battery_id"] == "d_urgent"]
        self.assertTrue((urgent["failure_cdf"].diff().fillna(0) >= 0).all())

    def test_contract_rejects_missing_battery(self):
        forecast = fixed_forecast()
        forecast.curves.drop(
            forecast.curves[forecast.curves["battery_id"] == "d_bundle"].index,
            inplace=True,
        )
        dates = pd.date_range(START, START + pd.Timedelta(days=6), freq="D")
        with self.assertRaises(ForecastContractError):
            validate_forecast(forecast, list(locations()["battery"]), dates)


class CostAndRoutingTests(unittest.TestCase):
    def test_portfolio_budget_keeps_only_independently_ranked_candidates(self):
        base = fixed_forecast()
        forecast = RiskForecast(
            base.metadata,
            base.curves,
            base.tail,
            pd.DataFrame(
                {
                    "battery_id": ["d_urgent", "d_defer", "d_bundle"],
                    "portfolio_rank": [1, 3, 2],
                    "scenario_service_budget": [2, 2, 2],
                    "scenario_candidate_pool": [2, 2, 2],
                }
            ),
        )
        keep = portfolio_keep_indices(
            forecast, ("d_urgent", "d_defer", "d_bundle")
        )
        np.testing.assert_array_equal(keep, np.array([0, 2]))

    def test_ordinary_forecast_has_no_portfolio_gate(self):
        self.assertIsNone(
            portfolio_keep_indices(
                fixed_forecast(), ("d_urgent", "d_defer", "d_bundle")
            )
        )

    def test_planned_swap_limit_scales_with_scenario_size(self):
        self.assertIsNone(planned_swap_limit(460, None))
        self.assertEqual(planned_swap_limit(460, 0.04), 19)
        self.assertEqual(planned_swap_limit(460, None, 15), 15)
        self.assertEqual(planned_swap_limit(460, 0.04, 12), 12)
        with self.assertRaises(ValueError):
            planned_swap_limit(460, 0.0)
        with self.assertRaises(ValueError):
            planned_swap_limit(460, None, 0)

    def test_asymmetric_timing_cost_prefers_early_quantile(self):
        dates = pd.date_range(START, START + pd.Timedelta(days=6), freq="D")
        forecast = validate_forecast(
            fixed_forecast(), list(locations()["battery"]), dates
        )
        costs = build_expected_cost_tables(forecast, locations(), settings(), dates)
        urgent = costs.battery_ids.index("d_urgent")
        best_day = int(np.argmin(costs.service_cost[urgent]))
        self.assertLessEqual(best_day, 2)

    def test_held_karp_route_beats_bad_alphabetical_order(self):
        matrix = {
            ("b0", "b0"): 0.0,
            ("b0", "b1"): 1.0,
            ("b1", "b0"): 1.0,
            ("b0", "b2"): 3.0,
            ("b2", "b0"): 3.0,
            ("b0", "b3"): 1.0,
            ("b3", "b0"): 1.0,
            ("b1", "b2"): 1.0,
            ("b2", "b1"): 1.0,
            ("b2", "b3"): 1.0,
            ("b3", "b2"): 1.0,
            ("b1", "b3"): 3.0,
            ("b3", "b1"): 3.0,
        }
        self.assertEqual(
            route_buildings(["b1", "b2", "b3"], "b0", matrix),
            ["b1", "b2", "b3"],
        )


class PlannerTests(unittest.TestCase):
    def test_unobserved_eol_is_free_when_deferred_but_uses_proxy_when_planned(self):
        deferred_day = START + pd.Timedelta(days=7)
        all_deferred = pd.DataFrame(
            {
                "day": [deferred_day] * 3,
                "battery": sorted(locations()["battery"]),
            }
        )
        missing_eol = pd.Series(pd.NaT, index=locations()["battery"])
        _, _, deferred_score = evaluate_plan(
            all_deferred,
            locations(),
            travel(),
            settings(),
            eol_times=missing_eol,
            start_time=START,
            verbose=0,
        )
        self.assertEqual(float(deferred_score["early_swap"]), 0.0)
        self.assertEqual(float(deferred_score["battery_swap"]), 0.0)

        planned = all_deferred.copy()
        planned.loc[planned["battery"] == "d_defer", "day"] = START
        planned = planned.sort_values("day").reset_index(drop=True)
        _, _, planned_score = evaluate_plan(
            planned,
            locations(),
            travel(),
            settings(),
            eol_times=missing_eol,
            start_time=START,
            verbose=0,
        )
        self.assertGreater(float(planned_score["early_swap"]), 0.0)

    def test_horizon_endpoint_is_inclusive(self):
        endpoint_settings = settings(planning_window_days=5)
        endpoint = START + pd.Timedelta(days=5)
        plan = pd.DataFrame(
            {
                "day": [endpoint, endpoint + pd.Timedelta(days=1), endpoint + pd.Timedelta(days=1)],
                "battery": ["d_defer", "d_bundle", "d_urgent"],
            }
        )
        eol = pd.Series(
            {"d_defer": endpoint, "d_bundle": pd.NaT, "d_urgent": pd.NaT}
        )
        _, _, score = evaluate_plan(
            plan,
            locations(),
            travel(),
            endpoint_settings,
            eol_times=eol,
            start_time=START,
            verbose=0,
        )
        self.assertEqual(float(score["late_swap"]), 0.0)
        self.assertEqual(float(score["early_swap"]), 0.0)
        self.assertEqual(float(score["battery_swap"]), 0.25)

    def test_daily_limit_is_strict_but_weekly_limit_hits_at_equality(self):
        threshold_settings = settings(
            worker_limit_daily_hours=0.28,
            worker_limit_weekly_hours=0.28,
        )
        plan = pd.DataFrame(
            {
                "day": [START, START + pd.Timedelta(days=7), START + pd.Timedelta(days=7)],
                "battery": ["d_defer", "d_bundle", "d_urgent"],
            }
        )
        _, _, score = evaluate_plan(
            plan,
            locations(),
            travel(),
            threshold_settings,
            eol_times=None,
            start_time=START,
            verbose=0,
        )
        self.assertEqual(float(score["daily_limit"]), 0.0)
        self.assertEqual(float(score["weekly_limit"]), 100.0)

    def test_planner_returns_complete_valid_batched_plan(self):
        planner = CompetitionPlanner(
            forecaster=FixedForecaster(),
            config=PlannerConfig(
                local_search_evaluations=12,
                optimizer=OptimizationConfig(solver_seconds=1.0),
            ),
        )
        plan = planner.plan(battery_data(), locations(), travel(), settings())
        check_plan_valid(plan, locations(), start_time=START)
        self.assertEqual(set(plan["battery"]), set(locations()["battery"]))
        self.assertEqual(len(plan), len(locations()))
        urgent_day = plan.set_index("battery").loc["d_urgent", "day"]
        defer_day = plan.set_index("battery").loc["d_defer", "day"]
        self.assertLessEqual(urgent_day, START + pd.Timedelta(days=6))
        self.assertGreater(defer_day, START + pd.Timedelta(days=6))

        _, _, score = evaluate_plan(
            plan,
            locations(),
            travel(),
            settings(),
            eol_times=None,
            start_time=START,
            verbose=0,
        )
        self.assertEqual(float(score["daily_limit"]), 0.0)
        self.assertEqual(float(score["weekly_limit"]), 0.0)
        fast = replay_operational_cost(
            plan, locations(), travel(), settings(), START
        )
        self.assertAlmostEqual(fast["total_cost"], float(score["total_cost"]), places=9)

    def test_emergency_replay_matches_official_operational_components(self):
        plan = pd.DataFrame(
            {
                "day": [START, START + pd.Timedelta(days=7), START + pd.Timedelta(days=7)],
                "battery": ["d_urgent", "d_bundle", "d_defer"],
            }
        )
        eol = pd.Series(
            {
                "d_urgent": START + pd.Timedelta(days=20),
                "d_defer": START + pd.Timedelta(days=2),
                "d_bundle": pd.NaT,
            }
        )
        _, _, official = evaluate_plan(
            plan,
            locations(),
            travel(),
            settings(),
            eol_times=eol,
            start_time=START,
            verbose=0,
        )
        fast = replay_operational_cost(
            plan,
            locations(),
            travel(),
            settings(),
            START,
            emergency_batteries=["d_defer"],
        )
        operation_columns = [
            "battery_swap",
            "building_change",
            "room_change",
            "travel",
            "overtime",
            "daily_limit",
            "weekly_limit",
        ]
        self.assertAlmostEqual(
            fast["total_cost"],
            float(official[operation_columns].sum()),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
