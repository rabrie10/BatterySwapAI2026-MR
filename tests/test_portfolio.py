from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from batteryswap_solution.forecast import (
    CONTRACT_VERSION,
    ForecastMetadata,
    RiskForecast,
)
from bsai.portfolio import (
    PortfolioForecaster,
    ScenarioIncidenceModel,
    scenario_feature_names,
    scenario_features,
)


class ConstantEstimator:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, rows):
        return np.full(len(rows), self.value)


def forecast() -> RiskForecast:
    start = pd.Timestamp("2026-01-01")
    dates = pd.date_range(start, periods=7, freq="D")
    ids = ["a", "b", "c"]
    profiles = {
        "a": np.linspace(0.0, 0.8, len(dates)),
        "b": np.linspace(0.0, 0.3, len(dates)),
        "c": np.linspace(0.0, 0.1, len(dates)),
    }
    curves = pd.DataFrame(
        [
            {"battery_id": battery, "forecast_date": day, "failure_cdf": value}
            for battery in ids
            for day, value in zip(dates, profiles[battery])
        ]
    )
    tail = pd.DataFrame(
        {
            "battery_id": ids,
            "prob_observed_after_horizon": [0.2, 0.7, 0.9],
            "mean_excess_rul_days_given_observed_after_horizon": [10.0] * 3,
            "prob_unobserved_eol": [0.0] * 3,
            "prob_no_observed_eol_by_horizon": [0.2, 0.7, 0.9],
        }
    )
    metadata = ForecastMetadata(
        CONTRACT_VERSION,
        "test/v1",
        start,
        dates[-1],
        6,
        dates[-1] + pd.Timedelta(days=30),
    )
    return RiskForecast(metadata, curves, tail, pd.DataFrame({"battery_id": ids}))


class FixedForecaster:
    model_version = "fixed/v1"
    last_cold_start = 0

    def predict(self, *args, **kwargs):
        return forecast()


class PortfolioTests(unittest.TestCase):
    def test_scenario_feature_schema_is_stable_and_finite(self):
        row = scenario_features(forecast())
        self.assertEqual(len(row), len(scenario_feature_names()))
        self.assertTrue(np.isfinite(row).all())

    def test_wrapper_keeps_ranking_separate_from_incidence(self):
        model = ScenarioIncidenceModel(
            ConstantEstimator(2.0),
            scenario_feature_names(),
            service_multiplier=1.0,
            service_buffer=0.0,
            minimum_budget=2,
        )
        wrapped = PortfolioForecaster(FixedForecaster(), model)
        result = wrapped.predict(None, None)
        summary = result.summaries.set_index("battery_id")
        self.assertEqual(wrapped.last_expected_due, 2.0)
        self.assertEqual(wrapped.last_service_budget, 2)
        self.assertEqual(int(summary.loc["a", "portfolio_rank"]), 1)
        self.assertEqual(int(summary.loc["c", "portfolio_rank"]), 3)
        self.assertTrue((summary["scenario_service_budget"] == 2).all())
        self.assertTrue((summary["scenario_candidate_pool"] == 10).all())


if __name__ == "__main__":
    unittest.main()
