"""Regression checks for production settings and observation-horizon semantics."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.validate_v6 import build_forecaster
from script import build_planner_config
from bsai import config as shipped
from bsai.features import FeatureContext
from bsai.forecaster import HazardForecaster
from bsai.wiener import WienerModel
from batteryswap_solution.forecast import validate_forecast


class _PositiveRiskModel:
    model_version = "regression-fixture/v1"
    horizons = (1, 7, 14, 30)
    cdf_at = WienerModel.cdf_at

    def context(self):
        return FeatureContext(climatology=np.full(12, 21.0))

    def predict_grid(self, features, remaining, devices=None):
        return np.tile([.2, .4, .6, .8], (len(features), 1))


class ConfigurationTests(unittest.TestCase):
    def test_stale_veto_cannot_create_events_beyond_observation_end(self):
        origin = pd.Timestamp("2026-03-02")
        stamps = pd.date_range(end=origin - pd.Timedelta(days=20), periods=60 * 24, freq="h")
        telemetry = pd.DataFrame({"voltage": 2.45, "temperature": 21.0},
                                 index=pd.MultiIndex.from_product(
                                     [["d_stale"], stamps], names=["device_id", "end_time"]))
        dates = pd.date_range(origin, periods=15)
        for remaining in [3, 14]:
            with self.subTest(remaining=remaining):
                end = origin + pd.Timedelta(days=remaining)
                locations = pd.DataFrame({"battery": ["d_stale"], "end_time": [end]})
                forecaster = HazardForecaster(_PositiveRiskModel(), use_split_climatology=False)
                forecast = forecaster.predict(telemetry, locations, prediction_origin=origin,
                                              horizon_days=14, evaluation_observation_end=end)
                self.assertEqual(forecaster.last_vetoed, 1)
                validated = validate_forecast(forecast, ["d_stale"], dates)
                self.assertEqual(float(validated.tail.iloc[0].prob_observed_after_horizon), 0.0)
                self.assertEqual(float(validated.tail.iloc[0].prob_unobserved_eol), 1.0)
                self.assertTrue((validated.curves.failure_cdf == 0.0).all())

    def test_production_validation_applies_requested_volatility_scale(self):
        model = SimpleNamespace(volatility_scale=1.4)
        args = SimpleNamespace(production=True, model="unused", volatility_scale=0.8,
                               probability_scale=1.2)
        with patch("tools.validate_v6.joblib.load", return_value=model), patch(
            "tools.validate_v6.HazardForecaster"
        ) as adapter:
            build_forecaster(args, {})
        self.assertEqual(model.volatility_scale, 0.8)
        adapter.assert_called_once_with(model, probability_scale=1.2)

    def test_submission_config_uses_shared_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            config = build_planner_config(shipped.SOLVER_SECONDS,
                                          shipped.LOCAL_SEARCH_EVALUATIONS,
                                          shipped.UNCERTAIN_SEARCH_EVALUATIONS)
        self.assertEqual(config.robust_emergency_samples, shipped.ROBUST_EMERGENCY_SAMPLES)
        self.assertEqual(config.candidate_margin_hours, shipped.CANDIDATE_MARGIN_HOURS)


if __name__ == "__main__":
    unittest.main()
