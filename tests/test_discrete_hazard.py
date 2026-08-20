from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd
from batteryswap_solution.forecast import validate_forecast
from src.risk.discrete_hazard import (
    DiscreteHazardForecaster,
    HAZARD_FEATURES,
    build_hazard_table,
    hazards_to_cdf,
    hazards_to_survival,
)
from src.risk.features import build_feature_series, lookup_asof
from src.risk.model import HorizonIsotonicCalibrator


def _landmarks(duration: float, event: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "device_id": ["d1"], "building_id": ["b1"], "cutoff": [pd.Timestamp("2025-01-01")],
            "duration_days": [duration], "event": [event], "sample_weight": [1.0],
            "latest_voltage": [2.8], "voltage_slope_28d": [-0.001],
            "voltage_slope_90d": [-0.001], "voltage_std_28d": [0.01],
            "crossing_days_extrapolated": [400.0], "temp_mean_28d": [20.0],
            "frac_low_voltage_28d": [0.0], "completeness_90d": [1.0], "age_days": [100.0],
            "days_since_last_reading": [0.0], "building_loo_latest_voltage": [2.9],
            "building_loo_voltage_slope_28d": [-0.001], "not_yet_deployed": [0.0],
            "cold_start": [0.0],
        }
    )


def _identity_calibrator() -> HorizonIsotonicCalibrator:
    horizons = (7.0, 42.0, 365.0)
    return HorizonIsotonicCalibrator(
        horizons=horizons,
        thresholds=tuple((0.0, 1.0) for _ in horizons),
        values=tuple((0.0, 1.0) for _ in horizons),
    )


class ConstantHazardModel:
    """Pickleable deterministic model used to isolate forecast-contract math."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(x), 0.01, dtype=float)
        return np.column_stack([1.0 - probability, probability])


class HazardTableTests(unittest.TestCase):
    def test_observed_event_has_exactly_one_positive_and_no_rows_after_eol(self):
        expanded = build_hazard_table(_landmarks(3.2, 1), max_horizon=10)
        self.assertEqual(expanded["horizon_day"].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(int(expanded["label"].sum()), 1)
        self.assertEqual(float(expanded.loc[expanded["label"] == 1, "horizon_day"].iloc[0]), 4.0)

    def test_event_exactly_on_boundary_is_in_that_interval(self):
        expanded = build_hazard_table(_landmarks(3.0, 1), max_horizon=10)
        self.assertEqual(expanded["horizon_day"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(expanded["label"].tolist(), [0, 0, 1])

    def test_censoring_stops_at_last_fully_known_interval(self):
        expanded = build_hazard_table(_landmarks(3.8, 0), max_horizon=10)
        self.assertEqual(expanded["horizon_day"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(int(expanded["label"].sum()), 0)

    def test_censoring_before_next_horizon_is_masked_not_negative(self):
        expanded = build_hazard_table(_landmarks(0.8, 0), max_horizon=10)
        self.assertTrue(expanded.empty)

    def test_normalized_weights_give_landmark_original_total_weight(self):
        expanded = build_hazard_table(_landmarks(4.0, 0), weighting="normalized")
        self.assertAlmostEqual(float(expanded["weight"].sum()), 1.0)


class HazardMathTests(unittest.TestCase):
    def test_hazard_to_survival_and_cdf(self):
        hazards = np.array([0.1, 0.2, 0.5])
        survival = hazards_to_survival(hazards)
        cdf = hazards_to_cdf(hazards)
        np.testing.assert_allclose(survival, [0.9, 0.72, 0.36])
        np.testing.assert_allclose(cdf, [0.1, 0.28, 0.64])
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))


class HazardForecastTests(unittest.TestCase):
    def _forecaster(self) -> DiscreteHazardForecaster:
        return DiscreteHazardForecaster(
            model=ConstantHazardModel(), calibrator=_identity_calibrator(), params={}, weighting="normalized"
        )

    @staticmethod
    def _locations(origin: pd.Timestamp, start: pd.Timestamp | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "battery": ["d1"], "building": ["b1"], "room": ["r1"],
                "start_time": [start if start is not None else origin - pd.Timedelta("100D")],
                "end_time": [origin + pd.Timedelta("100D")],
            }
        )

    def test_contract_mass_cold_start_and_serialization(self):
        origin = pd.Timestamp("2025-01-01")
        forecaster = pickle.loads(pickle.dumps(self._forecaster()))
        empty = pd.DataFrame(columns=["device_id", "end_time", "voltage", "temperature"])
        forecast = forecaster.predict(
            empty, self._locations(origin), prediction_origin=origin, horizon_days=42,
            evaluation_observation_end=origin + pd.Timedelta("100D"),
        )
        dates = pd.date_range(origin, origin + pd.Timedelta("42D"), freq="D")
        validated = validate_forecast(forecast, ["d1"], dates)
        self.assertTrue(bool(validated.summaries.iloc[0]["cold_start"]))
        final = float(validated.curves.iloc[-1]["failure_cdf"])
        tail = validated.tail.iloc[0]
        self.assertAlmostEqual(
            final + tail["prob_observed_after_horizon"] + tail["prob_unobserved_eol"], 1.0
        )

    def test_official_predeployment_prediction(self):
        origin = pd.Timestamp("2025-01-01")
        empty = pd.DataFrame(columns=["device_id", "end_time", "voltage", "temperature"])
        forecast = self._forecaster().predict(
            empty, self._locations(origin, origin + pd.Timedelta("20D")),
            prediction_origin=origin, horizon_days=42,
            evaluation_observation_end=origin + pd.Timedelta("100D"),
        )
        self.assertEqual(len(forecast.curves), 43)
        self.assertTrue(np.isfinite(forecast.curves["failure_cdf"]).all())

    def test_future_readings_do_not_change_landmark_features(self):
        dates = pd.date_range("2025-01-01", periods=20, freq="D")
        frame = pd.DataFrame({
            "device_id": "d1", "end_time": dates,
            "voltage": 3.0 - 0.01 * np.arange(20), "temperature": 20.0,
        })
        cutoff = dates[9]
        full = lookup_asof(build_feature_series(frame)["d1"], cutoff)
        truncated = lookup_asof(build_feature_series(frame.iloc[:10])["d1"], cutoff)
        for key in full:
            if pd.isna(full[key]) and pd.isna(truncated[key]):
                continue
            self.assertAlmostEqual(full[key], truncated[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
