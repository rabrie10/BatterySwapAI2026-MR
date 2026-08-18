from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from batteryswap_solution.forecast import validate_forecast

from src.risk.cutoffs import (
    assign_building_folds,
    build_cutoff_grid,
    build_example_table,
    terminal_times,
)
from src.risk.features import (
    build_feature_series,
    leave_one_out_building_features,
    lookup_asof,
)
from src.risk.model import (
    CURATED_FEATURES,
    PlattCalibrator,
    Task1Forecaster,
    FeatureTransform,
    masked_labels_at_horizon,
    select_and_fit,
)


def _synthetic_readings(device_id: str, start: pd.Timestamp, n_days: int, voltage: np.ndarray, temp: float = 20.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "device_id": device_id,
            "end_time": dates,
            "voltage": voltage,
            "temperature": temp,
        }
    )


class FeatureCausalityTests(unittest.TestCase):
    def test_rolling_features_do_not_leak_future_data(self):
        n = 100
        voltage = 3.0 - 0.004 * np.arange(n)
        frame = _synthetic_readings("d_x", pd.Timestamp("2024-01-01"), n, voltage)
        cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(days=49)

        full_series = build_feature_series(frame)["d_x"]
        truncated_frame = frame[frame["end_time"] <= cutoff]
        truncated_series = build_feature_series(truncated_frame)["d_x"]

        full_row = lookup_asof(full_series, cutoff)
        truncated_row = lookup_asof(truncated_series, cutoff)

        self.assertEqual(set(full_row) & set(truncated_row), set(truncated_row))
        for key, truncated_value in truncated_row.items():
            full_value = full_row[key]
            if pd.isna(full_value) and pd.isna(truncated_value):
                continue
            self.assertAlmostEqual(full_value, truncated_value, places=9, msg=key)

    def test_lookup_asof_uses_latest_available_row_before_gap(self):
        n = 30
        voltage = np.full(n, 3.0)
        frame = _synthetic_readings("d_gap", pd.Timestamp("2024-01-01"), n, voltage)
        # Sensor goes dark from 2024-01-20 onward; cutoff falls inside that gap.
        frame = frame[frame["end_time"] <= pd.Timestamp("2024-01-19")]
        series = build_feature_series(frame)["d_gap"]

        cutoff = pd.Timestamp("2024-01-28")
        row = lookup_asof(series, cutoff)
        self.assertEqual(row["days_since_last_reading"], 9.0)

    def test_lookup_asof_before_any_data_is_all_nan(self):
        series = build_feature_series(_synthetic_readings("d_y", pd.Timestamp("2024-03-01"), 10, np.full(10, 3.0)))["d_y"]
        row = lookup_asof(series, pd.Timestamp("2024-01-01"))
        self.assertTrue(all(pd.isna(v) for k, v in row.items() if k not in ("history_days_available", "n_readings_total")))
        self.assertEqual(row["n_readings_total"], 0.0)


class LeaveOneOutTests(unittest.TestCase):
    def test_building_loo_excludes_self_and_matches_manual_mean(self):
        table = pd.DataFrame(
            {
                "cutoff": [pd.Timestamp("2025-01-01")] * 4,
                "building_id": ["b1", "b1", "b1", "b2"],
                "latest_voltage": [3.0, 2.8, 2.6, 2.9],
                "voltage_slope_28d": [np.nan] * 4,
                "voltage_slope_90d": [np.nan] * 4,
                "frac_low_voltage_28d": [np.nan] * 4,
                "crossing_days_extrapolated": [np.nan] * 4,
                "history_days_available": [np.nan] * 4,
            }
        )
        out = leave_one_out_building_features(table)
        # Row 0 (voltage=3.0) peers are rows 1,2 (2.8, 2.6) -> mean 2.7
        self.assertAlmostEqual(out.loc[0, "building_loo_latest_voltage"], 2.7)
        self.assertAlmostEqual(out.loc[1, "building_loo_latest_voltage"], (3.0 + 2.6) / 2)
        # Single-member building has no peers -> NaN
        self.assertTrue(pd.isna(out.loc[3, "building_loo_latest_voltage"]))


class MaskedLabelBoundaryTests(unittest.TestCase):
    def test_event_exactly_at_horizon_counts_as_observed(self):
        table = pd.DataFrame({"duration_days": [42.0], "event": [1]})
        mask, label = masked_labels_at_horizon(table, 42)
        self.assertTrue(mask[0])
        self.assertEqual(label[0], 1.0)

    def test_censored_exactly_at_horizon_is_known_alive_not_masked(self):
        table = pd.DataFrame({"duration_days": [42.0], "event": [0]})
        mask, label = masked_labels_at_horizon(table, 42)
        self.assertTrue(mask[0])
        self.assertEqual(label[0], 0.0)

    def test_censored_before_horizon_without_event_is_masked(self):
        table = pd.DataFrame({"duration_days": [41.999], "event": [0]})
        mask, _ = masked_labels_at_horizon(table, 42)
        self.assertFalse(mask[0])

    def test_event_after_horizon_is_known_alive_zero_not_masked(self):
        table = pd.DataFrame({"duration_days": [100.0], "event": [1]})
        mask, label = masked_labels_at_horizon(table, 42)
        self.assertTrue(mask[0])
        self.assertEqual(label[0], 0.0)


class BuildingFoldTests(unittest.TestCase):
    def test_no_building_spans_multiple_folds(self):
        table = pd.DataFrame(
            {
                "building_id": ["b1"] * 5 + ["b2"] * 5 + ["b3"] * 5,
                "device_id": [f"d{i}" for i in range(15)],
            }
        )
        folds = assign_building_folds(table, n_folds=3)
        for building_id, group in table.groupby("building_id"):
            assigned = folds.loc[group.index].unique()
            self.assertEqual(len(assigned), 1)


class TerminalTimeTests(unittest.TestCase):
    def test_observed_eol_used_as_terminal_time(self):
        locations = pd.DataFrame(
            {
                "battery": ["d1", "d2"],
                "building": ["b1", "b1"],
                "start_time": [pd.Timestamp("2024-01-01")] * 2,
                "end_time": [pd.Timestamp("2024-06-01")] * 2,
            }
        )
        eol = pd.Series({"d1": pd.Timestamp("2024-03-01"), "d2": pd.NaT})
        result = terminal_times(locations, eol)
        self.assertEqual(result.loc["d1", "terminal_time"], pd.Timestamp("2024-03-01"))
        self.assertEqual(result.loc["d1", "observed_event"], 1)
        self.assertEqual(result.loc["d2", "terminal_time"], pd.Timestamp("2024-06-01"))
        self.assertEqual(result.loc["d2", "observed_event"], 0)


class FakeAFTModel:
    """Deterministic stand-in for a fitted lifelines AFT model.

    Ignores covariates and returns the same analytic survival curve
    S(t) = exp(-t/100) for every row, so contract math can be checked against
    a hand-computed closed-form integral instead of a real fitted model.
    """

    def predict_survival_function(self, X: pd.DataFrame, times) -> pd.DataFrame:
        times = np.asarray(times, dtype=float)
        values = np.exp(-times / 100.0)
        data = np.tile(values.reshape(-1, 1), (1, len(X)))
        return pd.DataFrame(data, index=times, columns=X.index)


class ContractMathTests(unittest.TestCase):
    def _locations(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "battery": ["d_test"],
                "building": ["b1"],
                "room": ["r1"],
                "start_time": [pd.Timestamp("2023-01-01")],
                "end_time": [pd.Timestamp("2025-01-01")],
            }
        )

    def _forecaster(self) -> Task1Forecaster:
        transform = FeatureTransform(
            columns=("latest_voltage",),
            medians={"latest_voltage": 3.0},
            means={"latest_voltage": 3.0},
            stds={"latest_voltage": 1.0},
        )
        return Task1Forecaster(
            model_family="fake",
            penalizer=0.0,
            aft_model=FakeAFTModel(),
            transform=transform,
            calibrator=PlattCalibrator(slope=1.0, intercept=0.0),
        )

    def test_tail_probabilities_match_hand_computed_exponential_integral(self):
        forecaster = self._forecaster()
        origin = pd.Timestamp("2025-06-01")
        horizon_days = 20
        observation_end = origin + pd.Timedelta(days=60)
        battery_data = pd.DataFrame(columns=["device_id", "end_time", "voltage", "temperature"])

        forecast = forecaster.predict(
            battery_data,
            self._locations(),
            prediction_origin=origin,
            horizon_days=horizon_days,
            evaluation_observation_end=observation_end,
        )

        s20 = np.exp(-20.0 / 100.0)
        s60 = np.exp(-60.0 / 100.0)
        expected_observed_after = s20 - s60
        expected_unobserved = s60
        expected_final_cdf = 1.0 - s20
        integral = 100.0 * (np.exp(-20.0 / 100.0) - np.exp(-60.0 / 100.0))
        expected_mean_excess = (integral - 40.0 * s60) / (s20 - s60)

        tail = forecast.tail.iloc[0]
        self.assertAlmostEqual(tail["prob_observed_after_horizon"], expected_observed_after, places=4)
        self.assertAlmostEqual(tail["prob_unobserved_eol"], expected_unobserved, places=4)
        self.assertAlmostEqual(
            tail["mean_excess_rul_days_given_observed_after_horizon"], expected_mean_excess, places=2
        )

        final_curve_value = forecast.curves.iloc[-1]["failure_cdf"]
        self.assertAlmostEqual(final_curve_value, expected_final_cdf, places=4)

        total = (
            final_curve_value
            + tail["prob_observed_after_horizon"]
            + tail["prob_unobserved_eol"]
        )
        self.assertAlmostEqual(total, 1.0, places=6)

        battery_ids = [str(b) for b in self._locations()["battery"]]
        dates = pd.date_range(origin, origin + pd.Timedelta(days=horizon_days), freq="D")
        validate_forecast(forecast, battery_ids, dates)  # must not raise

    def test_zero_span_beyond_horizon_gives_zero_observed_after_and_mean_excess(self):
        forecaster = self._forecaster()
        origin = pd.Timestamp("2025-06-01")
        horizon_days = 42
        observation_end = origin  # observation ends at origin itself: C <= horizon start
        battery_data = pd.DataFrame(columns=["device_id", "end_time", "voltage", "temperature"])

        forecast = forecaster.predict(
            battery_data,
            self._locations(),
            prediction_origin=origin,
            horizon_days=horizon_days,
            evaluation_observation_end=observation_end,
        )
        tail = forecast.tail.iloc[0]
        self.assertAlmostEqual(tail["prob_observed_after_horizon"], 0.0, places=6)
        self.assertAlmostEqual(tail["mean_excess_rul_days_given_observed_after_horizon"], 0.0, places=6)
        self.assertGreater(tail["prob_unobserved_eol"], 0.0)


class FakeZeroRiskAFTModel:
    """Stand-in AFT model that always predicts zero risk (S(t)=1 for all t).

    Isolates the physical-crossing-day blend: any elevated risk in the
    output must come from the physical term, not the AFT term.
    """

    def predict_survival_function(self, X: pd.DataFrame, times) -> pd.DataFrame:
        times = np.asarray(times, dtype=float)
        data = np.ones((len(times), len(X)))
        return pd.DataFrame(data, index=times, columns=X.index)


class PhysicalPriorBlendTests(unittest.TestCase):
    def _forecaster(self, physical_uncertainty_days: float = 10.0) -> Task1Forecaster:
        columns = tuple(CURATED_FEATURES)
        transform = FeatureTransform(
            columns=columns,
            medians={c: 0.0 for c in columns},
            means={c: 0.0 for c in columns},
            stds={c: 1.0 for c in columns},
        )
        return Task1Forecaster(
            model_family="fake-zero-risk",
            penalizer=0.0,
            aft_model=FakeZeroRiskAFTModel(),
            transform=transform,
            calibrator=PlattCalibrator(slope=1.0, intercept=0.0),
            physical_uncertainty_days=physical_uncertainty_days,
        )

    def _locations(self, battery_id: str, origin: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "battery": [battery_id],
                "building": ["b1"],
                "room": ["r1"],
                "start_time": [origin - pd.Timedelta(days=200)],
                "end_time": [origin + pd.Timedelta(days=100)],
            }
        )

    def test_physical_prior_lifts_risk_when_aft_predicts_zero(self):
        forecaster = self._forecaster()
        origin = pd.Timestamp("2025-06-01")
        n = 40
        # Steep, clean decline crossing the 2.4V EOL threshold around day 20.
        voltage = 3.0 - 0.03 * np.arange(n)
        battery_data = _synthetic_readings("d_decline", origin - pd.Timedelta(days=n), n, voltage)

        forecast = forecaster.predict(
            battery_data,
            self._locations("d_decline", origin),
            prediction_origin=origin,
            horizon_days=42,
            evaluation_observation_end=origin + pd.Timedelta(days=100),
        )
        final_cdf = forecast.curves.iloc[-1]["failure_cdf"]
        self.assertGreater(final_cdf, 0.9)  # AFT alone gives exactly 0 here

    def test_physical_prior_stays_near_zero_for_flat_voltage(self):
        forecaster = self._forecaster()
        origin = pd.Timestamp("2025-06-01")
        n = 40
        voltage = np.full(n, 3.0)  # no decline at all -> huge crossing-day estimate
        battery_data = _synthetic_readings("d_flat", origin - pd.Timedelta(days=n), n, voltage)

        forecast = forecaster.predict(
            battery_data,
            self._locations("d_flat", origin),
            prediction_origin=origin,
            horizon_days=42,
            evaluation_observation_end=origin + pd.Timedelta(days=100),
        )
        final_cdf = forecast.curves.iloc[-1]["failure_cdf"]
        self.assertLess(final_cdf, 0.05)


class EndToEndSyntheticFitTests(unittest.TestCase):
    """Fits a real (tiny) AFT model end-to-end and checks contract compliance."""

    def _build_synthetic_dataset(self):
        rng = np.random.default_rng(0)
        n_days = 220
        start = pd.Timestamp("2024-01-01")
        dates = pd.date_range(start, periods=n_days, freq="D")
        buildings = ["b1", "b2", "b3"]
        rows = []
        loc_rows = []
        eol = {}
        for i in range(12):
            device_id = f"d_{i}"
            building = buildings[i % 3]
            failing = i % 2 == 0
            decline = 0.006 if failing else 0.0002
            noise = rng.normal(0, 0.01, size=n_days)
            voltage = 3.1 - decline * np.arange(n_days) + noise
            temp = 20.0 + rng.normal(0, 1.0, size=n_days)
            rows.append(
                pd.DataFrame(
                    {"device_id": device_id, "end_time": dates, "voltage": voltage, "temperature": temp}
                )
            )
            crossing = np.argmax(voltage < 2.4) if (voltage < 2.4).any() else None
            window_end = start + pd.Timedelta(days=n_days - 1)
            if crossing is not None and crossing < n_days - 1:
                eol[device_id] = dates[crossing]
            else:
                eol[device_id] = pd.NaT
            loc_rows.append(
                {
                    "battery": device_id,
                    "building": building,
                    "room": f"r_{i % 4}",
                    "start_time": start,
                    "end_time": window_end,
                }
            )
        timeseries = pd.concat(rows, ignore_index=True)
        locations = pd.DataFrame(loc_rows)
        eol_times = pd.Series(eol)
        return locations, timeseries, eol_times

    def test_fitted_forecaster_produces_contract_valid_forecast(self):
        locations, timeseries, eol_times = self._build_synthetic_dataset()
        cutoff_dates = build_cutoff_grid(
            [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-01")],
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-07-01"),
            step_days=15,
        )
        from src.risk.features import build_feature_series

        feature_series = build_feature_series(timeseries)
        table = build_example_table(locations, eol_times, feature_series, cutoff_dates)
        self.assertGreater(table["event"].sum(), 0)

        forecaster, report = select_and_fit(table, n_folds=3, families=("weibull",), penalizers=(1.0,))
        self.assertIn("selected_family", report)

        origin = pd.Timestamp("2024-06-01")
        horizon_days = 42
        observation_end = locations["end_time"].max()
        cut = timeseries[timeseries["end_time"] <= origin]

        forecast = forecaster.predict(
            cut,
            locations,
            prediction_origin=origin,
            horizon_days=horizon_days,
            evaluation_observation_end=observation_end,
        )
        battery_ids = [str(b) for b in locations["battery"]]
        dates = pd.date_range(origin, origin + pd.Timedelta(days=horizon_days), freq="D")
        validated = validate_forecast(forecast, battery_ids, dates)
        self.assertEqual(set(validated.curves["battery_id"]), set(battery_ids))

        for battery_id, group in validated.curves.groupby("battery_id"):
            values = group.sort_values("forecast_date")["failure_cdf"].to_numpy()
            self.assertTrue(np.all(np.diff(values) >= -1e-9))


if __name__ == "__main__":
    unittest.main()
