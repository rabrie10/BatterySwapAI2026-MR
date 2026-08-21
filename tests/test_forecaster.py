"""The forecaster has to satisfy the Task 2 contract exactly, and it has to get
the censoring branch right -- that branch is what stops the planner servicing
batteries whose substitute EOL sits in the past.

Both an early and a late scenario are exercised. The late one is the important
case: near the end of the data a device's remaining observation window is
shorter than the planning window, so no EOL can be *recorded* inside the horizon
no matter what the voltage is doing. Getting that wrong is what makes a planner
service the whole fleet in the closing scenarios.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from batteryswap_public.utils import iterate_scenarios, load_dataset
from batteryswap_solution.forecast import validate_forecast

DATASET = Path("data/raw/train") if Path("data/raw/train").exists() else Path("dataset/train")
MODEL = Path("models/v8_ensemble.joblib")

EARLY_SCENARIO = 0
LATE_SCENARIO = 46


@unittest.skipUnless(MODEL.exists(), "trained model not available")
@unittest.skipUnless((DATASET / "scenarios.json").exists(), "train dataset not available")
class ForecasterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import joblib

        from bsai.forecaster import HazardForecaster

        cls.forecaster = HazardForecaster(joblib.load(MODEL))
        locations, timeseries, eol_times, scenarios = load_dataset(DATASET)
        wanted = {EARLY_SCENARIO, LATE_SCENARIO}
        cls.cases: dict[int, dict] = {}
        for index, (scenario, locs, cut, _) in enumerate(
            iterate_scenarios(locations, timeseries, eol_times, scenarios)
        ):
            if index not in wanted:
                continue
            start = pd.Timestamp(scenario["start_time"])
            horizon = int(scenario["settings"].planning_window_days)
            dates = pd.date_range(
                start.normalize(),
                start.normalize() + pd.Timedelta(days=horizon),
                freq="D",
            )
            forecast = cls.forecaster.predict(
                cut,
                locs,
                prediction_origin=start,
                horizon_days=horizon,
                evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
            )
            cls.cases[index] = dict(
                locs=locs, start=start, horizon=horizon, dates=dates, forecast=forecast
            )
            if wanted <= set(cls.cases):
                break

    def _each(self):
        for index, case in sorted(self.cases.items()):
            with self.subTest(scenario=index):
                yield case

    def test_passes_contract_validation(self) -> None:
        for case in self._each():
            ids = case["locs"]["battery"].astype(str).tolist()
            validated = validate_forecast(case["forecast"], ids, case["dates"])
            self.assertEqual(len(validated.curves), len(ids) * len(case["dates"]))

    def test_cdf_is_monotone_and_bounded(self) -> None:
        for case in self._each():
            pivot = case["forecast"].curves.pivot(
                index="battery_id", columns="forecast_date", values="failure_cdf"
            ).to_numpy()
            self.assertTrue(np.isfinite(pivot).all())
            self.assertTrue((pivot >= -1e-9).all() and (pivot <= 1 + 1e-9).all())
            self.assertTrue((np.diff(pivot, axis=1) >= -1e-9).all())

    def test_branches_sum_to_one(self) -> None:
        for case in self._each():
            forecast, dates = case["forecast"], case["dates"]
            tail = forecast.tail.set_index("battery_id")
            final = (
                forecast.curves[forecast.curves["forecast_date"] == dates[-1]]
                .set_index("battery_id")["failure_cdf"]
                .reindex(tail.index)
            )
            total = (
                final
                + tail["prob_observed_after_horizon"]
                + tail["prob_unobserved_eol"]
            )
            np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-6)

    def test_devices_whose_data_already_ended_get_no_mass(self) -> None:
        """A substitute EOL in the past makes servicing catastrophic."""
        checked = 0
        for case in self._each():
            ended = case["locs"][
                pd.to_datetime(case["locs"]["end_time"]).dt.normalize()
                < case["start"].normalize()
            ]["battery"].astype(str)
            if ended.empty:
                continue
            checked += 1
            forecast = case["forecast"]
            tail = forecast.tail.set_index("battery_id").loc[ended]
            np.testing.assert_allclose(
                tail["prob_unobserved_eol"].to_numpy(), 1.0, atol=1e-9
            )
            curves = forecast.curves
            mass = curves[curves["battery_id"].isin(set(ended))]["failure_cdf"]
            np.testing.assert_allclose(mass.to_numpy(), 0.0, atol=1e-9)
        self.assertGreater(checked, 0, "no scenario exercised the ended-device branch")

    def test_short_observation_window_caps_the_recorded_cdf(self) -> None:
        """No EOL can be recorded after a device stops being observed."""
        checked = 0
        for case in self._each():
            forecast, horizon = case["forecast"], case["horizon"]
            tail = forecast.tail.set_index("battery_id")
            summaries = forecast.summaries.set_index("battery_id")
            short = summaries[
                (summaries["remaining_observation_days"] >= 0)
                & (summaries["remaining_observation_days"] < horizon)
            ]
            if short.empty:
                continue
            checked += 1
            self.assertTrue(
                (tail.loc[short.index, "prob_observed_after_horizon"] <= 1e-6).all()
            )
        self.assertGreater(checked, 0, "no scenario exercised the short-window branch")


if __name__ == "__main__":
    unittest.main()
