"""The margin target must reproduce the EOL label exactly.

The whole V7 argument is that regressing the running minimum of the voltage
margin is an *exact restatement* of "an EOL record exists within h days", not an
approximation. If that equality ever breaks, the extra supervision is bought
with a biased target and the model is worse than the classifier it replaces --
so this is checked against the real dataset, not a fixture.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bsai.margin import (
    EOL_THRESHOLD,
    NO_CROSSING_MARGIN,
    TRAIN_HORIZONS,
    build_margin_targets,
    forward_running_minimum,
)

DATASET = Path("dataset/train")


class ForwardRunningMinimumTest(unittest.TestCase):
    def test_window_is_strictly_after_the_cutoff(self) -> None:
        margin = np.array([0.5, -1.0, 0.4, 0.3, 0.2])
        out = forward_running_minimum(margin, horizon=2, observation_end=4)
        # From index 0 the window is days 1..2, which contains the -1.0.
        self.assertAlmostEqual(out[0], -1.0)
        # From index 1 the window is days 2..3; the -1.0 is in the past.
        self.assertAlmostEqual(out[1], 0.3)

    def test_horizon_truncates_at_the_observation_end(self) -> None:
        margin = np.array([0.5, 0.4, 0.3, -0.2, -0.5])
        # A record could only be filed through day 2, so the later dip is
        # invisible however long the horizon is.
        out = forward_running_minimum(margin, horizon=99, observation_end=2)
        self.assertAlmostEqual(out[0], 0.3)
        self.assertGreater(out[0], 0.0)

    def test_no_observable_future_means_no_record(self) -> None:
        margin = np.array([0.5, 0.4, -0.3])
        out = forward_running_minimum(margin, horizon=99, observation_end=0)
        self.assertEqual(out[0], NO_CROSSING_MARGIN)

    def test_nan_days_cannot_trigger_a_crossing(self) -> None:
        """The official detector cannot fire on a day with no smoothed value."""
        margin = np.array([0.5, np.nan, np.nan, 0.2])
        out = forward_running_minimum(margin, horizon=2, observation_end=3)
        self.assertAlmostEqual(out[0], NO_CROSSING_MARGIN)
        self.assertFalse(np.isnan(out).any())

    def test_longer_horizon_never_raises_the_minimum(self) -> None:
        rng = np.random.default_rng(20260821)
        margin = np.cumsum(rng.normal(-0.002, 0.02, size=200)) + 0.4
        short = forward_running_minimum(margin, 10, observation_end=199)
        long = forward_running_minimum(margin, 40, observation_end=199)
        self.assertTrue((long <= short + 1e-12).all())


@unittest.skipUnless((DATASET / "eol_times.csv").exists(), "train dataset not available")
class MarginTargetMatchesLabelTest(unittest.TestCase):
    """y < 0 must equal the recorded-EOL label on every row of the real data."""

    @classmethod
    def setUpClass(cls) -> None:
        from batteryswap_public.utils import load_devices

        from bsai.hazard import build_training_frame
        from bsai.smoothing import SmoothingCache

        epoch = pd.Timestamp("1970-01-01")

        def ordinal(value) -> int:
            return int((pd.Timestamp(value).normalize() - epoch) / pd.Timedelta(days=1))

        devices = load_devices(DATASET / "devices.csv")
        building_of = dict(zip(devices["device_id"], devices["building_id"]))
        eol = pd.to_datetime(
            pd.read_csv(DATASET / "eol_times.csv").set_index("device_id")["end_time"]
        )
        observation_end = devices.set_index("device_id")["end_time"]

        raw = pd.read_parquet(
            DATASET / "battery_metrics.parquet", engine="fastparquet"
        )
        cache = SmoothingCache()
        cache.update(raw)
        del raw

        eol_index: dict[str, int | None] = {}
        observation_index: dict[str, int] = {}
        for device_id, series in cache.devices.items():
            moment = eol.get(device_id)
            eol_index[device_id] = (
                None if pd.isna(moment) else ordinal(moment) - series.origin
            )
            end = observation_end.get(device_id)
            observation_index[device_id] = (
                (series.origin + len(series) - 1)
                if pd.isna(end)
                else ordinal(end) - series.origin
            )

        # A coarse stride keeps the test near a minute while still covering
        # every device and both label classes.
        cls.frame = build_training_frame(
            cache, eol_index, building_of, observation_index, stride=29
        )
        cls.y = build_margin_targets(cls.frame, cache, TRAIN_HORIZONS)[1]

    def test_sign_of_the_target_is_the_label(self) -> None:
        frame, horizons = self.frame, TRAIN_HORIZONS
        count = len(frame)
        horizon = np.repeat(np.asarray(horizons), count)
        cutoff = np.tile(frame.cutoff, len(horizons))
        crossing = np.tile(frame.crossing, len(horizons))
        observation_end = np.tile(frame.observation_end, len(horizons))

        label = (
            (crossing >= 0)
            & (crossing > cutoff)
            & ((crossing - cutoff) <= horizon)
            & (crossing <= observation_end)
        )
        np.testing.assert_array_equal(self.y < 0, label)
        self.assertGreater(int(label.sum()), 0, "no positive rows to check against")
        self.assertGreater(int((~label).sum()), 0, "no negative rows to check against")

    def test_surviving_rows_carry_a_usable_margin(self) -> None:
        """The point of the target: negatives are not all the same negative."""
        alive = self.y[(self.y >= 0) & (self.y < NO_CROSSING_MARGIN)]
        self.assertGreater(alive.size, 0)
        self.assertGreater(float(np.percentile(alive, 90)), 0.1)
        self.assertGreater(int((alive < 0.02).sum()), 0)

    def test_margin_is_measured_against_the_official_threshold(self) -> None:
        self.assertEqual(EOL_THRESHOLD, 2.4)


if __name__ == "__main__":
    unittest.main()
