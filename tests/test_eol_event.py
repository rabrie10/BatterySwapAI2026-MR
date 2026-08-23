"""The official EOL rule, pinned as a property of the shipped smoothing.

`docs/FINAL_EOL_EVENT.md` reconstructs it from the data: EOL is the first day the
smoothed series is below 2.4 V, with no persistence condition beyond the
seven-day rolling median that produces the series. V8's state is that same
smoothed margin, so V8 already models the official event -- and the reflection
term in the passage law is *correct* rather than over-counting, because a
dip-and-recover is still an EOL.

These tests pin the two properties the conclusion rests on, at the level of the
smoothing itself, so a future change to `bsai/smoothing.py` cannot quietly
invalidate the reconstruction.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bsai.smoothing import SmoothingCache

EOL_THRESHOLD = 2.4


def _frame(voltage: np.ndarray, readings: int = 6) -> pd.DataFrame:
    """One device, ``readings`` hourly points per day inside the temperature gate."""
    start = pd.Timestamp("2024-01-01")
    rows = []
    for day, value in enumerate(voltage):
        for hour in range(readings):
            rows.append(
                {
                    "device_id": "d",
                    "end_time": start + pd.Timedelta(days=day, hours=hour),
                    "voltage": float(value),
                    "temperature": 20.0,
                }
            )
    return pd.DataFrame(rows)


class SmoothingPersistenceTest(unittest.TestCase):
    def test_a_single_low_day_does_not_move_the_median_below(self) -> None:
        """One day under the line is not an EOL -- the median absorbs it.

        This is the persistence the rule does have, and it lives inside the
        series rather than as a condition on top of it.
        """
        voltage = np.full(30, 2.50)
        voltage[20] = 2.20
        cache = SmoothingCache()
        cache.update(_frame(voltage))
        smoothed = cache.devices["d"].smooth_voltage
        self.assertFalse((smoothed < EOL_THRESHOLD).any())

    def test_a_sustained_run_does_move_it_below(self) -> None:
        voltage = np.full(30, 2.50)
        voltage[20:] = 2.20
        cache = SmoothingCache()
        cache.update(_frame(voltage))
        smoothed = cache.devices["d"].smooth_voltage
        self.assertTrue((smoothed < EOL_THRESHOLD).any())

    def test_the_series_can_dip_below_and_recover(self) -> None:
        """Which is why the reflection term is right, not over-counting.

        47 of the 81 devices that ever cross do exactly this, and every one is
        recorded as EOL on the day of its *first* touch.
        """
        voltage = np.full(60, 2.50)
        voltage[20:32] = 2.20
        cache = SmoothingCache()
        cache.update(_frame(voltage))
        smoothed = cache.devices["d"].smooth_voltage
        below = np.flatnonzero(smoothed < EOL_THRESHOLD)
        self.assertGreater(below.size, 0)
        after = smoothed[below[-1] + 1 :]
        after = after[~np.isnan(after)]
        self.assertTrue((after >= EOL_THRESHOLD).any())


if __name__ == "__main__":
    unittest.main()
