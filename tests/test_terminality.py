"""Trajectory signals: causality, the counting, and the band restriction.

Three things can silently invalidate the matched near-threshold study: a feature
that reads past its own cutoff, a new-low count that always fires on the first
day of its window, and a deployment that spends the signal outside the
conditioning it was measured under. All three are pinned here.
"""

from __future__ import annotations

import unittest

import numpy as np

from bsai.hazard import HORIZON_GRID
from bsai.rerank import DECISION_HORIZON, remap
from bsai.runtime import HARD_DEADLINE_SECONDS, SOFT_DEADLINE_SECONDS
from bsai.terminality import NAMES, NearThresholdScorer, features_at, std_ratio


def _series(n: int = 600, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    voltage = 2.7 - np.linspace(0.0, 0.25, n) + 0.004 * rng.standard_normal(n)
    temperature = 20.0 + 5.0 * np.sin(np.arange(n) / 365.0 * 2 * np.pi)
    return voltage, temperature


class CausalityTest(unittest.TestCase):
    def test_nothing_after_the_cutoff_is_read(self) -> None:
        voltage, temperature = _series()
        index = 400
        before = features_at(voltage, temperature, index)
        # Replace the entire future with nonsense; the answer must not move.
        tampered = voltage.copy()
        tampered[index + 1 :] = 0.0
        tampered_temp = temperature.copy()
        tampered_temp[index + 1 :] = -50.0
        after = features_at(tampered, tampered_temp, index)
        for name, a, b in zip(NAMES, before, after):
            if np.isnan(a) and np.isnan(b):
                continue
            self.assertAlmostEqual(a, b, places=12, msg=f"{name} read the future")

    def test_too_short_a_history_returns_all_nan(self) -> None:
        voltage, temperature = _series(n=30)
        self.assertTrue(all(np.isnan(v) for v in features_at(voltage, temperature, 5)))


class CountingTest(unittest.TestCase):
    def test_a_flat_series_sets_no_new_lows(self) -> None:
        flat = np.full(400, 2.5)
        temperature = np.full(400, 20.0)
        values = dict(zip(NAMES, features_at(flat, temperature, 399)))
        self.assertEqual(values["new_lows_90"], 0.0)

    def test_a_falling_series_sets_one_new_low_per_day(self) -> None:
        falling = 2.7 - np.linspace(0.0, 0.3, 400)
        temperature = np.full(400, 20.0)
        values = dict(zip(NAMES, features_at(falling, temperature, 399)))
        self.assertGreaterEqual(values["new_lows_90"], 85.0)
        self.assertEqual(values["floor_gap"], 0.0)
        self.assertEqual(values["days_since_new_low"], 0.0)

    def test_std_ratio_is_one_for_a_stationary_series(self) -> None:
        rng = np.random.default_rng(5)
        series = 2.5 + 0.01 * rng.standard_normal(400)
        self.assertAlmostEqual(std_ratio(series, 399), 1.0, delta=0.35)

    def test_std_ratio_rises_when_the_recent_window_gets_noisy(self) -> None:
        rng = np.random.default_rng(6)
        series = 2.5 + 0.002 * rng.standard_normal(400)
        series[-30:] += 0.03 * rng.standard_normal(30)
        self.assertGreater(std_ratio(series, 399), 2.0)


class DeploymentTest(unittest.TestCase):
    def setUp(self) -> None:
        from bsai.features import FEATURE_NAMES

        self.column = list(HORIZON_GRID).index(DECISION_HORIZON)
        self.voltage_index = FEATURE_NAMES.index("voltage")
        rng = np.random.default_rng(9)
        self.rows = 40
        self.grid = np.maximum.accumulate(
            np.sort(rng.random((self.rows, len(HORIZON_GRID))), axis=1), axis=1
        )
        self.remaining = np.full(self.rows, 300.0)
        self.features = np.zeros((self.rows, len(FEATURE_NAMES)), dtype=np.float32)
        # Half the rows inside the 0-0.10 V band, half well outside it.
        margins = np.where(np.arange(self.rows) < 20, 0.05, 0.40)
        self.features[:, self.voltage_index] = 2.4 + margins
        self.devices = np.asarray([f"d{i}" for i in range(self.rows)])
        voltage, _ = _series()
        self.scorer = NearThresholdScorer(
            series={d: (voltage, 0) for d in self.devices},
            end_ordinal={d: 800.0 for d in self.devices},
        )

    def test_rows_outside_the_band_keep_the_incumbent_order(self) -> None:
        score = self.scorer.score(
            self.features, self.remaining, self.devices, self.grid
        )
        outside = np.arange(20, self.rows)
        level = self.grid[:, self.column]
        self.assertTrue(
            np.array_equal(
                np.argsort(-score[outside]), np.argsort(-level[outside])
            )
        )

    def test_the_permutation_preserves_the_probability_multiset(self) -> None:
        score = self.scorer.score(
            self.features, self.remaining, self.devices, self.grid
        )
        moved = remap(self.grid, self.remaining, score, self.column)
        self.assertTrue(
            np.allclose(
                np.sort(moved[:, self.column]), np.sort(self.grid[:, self.column])
            )
        )
        self.assertTrue(
            np.isclose(moved[:, self.column].sum(), self.grid[:, self.column].sum())
        )

    def test_zero_weight_is_the_incumbent_exactly(self) -> None:
        self.scorer.weight = 0.0
        score = self.scorer.score(
            self.features, self.remaining, self.devices, self.grid
        )
        moved = remap(self.grid, self.remaining, score, self.column)
        self.assertTrue(np.array_equal(moved, self.grid))

    def test_an_unknown_device_falls_back_to_the_incumbent(self) -> None:
        scorer = NearThresholdScorer(series={}, end_ordinal={})
        score = scorer.score(self.features, self.remaining, self.devices, self.grid)
        moved = remap(self.grid, self.remaining, score, self.column)
        self.assertTrue(np.array_equal(moved, self.grid))


class GovernorTest(unittest.TestCase):
    """The deadlines have to bracket the *measured* run, not an old one.

    The shipped configuration plans 48 train scenarios in 673 s through
    ``script.py``, so 96 project to about 22.5 minutes. A soft deadline below
    that degrades a healthy submission; a hard deadline at or past 30 minutes
    fails to protect the cap. See docs/FINAL_J2W_RESULTS.md section 12.
    """

    PROJECTED_SECONDS = 2 * 673 + 10
    CAP_SECONDS = 30 * 60

    def test_soft_deadline_is_above_the_measured_projection(self) -> None:
        self.assertGreater(SOFT_DEADLINE_SECONDS, self.PROJECTED_SECONDS)

    def test_deadlines_are_ordered_and_inside_the_cap(self) -> None:
        self.assertLess(SOFT_DEADLINE_SECONDS, HARD_DEADLINE_SECONDS)
        self.assertLess(HARD_DEADLINE_SECONDS, self.CAP_SECONDS)

    def test_the_hard_deadline_leaves_room_for_the_all_defer_tail(self) -> None:
        self.assertGreaterEqual(self.CAP_SECONDS - HARD_DEADLINE_SECONDS, 120)


if __name__ == "__main__":
    unittest.main()
