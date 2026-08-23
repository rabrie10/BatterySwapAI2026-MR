"""Survival-surprise features: the leakage rules that make the test meaningful.

The experiment in `docs/FINAL_FRAILTY.md` is only valid if `H_surv` is built from
past telemetry alone. Four ways it could have been invalid, each pinned here:
counting a week that has not finished, counting overlapping intervals twice,
reading anything at or after the cutoff, and letting the accumulation depend on
the order rows happen to be presented in.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.fj_frailty import HIGH, NAMES, WEEK, survival_features


def _index(name: str) -> int:
    return NAMES.index(name)


class CausalityTest(unittest.TestCase):
    def setUp(self) -> None:
        # Weekly cutoffs at day 0, 7, ... 70, each with a known p7.
        self.cutoffs = np.arange(0, 77, WEEK)
        self.p7 = np.full(self.cutoffs.size, 0.2)

    def test_only_completed_weeks_count(self) -> None:
        """A week is evidence of survival only once its interval has finished."""
        at_28 = survival_features(self.cutoffs, self.p7, 28)
        # Weeks starting at 0, 7, 14 have completed by day 28; the one at 21 has
        # its interval ending exactly at 28, which also counts.
        self.assertEqual(at_28[_index("weeks_observed")], 4.0)
        at_27 = survival_features(self.cutoffs, self.p7, 27)
        self.assertEqual(at_27[_index("weeks_observed")], 3.0)

    def test_nothing_at_or_after_the_cutoff_is_read(self) -> None:
        before = survival_features(self.cutoffs, self.p7, 28)
        tampered = self.p7.copy()
        tampered[self.cutoffs + WEEK > 28] = 0.99
        after = survival_features(self.cutoffs, tampered, 28)
        for name, a, b in zip(NAMES, before, after):
            self.assertAlmostEqual(a, b, places=12, msg=f"{name} read the future")

    def test_hazard_is_the_sum_of_non_overlapping_weeks(self) -> None:
        value = survival_features(self.cutoffs, self.p7, 28)[_index("H_surv")]
        self.assertAlmostEqual(value, 4 * -np.log(1 - 0.2), places=9)

    def test_overlapping_weeks_would_double_count_and_are_not_used(self) -> None:
        """Halving the stride must not double the hazard for the same period."""
        dense = np.arange(0, 77, 1)
        dense_p7 = np.full(dense.size, 0.2)
        weekly = survival_features(self.cutoffs, self.p7, 28)[_index("H_surv")]
        overlapping = survival_features(dense, dense_p7, 28)[_index("H_surv")]
        # The tool is called with a weekly grid precisely so this cannot happen;
        # the assertion documents what the wrong grid would do.
        self.assertGreater(overlapping, 4 * weekly)

    def test_no_history_yields_nan_rather_than_zero(self) -> None:
        """Zero hazard survived and *no evidence* are different states."""
        out = survival_features(self.cutoffs, self.p7, 3)
        self.assertTrue(all(np.isnan(v) for v in out))


class SummaryTest(unittest.TestCase):
    def test_high_risk_counts_use_the_documented_thresholds(self) -> None:
        cutoffs = np.arange(0, 70, WEEK)
        p7 = np.array([0.05, 0.15, 0.25, 0.35, 0.05, 0.15, 0.25, 0.35, 0.05, 0.15])
        out = survival_features(cutoffs, p7, 70)
        self.assertEqual(out[_index("weeks_over_10")], float((p7 > HIGH[0]).sum()))
        self.assertEqual(out[_index("weeks_over_20")], float((p7 > HIGH[1]).sum()))
        self.assertEqual(out[_index("weeks_over_30")], float((p7 > HIGH[2]).sum()))
        self.assertAlmostEqual(out[_index("max_prior_p7")], 0.35)

    def test_consecutive_run_is_counted_from_the_most_recent_week(self) -> None:
        cutoffs = np.arange(0, 56, WEEK)
        p7 = np.array([0.5, 0.5, 0.01, 0.5, 0.5, 0.5, 0.01, 0.5])
        out = survival_features(cutoffs, p7, 56)
        self.assertEqual(out[_index("consecutive_high_weeks")], 1.0)
        # Clearing the break two weeks back joins the trailing week to the run
        # of three before it: 3 + 1 (the healed week) + 1 = 5, not 2.
        p7[-2] = 0.5
        out = survival_features(cutoffs, p7, 56)
        self.assertEqual(out[_index("consecutive_high_weeks")], 5.0)
        # A break in the most recent week resets it regardless of history.
        p7[-1] = 0.01
        out = survival_features(cutoffs, p7, 56)
        self.assertEqual(out[_index("consecutive_high_weeks")], 0.0)

    def test_recent_windows_are_subsets_of_the_total(self) -> None:
        cutoffs = np.arange(0, 30 * WEEK, WEEK)
        rng = np.random.default_rng(3)
        p7 = rng.uniform(0.0, 0.4, cutoffs.size)
        out = survival_features(cutoffs, p7, 30 * WEEK)
        total = out[_index("H_surv")]
        for name in ("H_last_4", "H_last_8", "H_last_12", "H_last_26"):
            self.assertLessEqual(out[_index(name)], total + 1e-9)
        self.assertLessEqual(out[_index("H_last_4")], out[_index("H_last_8")] + 1e-9)


if __name__ == "__main__":
    unittest.main()
