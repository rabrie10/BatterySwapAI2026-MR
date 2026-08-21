"""Candidate reduction is a speedup, not a decision.

Every search evaluation is linear in the battery count, and about nine of some
four hundred batteries are ever due inside a window. Restricting the search to
batteries servicing could plausibly help is what makes the evaluation budget
fit -- but it must never drop a battery the optimizer would have serviced, and
the plan it produces must still name every battery exactly once.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from batteryswap_solution.costs import CostTables, select_candidates
from batteryswap_solution.planner import CompetitionPlanner


def _tables(service: np.ndarray, defer: np.ndarray) -> CostTables:
    ids = tuple(f"d_{i}" for i in range(service.shape[0]))
    dates = pd.date_range("2026-01-05", periods=service.shape[1], freq="D")
    return CostTables(
        battery_ids=ids,
        candidate_dates=dates,
        service_cost=service,
        defer_cost=defer,
        event_pmf=np.zeros_like(service),
        horizon_event_probability=np.zeros(service.shape[0]),
    )


class SelectCandidatesTest(unittest.TestCase):
    def test_keeps_batteries_where_servicing_can_pay(self) -> None:
        service = np.array([[5.0, 4.0], [200.0, 190.0], [1.0, 2.0]])
        defer = np.array([300.0, 0.0, 0.5])
        keep = select_candidates(_tables(service, defer), margin_hours=24.0)
        self.assertIn(0, keep)  # deferring is catastrophic
        self.assertIn(2, keep)  # nearly free to service
        self.assertNotIn(1, keep)  # 190 hours of earliness against no risk

    def test_margin_is_generous_enough_to_be_a_speedup(self) -> None:
        """A battery only ten hours from breaking even stays in the search."""
        service = np.array([[10.0]])
        defer = np.array([0.0])
        keep = select_candidates(_tables(service, defer), margin_hours=24.0)
        self.assertEqual(list(keep), [0])

    def test_cap_keeps_the_largest_gains_and_stays_sorted(self) -> None:
        count = 40
        service = np.zeros((count, 1))
        defer = np.arange(count, dtype=float)
        keep = select_candidates(
            _tables(service, defer), margin_hours=1e9, max_candidates=5
        )
        self.assertEqual(len(keep), 5)
        self.assertEqual(list(keep), sorted(keep))
        self.assertEqual(set(keep), set(range(count - 5, count)))

    def test_take_slices_every_array_consistently(self) -> None:
        service = np.arange(12, dtype=float).reshape(4, 3)
        defer = np.arange(4, dtype=float)
        tables = _tables(service, defer)
        subset = tables.take(np.array([1, 3]))
        self.assertEqual(subset.battery_ids, ("d_1", "d_3"))
        np.testing.assert_array_equal(subset.service_cost, service[[1, 3]])
        np.testing.assert_array_equal(subset.defer_cost, defer[[1, 3]])
        self.assertIs(subset.candidate_dates, tables.candidate_dates)


class RestoreExcludedTest(unittest.TestCase):
    def test_excluded_batteries_come_back_deferred(self) -> None:
        plan = pd.DataFrame(
            {
                "day": pd.DatetimeIndex(["2026-01-05", "2026-01-07"]),
                "battery": ["d_1", "d_2"],
            }
        )
        defer_day = pd.Timestamp("2026-02-17")
        restored = CompetitionPlanner._restore_excluded(plan, ["d_9", "d_3"], defer_day)
        self.assertEqual(len(restored), 4)
        self.assertEqual(sorted(restored["battery"]), ["d_1", "d_2", "d_3", "d_9"])
        deferred = restored[restored["battery"].isin(["d_3", "d_9"])]
        self.assertTrue((deferred["day"] == defer_day).all())

    def test_days_stay_non_decreasing(self) -> None:
        """check_plan_valid rejects a plan whose day column ever goes backwards."""
        plan = pd.DataFrame(
            {
                "day": pd.DatetimeIndex(["2026-01-05", "2026-01-07"]),
                "battery": ["d_1", "d_2"],
            }
        )
        restored = CompetitionPlanner._restore_excluded(
            plan, ["d_3"], pd.Timestamp("2026-02-17")
        )
        self.assertTrue((restored["day"].diff().dropna() >= pd.Timedelta(0)).all())
        self.assertTrue(restored.index.equals(pd.RangeIndex(len(restored))))

    def test_no_excluded_batteries_is_a_no_op(self) -> None:
        plan = pd.DataFrame(
            {"day": pd.DatetimeIndex(["2026-01-05"]), "battery": ["d_1"]}
        )
        self.assertIs(
            CompetitionPlanner._restore_excluded(plan, [], pd.Timestamp("2026-02-17")),
            plan,
        )


if __name__ == "__main__":
    unittest.main()
