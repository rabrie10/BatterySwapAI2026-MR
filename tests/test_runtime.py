"""The governor exists so a slow split cannot cost us the whole run.

A late scenario planned badly costs a few hundred; a run that overruns the
30-minute wall clock scores nothing at all.
"""

from __future__ import annotations

import unittest

import pandas as pd

from bsai.runtime import BudgetedPlanner


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _RecordingPlanner:
    def __init__(self) -> None:
        self.config = "full"
        self.calls: list[str] = []

    def plan(self, battery_data, locations, travel_costs, settings):
        self.calls.append(self.config)
        return pd.DataFrame(
            {
                "day": pd.DatetimeIndex([pd.Timestamp("2026-01-05")] * len(locations)),
                "battery": locations["battery"].astype(str).tolist(),
            }
        )


def _locations(count: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"battery": [f"d_{i}" for i in range(count)]})


class BudgetedPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _FakeClock()
        self.inner = _RecordingPlanner()
        self.planner = BudgetedPlanner(
            self.inner,
            fast_config="fast",
            soft_deadline=100.0,
            hard_deadline=200.0,
            clock=self.clock,
        )

    def _plan(self):
        return self.planner.plan(None, _locations(), None, None)

    def test_runs_at_full_quality_before_the_soft_deadline(self) -> None:
        self.clock.now = 50.0
        self._plan()
        self.assertEqual(self.inner.calls, ["full"])
        self.assertEqual(self.planner.scenarios_degraded, 0)

    def test_degrades_once_past_the_soft_deadline(self) -> None:
        self.clock.now = 150.0
        self._plan()
        self.clock.now = 160.0
        self._plan()
        self.assertEqual(self.inner.calls, ["fast", "fast"])
        self.assertEqual(self.planner.scenarios_degraded, 2)

    def test_defers_wholesale_past_the_hard_deadline(self) -> None:
        self.clock.now = 250.0
        plan = self._plan()
        self.assertEqual(self.inner.calls, [])
        self.assertEqual(self.planner.scenarios_deferred, 1)
        self.assertEqual(len(plan), 3)
        self.assertEqual(sorted(plan["battery"]), ["d_0", "d_1", "d_2"])

    def test_deferred_plan_is_outside_any_planning_window(self) -> None:
        self.clock.now = 250.0
        plan = self._plan()
        self.assertGreater(
            plan["day"].min(), pd.Timestamp.now() + pd.Timedelta(days=365 * 10)
        )


if __name__ == "__main__":
    unittest.main()
