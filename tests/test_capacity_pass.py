"""Deterministic capacity post-pass: split overloaded days, merge light ones.

The fixtures fabricate plans whose exact replay trips the evaluator's flat
100-per-hit daily/weekly limits, hand the plan to
``CompetitionPlanner._capacity_repair`` with all-zero timing tables (so every
acceptance decision is purely the operational replay delta), and assert on the
exact replay of the result.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from batteryswap_public.evaluate import EvaluationSettings

from batteryswap_solution.costs import CostTables
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import replay_operational_cost


START = pd.Timestamp("2026-03-02")  # a Monday, like the scenario starts
WINDOW_DAYS = 13
DATES = pd.date_range(START, START + pd.Timedelta(days=WINDOW_DAYS), freq="D")
DEFER_DAY = DATES[-1] + pd.Timedelta(days=1)


def settings() -> EvaluationSettings:
    # Defaults: daily limit 24 (strict >), weekly limit 24 (hit at equality),
    # overtime beyond 8 h at factor 2, both limit penalties 100.
    return EvaluationSettings(
        base_location="b_base",
        base_room="r_base",
        planning_window_days=WINDOW_DAYS,
    )


def zero_costs(battery_ids: list[str]) -> CostTables:
    """Timing-free cost tables: acceptance reduces to the operational delta."""
    count = len(battery_ids)
    return CostTables(
        battery_ids=tuple(battery_ids),
        candidate_dates=DATES,
        service_cost=np.zeros((count, len(DATES))),
        defer_cost=np.zeros(count),
        event_pmf=np.zeros((count, len(DATES))),
        horizon_event_probability=np.zeros(count),
    )


def travel_frame(hours: dict[tuple[str, str], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"from": from_building, "to": to_building, "hours": value}
            for (from_building, to_building), value in hours.items()
        ]
    )


def symmetric(*legs: tuple[str, str, float]) -> dict[tuple[str, str], float]:
    hours: dict[tuple[str, str], float] = {}
    for left, right, value in legs:
        hours[(left, right)] = value
        hours[(right, left)] = value
    return hours


def planner() -> CompetitionPlanner:
    return CompetitionPlanner(config=PlannerConfig())


def run_pass(
    plan: pd.DataFrame,
    costs: CostTables,
    locations: pd.DataFrame,
    travel: pd.DataFrame,
) -> pd.DataFrame:
    return planner()._capacity_repair(
        plan, costs, locations, travel, settings(), START, DEFER_DAY
    )


def replay(plan: pd.DataFrame, locations: pd.DataFrame, travel: pd.DataFrame):
    return replay_operational_cost(
        plan, locations, travel, settings(), START, include_details=True
    )


def hit_counts(details: dict) -> tuple[int, int]:
    daily = sum(1 for record in details["_daily_records"] if record["limit_hit"])
    weekly = sum(1 for record in details["_weekly_records"] if record["limit_hit"])
    return daily, weekly


class DailySplitTest(unittest.TestCase):
    """A 25.5 h day chaining a 20 h-roundtrip building with another must split."""

    def setUp(self) -> None:
        self.locations = pd.DataFrame(
            {
                "battery": ["d_far", "d_idle", "d_mid"],
                "building": ["b_far", "b_mid", "b_mid"],
                "room": ["r_f", "r_m", "r_m"],
                "start_time": [START - pd.Timedelta(days=100)] * 3,
                "end_time": [START + pd.Timedelta(days=30)] * 3,
            }
        )
        self.travel = travel_frame(
            symmetric(
                ("b_base", "b_base", 0.03),
                ("b_far", "b_far", 0.03),
                ("b_mid", "b_mid", 0.03),
                ("b_base", "b_far", 10.0),
                ("b_base", "b_mid", 3.0),
                ("b_far", "b_mid", 9.0),
            )
        )
        self.costs = zero_costs(["d_far", "d_idle", "d_mid"])
        overloaded_day = START + pd.Timedelta(days=1)
        self.plan = pd.DataFrame(
            {
                "day": [overloaded_day, overloaded_day, DEFER_DAY],
                "battery": ["d_far", "d_mid", "d_idle"],
            }
        )

    def test_overloaded_day_is_split_and_limits_clear(self) -> None:
        before = replay(self.plan, self.locations, self.travel)
        self.assertEqual(hit_counts(before), (1, 0))

        repaired = run_pass(self.plan, self.costs, self.locations, self.travel)
        after = replay(repaired, self.locations, self.travel)

        self.assertEqual(hit_counts(after), (0, 0))
        # The only fix that clears the daily hit without buying the weekly one
        # is keeping the far building on its day and pushing the mid building
        # into the next week bucket: +-1..3 day targets all re-trip a limit.
        days = repaired.set_index("battery")["day"]
        self.assertEqual(days.loc["d_far"], START + pd.Timedelta(days=1))
        self.assertEqual(days.loc["d_mid"], START + pd.Timedelta(days=8))
        self.assertEqual(days.loc["d_idle"], DEFER_DAY)
        self.assertLess(
            float(after["total_cost"]), float(before["total_cost"]) - 80.0
        )

    def test_pass_is_deterministic_and_preserves_batteries(self) -> None:
        first = run_pass(self.plan, self.costs, self.locations, self.travel)
        second = run_pass(self.plan, self.costs, self.locations, self.travel)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(sorted(first["battery"]), sorted(self.plan["battery"]))
        self.assertEqual(
            int((first["day"] <= DATES[-1]).sum()),
            int((self.plan["day"] <= DATES[-1]).sum()),
        )


class WeeklyBucketTest(unittest.TestCase):
    """Three 9 h days in one week bucket (25.3 h >= 24) shed one day +7."""

    def setUp(self) -> None:
        self.locations = pd.DataFrame(
            {
                "battery": ["d_b1", "d_b2", "d_b3"],
                "building": ["b_1", "b_2", "b_3"],
                "room": ["r_1", "r_2", "r_3"],
                "start_time": [START - pd.Timedelta(days=100)] * 3,
                "end_time": [START + pd.Timedelta(days=30)] * 3,
            }
        )
        legs = [("b_base", building, 4.0) for building in ("b_1", "b_2", "b_3")]
        legs += [
            ("b_1", "b_2", 12.0),
            ("b_1", "b_3", 12.0),
            ("b_2", "b_3", 12.0),
        ]
        legs += [
            (building, building, 0.03)
            for building in ("b_base", "b_1", "b_2", "b_3")
        ]
        self.travel = travel_frame(symmetric(*legs))
        self.costs = zero_costs(["d_b1", "d_b2", "d_b3"])
        self.plan = pd.DataFrame(
            {
                "day": [
                    START + pd.Timedelta(days=1),
                    START + pd.Timedelta(days=2),
                    START + pd.Timedelta(days=3),
                ],
                "battery": ["d_b1", "d_b2", "d_b3"],
            }
        )

    def test_weekly_hit_clears_by_moving_first_group_a_week_out(self) -> None:
        before = replay(self.plan, self.locations, self.travel)
        self.assertEqual(hit_counts(before), (0, 1))

        repaired = run_pass(self.plan, self.costs, self.locations, self.travel)
        after = replay(repaired, self.locations, self.travel)

        self.assertEqual(hit_counts(after), (0, 0))
        # All three +7 shifts clear the bucket at identical cost; the
        # (delta, target day, batteries) tie-break picks the earliest target.
        days = repaired.set_index("battery")["day"]
        self.assertEqual(days.loc["d_b1"], START + pd.Timedelta(days=8))
        self.assertEqual(days.loc["d_b2"], START + pd.Timedelta(days=2))
        self.assertEqual(days.loc["d_b3"], START + pd.Timedelta(days=3))
        self.assertLess(
            float(after["total_cost"]), float(before["total_cost"]) - 90.0
        )


class LightDayMergeTest(unittest.TestCase):
    """Two 3 h days in the same building merge and save a return trip."""

    def setUp(self) -> None:
        self.locations = pd.DataFrame(
            {
                "battery": ["d_n1", "d_n2"],
                "building": ["b_near", "b_near"],
                "room": ["r_n", "r_n"],
                "start_time": [START - pd.Timedelta(days=100)] * 2,
                "end_time": [START + pd.Timedelta(days=30)] * 2,
            }
        )
        self.travel = travel_frame(
            symmetric(
                ("b_base", "b_base", 0.03),
                ("b_near", "b_near", 0.03),
                ("b_base", "b_near", 1.0),
            )
        )
        self.costs = zero_costs(["d_n1", "d_n2"])
        self.plan = pd.DataFrame(
            {
                "day": [START + pd.Timedelta(days=1), START + pd.Timedelta(days=2)],
                "battery": ["d_n1", "d_n2"],
            }
        )

    def test_adjacent_light_days_merge_onto_the_earlier_day(self) -> None:
        before = replay(self.plan, self.locations, self.travel)
        self.assertEqual(hit_counts(before), (0, 0))

        repaired = run_pass(self.plan, self.costs, self.locations, self.travel)
        after = replay(repaired, self.locations, self.travel)

        days = repaired.set_index("battery")["day"]
        self.assertEqual(days.loc["d_n1"], START + pd.Timedelta(days=1))
        self.assertEqual(days.loc["d_n2"], START + pd.Timedelta(days=1))
        self.assertEqual(hit_counts(after), (0, 0))
        self.assertLess(float(after["total_cost"]), float(before["total_cost"]))


if __name__ == "__main__":
    unittest.main()
