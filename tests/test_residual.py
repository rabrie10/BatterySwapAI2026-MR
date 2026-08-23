"""The residual ranking model: labels, weights, capacity and the invariant.

The three things that can silently invalidate this experiment are a censored row
labelled as a survivor, a service-value weight that does not match the official
cost model, and a residual that changes the probability *level* rather than the
order. All three are pinned here.
"""

from __future__ import annotations

import unittest

import numpy as np

from bsai.hazard import HORIZON_GRID
from bsai.rerank import DECISION_HORIZON, remap
from bsai.residual import (
    EARLY_RATE,
    EMERGENCY_DAY,
    LATE_RATE,
    SIGNALS,
    SWAP_DAY,
    OofResidualScorer,
    ResidualScorer,
    build_pairs,
    fit_pairwise,
    fit_pointwise,
    landmark_mask,
    service_value,
)


class LabelTest(unittest.TestCase):
    def test_censored_before_the_horizon_is_excluded(self) -> None:
        """No record, and the window closes first: the fate is unknown."""
        days = np.array([np.inf, np.inf, np.inf, 20.0, 100.0])
        remaining = np.array([10.0, 41.9, 42.0, 5.0, 300.0])
        due = np.array([False, False, False, True, False])
        keep = landmark_mask(days, remaining, due)
        # rows 0 and 1 close before 42 days with no record -> unknown
        self.assertFalse(keep[0])
        self.assertFalse(keep[1])
        # row 2 has exactly 42 days left and no record -> reliable negative
        self.assertTrue(keep[2])
        # a positive is always kept, whatever its window
        self.assertTrue(keep[3])
        # a record after the horizon is a reliable negative
        self.assertTrue(keep[4])

    def test_a_positive_is_never_dropped_by_its_window(self) -> None:
        keep = landmark_mask(
            np.array([5.0]), np.array([1.0]), np.array([True])
        )
        self.assertTrue(keep[0])


class ServiceValueTest(unittest.TestCase):
    def test_matches_the_official_cost_arithmetic(self) -> None:
        for effective in (2.0, 20.0, 41.0, 100.0, 300.0):
            served = EARLY_RATE * max(effective - SWAP_DAY, 0.0) + LATE_RATE * max(
                SWAP_DAY - effective, 0.0
            )
            deferred = LATE_RATE * max(EMERGENCY_DAY - effective, 0.0)
            self.assertAlmostEqual(
                float(service_value(np.array([effective]))[0]), deferred - served
            )

    def test_servicing_pays_inside_the_window_and_not_outside(self) -> None:
        inside = service_value(np.array([5.0, 20.0, 41.0]))
        outside = service_value(np.array([100.0, 300.0]))
        self.assertTrue((inside > 0).all())
        self.assertTrue((outside < 0).all())
        # A death on day 5 is worth more than one on day 41.
        self.assertGreater(inside[0], inside[2])
        # A far substitute end of life is the expensive false positive.
        self.assertLess(outside[1], outside[0])


class ObjectiveTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.n = 400
        self.matrix = rng.normal(size=(self.n, 3))
        self.anchor = rng.normal(size=self.n)
        # A signal the anchor does not carry, so a fit has something to find.
        self.label = (self.matrix[:, 0] + 0.2 * rng.normal(size=self.n) > 0).astype(float)
        self.weight = np.ones(self.n)

    def test_pointwise_recovers_the_direction(self) -> None:
        w = fit_pointwise(
            self.matrix, self.anchor, self.label, self.weight, l2=1e-4
        )
        self.assertGreater(w[0], 0.5)
        self.assertLess(abs(w[1]), abs(w[0]))

    def test_focal_and_plain_agree_on_the_sign(self) -> None:
        plain = fit_pointwise(self.matrix, self.anchor, self.label, self.weight, l2=1e-3)
        focal = fit_pointwise(
            self.matrix, self.anchor, self.label, self.weight, l2=1e-3, focal_gamma=2.0
        )
        self.assertGreater(plain[0], 0.0)
        self.assertGreater(focal[0], 0.0)

    def test_regularisation_shrinks_the_residual(self) -> None:
        loose = fit_pointwise(self.matrix, self.anchor, self.label, self.weight, l2=1e-4)
        tight = fit_pointwise(self.matrix, self.anchor, self.label, self.weight, l2=10.0)
        self.assertLess(np.abs(tight).sum(), np.abs(loose).sum())

    def test_pairwise_recovers_the_direction(self) -> None:
        positives = np.flatnonzero(self.label > 0.5)[:80]
        negatives = np.flatnonzero(self.label < 0.5)[:80]
        w = fit_pairwise(
            self.matrix, self.anchor, positives, negatives,
            np.ones(positives.size), l2=1e-4,
        )
        self.assertGreater(w[0], 0.0)

    def test_pairs_are_within_scenario_ambiguous_and_value_weighted(self) -> None:
        scenario = np.array([0, 0, 0, 1, 1, 1])
        battery = np.array(["a", "b", "c", "a", "b", "c"])
        due = np.array([True, False, False, True, False, False])
        anchor = np.array([0.0, 0.1, 9.0, 0.0, 0.1, 9.0])
        value = np.array([300.0, -50.0, -10.0, 300.0, -50.0, -10.0])
        usable = np.ones(6, dtype=bool)
        pos, neg, weight = build_pairs(
            scenario, battery, due, anchor, value, usable,
            delta=1.0, rng=np.random.default_rng(0),
        )
        # The far-away negative (anchor 9.0) is not an ambiguous pair.
        self.assertEqual(pos.size, 2)
        self.assertTrue(set(neg.tolist()) <= {1, 4})
        self.assertTrue((weight > 0).all())
        # Each due device carries one unit of weight in total.
        self.assertAlmostEqual(float(weight.sum()), 1.0)

    def test_no_pairs_across_scenarios(self) -> None:
        scenario = np.array([0, 1])
        pos, neg, _ = build_pairs(
            scenario, np.array(["a", "b"]), np.array([True, False]),
            np.zeros(2), np.array([300.0, -50.0]), np.ones(2, dtype=bool),
            delta=5.0, rng=np.random.default_rng(0),
        )
        self.assertEqual(pos.size, 0)


class DeploymentTest(unittest.TestCase):
    def test_the_residual_only_reorders(self) -> None:
        rng = np.random.default_rng(6)
        rows = 30
        grid = np.maximum.accumulate(
            np.sort(rng.random((rows, len(HORIZON_GRID))), axis=1), axis=1
        )
        remaining = np.full(rows, 300.0)
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        scorer = ResidualScorer(weights=rng.normal(size=len(SIGNALS)) * 3.0)
        features = rng.random((rows, 64)).astype(np.float32) + 2.4
        score = scorer.score(features, remaining, np.arange(rows).astype(str), grid)
        moved = remap(grid, remaining, score, column)
        self.assertTrue(
            np.allclose(np.sort(moved[:, column]), np.sort(grid[:, column]))
        )
        self.assertTrue(np.isclose(moved[:, column].sum(), grid[:, column].sum()))
        self.assertTrue((np.diff(moved, axis=1) >= -1e-12).all())

    def test_zero_weights_are_the_incumbent_exactly(self) -> None:
        rng = np.random.default_rng(7)
        rows = 25
        grid = np.maximum.accumulate(
            np.sort(rng.random((rows, len(HORIZON_GRID))), axis=1), axis=1
        )
        remaining = np.full(rows, 300.0)
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        scorer = ResidualScorer(weights=np.zeros(len(SIGNALS)))
        features = rng.random((rows, 64)).astype(np.float32) + 2.4
        score = scorer.score(features, remaining, np.arange(rows).astype(str), grid)
        self.assertTrue(np.array_equal(remap(grid, remaining, score, column), grid))

    def test_out_of_fold_scorer_leaves_unknown_buildings_alone(self) -> None:
        rng = np.random.default_rng(8)
        rows = 12
        grid = np.maximum.accumulate(
            np.sort(rng.random((rows, len(HORIZON_GRID))), axis=1), axis=1
        )
        devices = np.asarray([f"d{i}" for i in range(rows)])
        scorer = OofResidualScorer(
            by_building={"known": np.ones(len(SIGNALS))},
            building_of={d: ("known" if i < 6 else "fresh") for i, d in enumerate(devices)},
        )
        features = rng.random((rows, 64)).astype(np.float32) + 2.4
        score = scorer.score(features, np.full(rows, 300.0), devices, grid)
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        level = np.clip(grid[:, column], 1e-9, 1 - 1e-9)
        anchor = np.log(level / (1 - level))
        # Rows from a building with no fitted fold keep the incumbent score.
        self.assertTrue(np.allclose(score[6:], anchor[6:]))
        self.assertFalse(np.allclose(score[:6], anchor[:6]))


if __name__ == "__main__":
    unittest.main()
