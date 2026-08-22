"""The order-only invariant is the whole point, so it is tested rather than assumed."""

from __future__ import annotations

import unittest

import numpy as np

from bsai.hazard import HORIZON_GRID
from bsai.rerank import DECISION_HORIZON, LinearScorer, RankRemapModel, remap


class _Base:
    horizons = HORIZON_GRID
    model_version = "test/v1"
    volatility_scale = 1.0
    calibration = None

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid

    def predict_grid(self, features, remaining, devices=None):
        return self.grid.copy()

    def context(self):
        return None

    def cdf_at(self, grid_values, days):
        return grid_values


class _Scorer:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def score(self, features, remaining, devices, grid):
        return self.values


def _random_grid(rows: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.sort(rng.random((rows, len(HORIZON_GRID))), axis=1)
    return np.maximum.accumulate(grid, axis=1)


class RemapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.column = list(HORIZON_GRID).index(DECISION_HORIZON)
        self.grid = _random_grid(40)
        self.remaining = np.full(40, 300.0)

    def test_multiset_of_probabilities_is_preserved(self) -> None:
        rng = np.random.default_rng(11)
        out = remap(self.grid, self.remaining, rng.random(40), self.column)
        self.assertTrue(
            np.allclose(
                np.sort(out[:, self.column]), np.sort(self.grid[:, self.column])
            )
        )

    def test_total_mass_is_preserved(self) -> None:
        rng = np.random.default_rng(12)
        out = remap(self.grid, self.remaining, rng.random(40), self.column)
        self.assertTrue(
            np.isclose(out[:, self.column].sum(), self.grid[:, self.column].sum())
        )
        self.assertTrue(np.allclose(np.sort(out, axis=0), np.sort(self.grid, axis=0)))

    def test_whole_curves_move_together(self) -> None:
        """Every output row is one of the input rows, unbroken."""
        rng = np.random.default_rng(13)
        out = remap(self.grid, self.remaining, rng.random(40), self.column)
        for row in out:
            self.assertTrue(
                any(np.array_equal(row, other) for other in self.grid),
                "a remapped curve is not one of the base model's curves",
            )

    def test_monotone_in_horizon_survives(self) -> None:
        rng = np.random.default_rng(14)
        out = remap(self.grid, self.remaining, rng.random(40), self.column)
        self.assertTrue((np.diff(out, axis=1) >= -1e-12).all())

    def test_constant_score_is_the_identity(self) -> None:
        out = remap(self.grid, self.remaining, np.zeros(40), self.column)
        self.assertTrue(np.array_equal(out, self.grid))

    def test_oracle_score_puts_the_largest_curve_first(self) -> None:
        score = np.arange(40, dtype=float)[::-1]
        out = remap(self.grid, self.remaining, score, self.column)
        self.assertAlmostEqual(
            out[0, self.column], float(self.grid[:, self.column].max())
        )
        self.assertTrue((np.diff(out[:, self.column]) <= 1e-12).all())

    def test_mass_is_preserved_inside_each_remaining_group(self) -> None:
        remaining = np.where(np.arange(40) < 7, 30.0, 300.0)
        rng = np.random.default_rng(15)
        out = remap(self.grid, remaining, rng.random(40), self.column)
        for value in (30.0, 300.0):
            mask = remaining == value
            self.assertTrue(
                np.allclose(
                    np.sort(out[mask, self.column]),
                    np.sort(self.grid[mask, self.column]),
                )
            )

    def test_model_wrapper_matches_the_bare_remap(self) -> None:
        rng = np.random.default_rng(16)
        score = rng.random(40)
        model = RankRemapModel(base=_Base(self.grid), scorer=_Scorer(score))
        out = model.predict_grid(np.zeros((40, 3)), self.remaining, None)
        self.assertTrue(
            np.array_equal(out, remap(self.grid, self.remaining, score, self.column))
        )

    def test_single_row_scenario_is_untouched(self) -> None:
        model = RankRemapModel(base=_Base(self.grid[:1]), scorer=_Scorer(np.zeros(1)))
        out = model.predict_grid(np.zeros((1, 3)), self.remaining[:1], None)
        self.assertTrue(np.array_equal(out, self.grid[:1]))

    def test_non_finite_score_keeps_the_mass(self) -> None:
        score = np.arange(40, dtype=float)
        score[5] = np.nan
        out = remap(self.grid, self.remaining, score, self.column)
        self.assertTrue(
            np.allclose(
                np.sort(out[:, self.column]), np.sort(self.grid[:, self.column])
            )
        )


class LinearScorerTest(unittest.TestCase):
    def test_zero_weights_reproduce_the_incumbent_order(self) -> None:
        grid = _random_grid(30, seed=8)
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        scorer = LinearScorer(weights={}, signals=lambda *a: {})
        score = scorer.score(np.zeros((30, 2)), np.full(30, 300.0), None, grid)
        self.assertTrue(
            np.array_equal(np.argsort(-score), np.argsort(-grid[:, column]))
        )

    def test_signal_is_clipped_and_nan_safe(self) -> None:
        grid = _random_grid(4, seed=9)
        values = np.array([100.0, -100.0, np.nan, 0.5])
        scorer = LinearScorer(
            weights={"x": 1.0}, signals=lambda *a: {"x": values}, clip=2.0
        )
        base = LinearScorer(weights={}, signals=lambda *a: {}).score(
            np.zeros((4, 2)), np.full(4, 300.0), None, grid
        )
        got = scorer.score(np.zeros((4, 2)), np.full(4, 300.0), None, grid)
        self.assertTrue(np.allclose(got - base, [2.0, -2.0, 0.0, 0.5]))


if __name__ == "__main__":
    unittest.main()
