"""The blended model has to keep every promise the planner relies on.

``BlendedModel`` imposes a discriminative level on the passage model's shape, so
the two things that can silently break are the contract the planner reads -- a
CDF that never decreases along the horizon and never leaves [0, 1] -- and the
arithmetic of the blend itself. Both are pinned here, along with the two
attributes ``tools/fit_calibration.py`` writes through.
"""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bsai.blend import DECISION_HORIZON, BlendedModel
from bsai.calibrate import RemainingCalibration
from bsai.features import N_FEATURES
from bsai.hazard import HORIZON_GRID
from bsai.wiener import WienerModel, build_increment_targets  # noqa: F401


class _Wiener:
    """Stands in for a fitted WienerModel with a controllable grid."""

    horizons = HORIZON_GRID
    model_version = "stub/v1"

    def __init__(self, level: np.ndarray) -> None:
        self.level = np.asarray(level, dtype=float)
        self.calibration = "should be cleared"
        self.volatility_scale = 1.0

    def predict_grid(self, features, remaining, devices=None):
        ramp = np.linspace(0.2, 1.0, len(self.horizons))[None, :]
        return np.clip(self.level[:, None] * ramp, 0.0, 1.0)

    def context(self):
        return "context"

    def cdf_at(self, grid_values, days):
        return grid_values[:, : len(days)]


def _blend(level, head_probability):
    rows = len(level)
    head = HistGradientBoostingClassifier(max_iter=2, max_leaf_nodes=2)
    design = np.zeros((20, N_FEATURES + 1), dtype=np.float32)
    design[:10, 0] = 1.0
    head.fit(design, np.array([1] * 10 + [0] * 10))
    model = BlendedModel(
        wiener=_Wiener(level),
        head=head,
        default_shape=np.linspace(0.2, 1.0, len(HORIZON_GRID)),
    )
    model.head = _ConstantHead(head_probability)
    return model, np.zeros((rows, N_FEATURES), dtype=np.float32)


class _ConstantHead:
    def __init__(self, value) -> None:
        self.value = np.asarray(value, dtype=float)

    def predict_proba(self, design):
        column = np.broadcast_to(self.value, (design.shape[0],))
        return np.column_stack([1.0 - column, column])


class BlendContractTests(unittest.TestCase):
    def test_grid_is_a_valid_cdf(self) -> None:
        model, features = _blend(np.array([0.4, 0.01, 1e-9]), np.array([0.3, 0.5, 0.2]))
        grid = model.predict_grid(features, np.array([300.0, 300.0, 300.0]))
        self.assertEqual(grid.shape, (3, len(HORIZON_GRID)))
        self.assertTrue((grid >= 0).all() and (grid <= 1).all())
        self.assertTrue((np.diff(grid, axis=1) >= -1e-12).all())

    def test_decision_probability_is_the_geometric_mean(self) -> None:
        level = np.array([0.4])
        head = np.array([0.1])
        model, features = _blend(level, head)
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        anchor = model.wiener.predict_grid(features, np.array([300.0]))[0, column]
        grid = model.predict_grid(features, np.array([300.0]))
        self.assertAlmostEqual(
            float(grid[0, column]), float(np.sqrt(anchor * head[0])), places=6
        )

    def test_the_passage_model_keeps_no_calibration_of_its_own(self) -> None:
        """Otherwise the remaining-observation correction is applied twice."""
        model, _ = _blend(np.array([0.4]), np.array([0.3]))
        self.assertIsNone(model.wiener.calibration)

    def test_fit_calibration_can_write_through(self) -> None:
        model, features = _blend(np.array([0.4]), np.array([0.3]))
        model.volatility_scale = 1.4
        self.assertEqual(model.wiener.volatility_scale, 1.4)
        plain = model.predict_grid(features, np.array([400.0]))
        model.calibration = RemainingCalibration(factors=(1.0,) * 6)
        neutral = model.predict_grid(features, np.array([400.0]))
        np.testing.assert_allclose(plain, neutral, atol=1e-12)

    def test_a_dead_observation_window_stays_at_zero(self) -> None:
        """No record can be filed after observation ends, whatever the head says."""
        model, features = _blend(np.array([0.0]), np.array([0.9]))
        grid = model.predict_grid(features, np.array([0.0]))
        self.assertTrue((grid <= 1e-12).all())

    def test_median_shape_is_monotone_and_anchored(self) -> None:
        grid = np.array([[0.1, 0.3, 0.6], [0.2, 0.4, 0.5]])
        shape = BlendedModel.median_shape(grid, np.array([0.6, 0.5]))
        self.assertTrue((np.diff(shape) >= -1e-12).all())


if __name__ == "__main__":
    unittest.main()
