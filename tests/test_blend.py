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


def _blend(level, head_probability, heads=2):
    model = BlendedModel(
        wiener=_Wiener(level),
        heads=[_ConstantHead(head_probability) for _ in range(heads)],
    )
    return model, np.zeros((len(level), N_FEATURES), dtype=np.float32)


class _ConstantHead:
    """A head whose probability rises with the horizon, as a real one must."""

    def __init__(self, value) -> None:
        self.value = np.asarray(value, dtype=float)

    def predict_proba(self, design):
        rows = design.shape[0]
        horizon = design[:, -1]
        ramp = np.clip(horizon / float(max(HORIZON_GRID)), 0.05, 1.0)
        # The tall design stacks the same rows once per horizon.
        column = np.tile(self.value, rows // self.value.size) * ramp
        return np.column_stack([1.0 - column, column])


class BlendContractTests(unittest.TestCase):
    def test_grid_is_a_valid_cdf(self) -> None:
        model, features = _blend(np.array([0.4, 0.01, 1e-9]), np.array([0.3, 0.5, 0.2]))
        grid = model.predict_grid(features, np.array([300.0, 300.0, 300.0]))
        self.assertEqual(grid.shape, (3, len(HORIZON_GRID)))
        self.assertTrue((grid >= 0).all() and (grid <= 1).all())
        self.assertTrue((np.diff(grid, axis=1) >= -1e-12).all())

    def test_decision_probability_is_the_geometric_mean(self) -> None:
        model, features = _blend(np.array([0.4]), np.array([0.6]))
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        remaining = np.array([300.0])
        passage = model.wiener.predict_grid(features, remaining)[0, column]
        head = model.head_grid(features, remaining)[0, column]
        grid = model.predict_grid(features, remaining)
        self.assertAlmostEqual(
            float(grid[0, column]), float(np.sqrt(passage * head)), places=6
        )

    def test_batched_head_grid_matches_one_call_per_horizon(self) -> None:
        """The tall design is an optimisation; it must not change the answer."""
        model, features = _blend(np.array([0.4, 0.2]), np.array([0.5, 0.3]))
        remaining = np.array([300.0, 120.0])
        batched = model.head_grid(features, remaining)
        for column, horizon in enumerate(HORIZON_GRID):
            design = model.head_design(features, remaining, horizon)
            total = np.zeros(2)
            for head in model.heads:
                total += np.log(np.clip(head.predict_proba(design)[:, 1], 1e-12, 1.0))
            np.testing.assert_allclose(
                batched[:, column], np.exp(total / len(model.heads)), atol=1e-12
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

    def test_a_record_needs_observation_left_to_be_filed(self) -> None:
        """Same EOL, different observation window: only one can be recorded."""
        label = BlendedModel.record_label(
            np.array([30.0, 30.0, 60.0]), np.array([300.0, 10.0, 300.0]), 42
        )
        np.testing.assert_array_equal(label, np.array([1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
