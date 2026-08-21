from __future__ import annotations

import unittest

import numpy as np

from bsai.features import N_FEATURES
from bsai.hazard import TrainingFrame
from bsai.hybrid import (
    HAZARD_BINS,
    PlattMap,
    build_hazard_table,
    event_duration,
    fit_platt,
    horizon_labels,
)
from bsai.ensemble import ProbabilityBlendModel, logit_blend


def frame() -> TrainingFrame:
    # d0 fails on day 5, d1 is censored on day 6, d2 fails after day 10.
    return TrainingFrame(
        features=np.zeros((3, N_FEATURES), dtype=np.float32),
        device=np.array(["d0", "d1", "d2"]),
        building=np.array(["b0", "b1", "b2"]),
        cutoff=np.array([10, 10, 10]),
        crossing=np.array([15, -1, 22]),
        last_observed=np.array([30, 16, 30]),
        observation_end=np.array([30, 16, 30]),
    )


class HybridTargetsTest(unittest.TestCase):
    def test_event_and_censor_durations(self) -> None:
        duration, event, followup = event_duration(frame())
        np.testing.assert_array_equal(duration, [5.0, 6.0, 12.0])
        np.testing.assert_array_equal(event, [1, 0, 1])
        np.testing.assert_array_equal(followup, [20.0, 6.0, 20.0])

    def test_censored_partial_interval_is_not_a_negative(self) -> None:
        design, label, weight = build_hazard_table(frame(), bins=(3, 7, 14))
        self.assertEqual(design.shape[1], N_FEATURES + 4)
        self.assertEqual(int(label.sum()), 2)
        # d1 is known negative through day 3, but censoring at day 6 means its
        # (3, 7] interval is unknown and must not be included.
        self.assertEqual(len(label), 6)
        self.assertTrue(np.isfinite(weight).all())
        self.assertTrue((weight > 0).all())

    def test_effective_horizon_respects_censoring(self) -> None:
        labels = horizon_labels(frame(), horizons=(3, 7, 14))
        np.testing.assert_array_equal(
            labels,
            [[0, 1, 1], [0, 0, 0], [0, 0, 1]],
        )


class PlattTest(unittest.TestCase):
    def test_map_is_monotone_and_bounded(self) -> None:
        probability = np.linspace(0.01, 0.99, 500)
        label = (probability > 0.65).astype(int)
        mapping = fit_platt(probability, label)
        output = mapping.apply(probability)
        self.assertTrue((np.diff(output) >= 0).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_identity_map(self) -> None:
        values = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(PlattMap().apply(values), values)


class BlendTest(unittest.TestCase):
    def test_endpoints_return_the_component(self) -> None:
        left = np.array([0.01, 0.2, 0.8])
        right = np.array([0.4, 0.6, 0.9])
        np.testing.assert_allclose(logit_blend(left, right, 1.0), left)
        np.testing.assert_allclose(logit_blend(left, right, 0.0), right)

    def test_midpoint_is_bounded(self) -> None:
        mixed = logit_blend(np.array([0.01, 0.9]), np.array([0.8, 0.2]), 0.5)
        self.assertTrue(((mixed > 0) & (mixed < 1)).all())

    def test_row_specific_weights(self) -> None:
        left = np.array([[0.8, 0.8], [0.8, 0.8]])
        right = np.array([[0.2, 0.2], [0.2, 0.2]])
        mixed = logit_blend(left, right, np.array([[1.0], [0.0]]))
        np.testing.assert_allclose(mixed[0], left[0])
        np.testing.assert_allclose(mixed[1], right[1])

    def test_phase_scale_decreases_across_the_scenario_timeline(self) -> None:
        model = ProbabilityBlendModel(left=object(), right=object())
        opening = model.probability_scale_for_origin("2025-09-01")
        closing = model.probability_scale_for_origin("2026-07-27")
        self.assertGreater(opening, closing)
        self.assertAlmostEqual(opening, 1.5)
        self.assertAlmostEqual(closing, 0.36)


if __name__ == "__main__":
    unittest.main()
