"""Small forecast-model ensembles that preserve the hazard-model interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureContext


def logit_blend(left: np.ndarray, right: np.ndarray, left_weight) -> np.ndarray:
    """Geometric odds blend, bounded and rank-sensitive in the rare-event tail."""
    weight = np.clip(np.asarray(left_weight, dtype=float), 0.0, 1.0)
    a = np.clip(np.asarray(left, dtype=float), 1e-6, 1.0 - 1e-6)
    b = np.clip(np.asarray(right, dtype=float), 1e-6, 1.0 - 1e-6)
    logit_a = np.log(a / (1.0 - a))
    logit_b = np.log(b / (1.0 - b))
    mixed = weight * logit_a + (1.0 - weight) * logit_b
    return 1.0 / (1.0 + np.exp(-np.clip(mixed, -40.0, 40.0)))


@dataclass
class ProbabilityBlendModel:
    left: object
    right: object
    left_weight: float = 0.5
    tail_anchor: int = 42
    right_tail: bool = True
    adapt_to_observation_window: bool = True
    phase_start: str = "2025-09-01"
    phase_knots: tuple[float, ...] = (3.5, 11.5, 19.5, 31.5, 43.5)
    phase_scales: tuple[float, ...] = (1.50, 1.09, 0.90, 0.57, 0.36)
    model_version: str = "bsai-probability-blend/v1"

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(self.left.horizons)

    def context(self) -> FeatureContext:
        return self.left.context()

    def probability_scale_for_origin(self, origin) -> float:
        """Scenario-level incidence correction fitted on OOF weekly forecasts.

        The raw models reverse the observed time trend: they under-predict the
        opening scenarios and over-predict the closing ones.  The knots are
        eight-scenario block ratios, pooled monotonically where adjacent blocks
        disagree.  Only the probability level changes; battery ranking and the
        conditional curve shape are untouched.
        """
        start = pd.Timestamp(self.phase_start).normalize()
        value = pd.Timestamp(origin)
        if value.tzinfo is not None:
            value = value.tz_localize(None)
        week = float((value.normalize() - start).total_seconds() / (7.0 * 86_400.0))
        return float(np.interp(week, self.phase_knots, self.phase_scales))

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        left = self.left.predict_grid(features, remaining, devices)
        right = self.right.predict_grid(features, remaining, devices)
        weight = self.left_weight
        if self.adapt_to_observation_window:
            observable_fraction = np.clip(
                np.asarray(remaining, dtype=float) / max(float(self.tail_anchor), 1.0),
                0.0,
                1.0,
            )
            weight = float(self.left_weight) * observable_fraction[:, None]
        mixed = logit_blend(left, right, weight)
        if self.right_tail and self.tail_anchor in self.horizons:
            anchor_column = self.horizons.index(self.tail_anchor)
            anchor = mixed[:, anchor_column]
            right_anchor = right[:, anchor_column]
            for column in range(anchor_column + 1, len(self.horizons)):
                conditional = np.clip(
                    (right[:, column] - right_anchor)
                    / np.maximum(1.0 - right_anchor, 1e-8),
                    0.0,
                    1.0,
                )
                mixed[:, column] = anchor + (1.0 - anchor) * conditional
        return np.maximum.accumulate(np.clip(mixed, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        xs = np.concatenate([[0.0], np.asarray(self.horizons, dtype=float)])
        out = np.empty((grid_values.shape[0], len(days)), dtype=float)
        for row in range(grid_values.shape[0]):
            ys = np.concatenate([[0.0], grid_values[row]])
            out[row] = np.interp(days, xs, ys)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)


__all__ = ["ProbabilityBlendModel", "logit_blend"]
