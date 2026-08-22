"""Wiener first passage with a discriminative head, blended in probability space.

Two measurements motivate this, both on the scenario-cutoff population rather
than the strided training grid.

**The first-passage law is right about shape and wrong about level.** Realised
against predicted deaths, bucketed by margin:

    margin 0.05-0.10   146 deaths, 94 predicted   ratio 1.6
    margin 0.10-0.15    65 deaths, 31 predicted   ratio 2.1
    margin 0.15-0.20    41 deaths,  7 predicted   ratio 6.0
    margin 0.20-0.30    25 deaths,  1 predicted   ratio 19.1
    margin 0.30-0.50     6 deaths,  0.06 predicted ratio 99.2

A Gaussian increment gives a tail that is a hundred times too thin once the
margin is more than a couple of tenths of a volt, so 30% of all deaths land in a
region the model reports as impossible. That cannot be repaired by a
multiplicative calibration -- a factor of a hundred on 1e-14 is still zero.

**A boosted classifier on the same features fixes the level but is worse on its
own.** Fitted out of fold by building on the 48 scenario cutoffs it scores
PR-AUC 0.3605 against the passage model's 0.3083, but it loses at every swap
count the leaderboard charges, because it has 454 positives from 82 distinct
devices and no horizon structure at all.

The geometric mean of the two beats both, and by more than either margin:

    PR-AUC              0.3083  ->  0.3889
    AUC below 0.12 V    0.7589  ->  0.7957
    timing at 12 swaps  1802    ->  1704

Blending in probability space rather than by rank is what makes it shippable:
rank depends on the whole scored set, and ``predict_grid`` is called one building
at a time.

The head only ever predicts the 42-day decision, so the passage model keeps its
job of supplying the *shape* across horizons -- the planner needs a full CDF to
choose a service day. The blended level is imposed on that shape. Where the
passage probability is too small for its own shape to be meaningful, a fleet
median shape recorded at fit time is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID
from .wiener import WienerModel

DECISION_HORIZON = 42
# Below this the passage model's own shape is noise, so the fleet median is used.
MIN_SHAPE_ANCHOR = 1e-3
MIN_PROBABILITY = 1e-12

HEAD_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=20260822,
)


@dataclass
class BlendedModel:
    """Presents the ``WienerModel`` interface; the planner is unchanged."""

    wiener: WienerModel
    head: HistGradientBoostingClassifier
    default_shape: np.ndarray
    weight: float = 0.5
    calibration: object | None = None
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_NAMES))
    model_version: str = "bsai-blend/v1"

    def __post_init__(self) -> None:
        # The passage model's own calibration is held here instead, so the two
        # corrections are not applied twice.
        self.wiener.calibration = None

    @property
    def volatility_scale(self) -> float:
        return self.wiener.volatility_scale

    @volatility_scale.setter
    def volatility_scale(self, value: float) -> None:
        self.wiener.volatility_scale = float(value)

    def context(self) -> FeatureContext:
        return self.wiener.context()

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        return self.wiener.cdf_at(grid_values, days)

    @staticmethod
    def head_design(features: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        return np.hstack(
            [features, np.asarray(remaining, dtype=np.float32)[:, None]]
        )

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        remaining = np.asarray(remaining, dtype=float)
        grid = self.wiener.predict_grid(features, remaining)
        column = list(self.horizons).index(DECISION_HORIZON)
        anchor = grid[:, column]

        head = self.head.predict_proba(
            self.head_design(features, remaining)
        )[:, 1]
        blended = np.clip(anchor, MIN_PROBABILITY, 1.0) ** (1.0 - self.weight)
        blended *= np.clip(head, MIN_PROBABILITY, 1.0) ** self.weight

        usable = anchor > MIN_SHAPE_ANCHOR
        shape = np.where(
            usable[:, None],
            grid / np.maximum(anchor, MIN_SHAPE_ANCHOR)[:, None],
            self.default_shape[None, :],
        )
        out = np.clip(shape * blended[:, None], 0.0, 1.0)
        # A device with no observation left cannot file a record, whatever the
        # head thinks; the passage model already zeroes those columns.
        out = np.where(grid <= 0.0, np.minimum(out, grid), out)
        if self.calibration is not None:
            out = self.calibration.apply(out, remaining)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    @classmethod
    def fit_head(
        cls,
        features: np.ndarray,
        remaining: np.ndarray,
        due: np.ndarray,
        *,
        params: dict | None = None,
    ) -> HistGradientBoostingClassifier:
        settings = dict(HEAD_PARAMS)
        settings.update(params or {})
        head = HistGradientBoostingClassifier(**settings)
        head.fit(cls.head_design(features, remaining), np.asarray(due, dtype=int))
        return head

    @staticmethod
    def median_shape(grid: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Fleet median CDF normalised to one at the decision horizon."""
        usable = anchor > 1e-2
        if not usable.any():
            out = np.linspace(0.0, 1.0, grid.shape[1])
            return np.maximum.accumulate(out)
        normalised = grid[usable] / anchor[usable][:, None]
        return np.maximum.accumulate(np.median(normalised, axis=0))
