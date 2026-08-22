"""Wiener first passage blended with a discriminative head, horizon by horizon.

Two measurements motivate this, both on the scenario-cutoff population rather
than the strided training grid.

**The passage law is right about shape and wrong about level.** Realised against
predicted deaths, bucketed by margin (``smooth_v - 2.4``):

    margin 0.05-0.10   141 deaths,  118 predicted   ratio 1.2
    margin 0.10-0.15    74 deaths,   36 predicted   ratio 2.1
    margin 0.15-0.20    42 deaths,    7 predicted   ratio 6.0
    margin 0.20-0.30    25 deaths,    1 predicted   ratio 19.1
    margin 0.30-0.50     8 deaths, 0.06 predicted   ratio 99.2

A Gaussian increment gives a tail a hundred times too thin once the margin is
more than a couple of tenths of a volt, so the due batteries the plan misses
carry a *median predicted probability of 0.008* -- declared safe, not merely
ranked low. Only 3.2% of misses were ranked above the lowest-ranked battery the
planner did swap, so the decision layer is not at fault. Inside that "safe"
population plain ``voltage`` separates the deaths at AUC 0.913.

**A boosted classifier fixes the level and is worse on its own.** Out of fold by
building it beats the passage model on PR-AUC and loses at every swap count the
leaderboard charges, because it has 454 positives from 82 distinct devices.

The geometric mean beats both ends of the weight sweep, which is the signature of
two decorrelated views rather than one dominating.

**The head is fitted at several horizons, not just the 42-day decision.** Stacking
the same rows at 14 to 126 days takes the positives from 454 to 1114 and gives
the head a horizon axis of its own, under a monotone constraint. That earns two
things: a better ranking at the decision (timing 1598 -> 1506 at fifteen swaps,
bagged over five seeds), and a *shape* -- so the blend applies at every horizon
instead of being imposed on the passage model's shape at one point. The planner
reads a full CDF to choose a service day, and the geometric mean of two functions
monotone in the horizon is itself monotone, so the contract survives.

Heads are bagged over five seeds. A single head moves the timing screen by ±34
between seeds, which is larger than most of the effects worth chasing here.

Blending in probability space rather than by rank is what makes it shippable:
rank depends on the whole scored set, and ``predict_grid`` is called one building
at a time. The two are equivalent in quality (PR-AUC 0.3889 against 0.3876).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID
from .wiener import WienerModel

DECISION_HORIZON = 42
MIN_PROBABILITY = 1e-12

# Horizons the head is stacked over. Short enough to keep the 42-day decision
# sharp, long enough that the tail of the grid is anchored by data rather than by
# extrapolation.
HEAD_HORIZONS = (14, 21, 28, 42, 63, 91, 126)
HEAD_SEEDS = (20260822, 1, 2, 3, 7)

HEAD_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
)


@dataclass
class BlendedModel:
    """Presents the ``WienerModel`` interface; the planner is unchanged."""

    wiener: WienerModel
    heads: list
    weight: float = 0.5
    calibration: object | None = None
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_NAMES))
    model_version: str = "bsai-blend/v2"

    def __post_init__(self) -> None:
        # The passage model's own calibration is held here instead, so the
        # remaining-observation correction is not applied twice.
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
    def head_design(
        features: np.ndarray, remaining: np.ndarray, horizon: float
    ) -> np.ndarray:
        rows = features.shape[0]
        return np.hstack(
            [
                features,
                np.asarray(remaining, dtype=np.float32).reshape(rows, 1),
                np.full((rows, 1), float(horizon), dtype=np.float32),
            ]
        )

    def head_grid(self, features: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        """The bagged head's CDF on the model's own horizon grid.

        One tall design per head rather than one call per horizon: five heads
        over a twenty-four point grid is a hundred and twenty calls into the tree
        ensemble, and the per-call overhead dominates. Batching turns 1.6 seconds
        per scenario into a tenth of that, which is the difference between fitting
        the thirty-minute evaluation budget and not.
        """
        rows = features.shape[0]
        count = len(self.horizons)
        tall = np.hstack(
            [
                np.tile(features, (count, 1)),
                np.tile(
                    np.asarray(remaining, dtype=np.float32).reshape(rows, 1), (count, 1)
                ),
                np.repeat(
                    np.asarray(self.horizons, dtype=np.float32), rows
                ).reshape(-1, 1),
            ]
        )
        total = np.zeros(tall.shape[0])
        for head in self.heads:
            total += np.log(
                np.clip(head.predict_proba(tall)[:, 1], MIN_PROBABILITY, 1.0)
            )
        return np.exp(total / len(self.heads)).reshape(count, rows).T

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
        passage = self.wiener.predict_grid(features, remaining)
        head = self.head_grid(features, remaining)

        out = np.clip(passage, MIN_PROBABILITY, 1.0) ** (1.0 - self.weight)
        out *= np.clip(head, MIN_PROBABILITY, 1.0) ** self.weight
        # No record can be filed once observation has ended, whatever the head
        # believes; the passage model already zeroes those columns.
        out = np.where(passage <= 0.0, 0.0, out)
        if self.calibration is not None:
            out = self.calibration.apply(out, remaining)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    @staticmethod
    def record_label(
        days_to_eol: np.ndarray, remaining: np.ndarray, horizon: float
    ) -> np.ndarray:
        """Did an EOL *record* land within ``horizon`` days?

        A device whose observation ends before the horizon is a genuine negative,
        not a censored unknown -- dropping those rows is what removed the
        population that dominates the closing scenarios (HANDOVER.md section 3).
        """
        return (
            (np.asarray(days_to_eol) <= horizon)
            & (np.asarray(days_to_eol) <= np.asarray(remaining))
        ).astype(int)

    @classmethod
    def fit_heads(
        cls,
        features: np.ndarray,
        remaining: np.ndarray,
        days_to_eol: np.ndarray,
        *,
        horizons: tuple[int, ...] = HEAD_HORIZONS,
        seeds: tuple[int, ...] = HEAD_SEEDS,
        params: dict | None = None,
    ) -> list:
        """Stack the rows over horizons and fit one head per seed.

        The label is "an EOL *record* exists within h days", so a device whose
        observation ends before h is a genuine negative rather than a censored
        unknown -- the same construction the shipped labels use.
        """
        designs, labels = [], []
        for horizon in horizons:
            designs.append(cls.head_design(features, remaining, horizon))
            labels.append(cls.record_label(days_to_eol, remaining, horizon))
        design = np.vstack(designs)
        label = np.concatenate(labels)
        constraint = np.zeros(design.shape[1], dtype=int)
        constraint[-1] = 1  # the CDF cannot fall as the horizon grows

        settings = dict(HEAD_PARAMS)
        settings.update(params or {})
        heads = []
        for seed in seeds:
            head = HistGradientBoostingClassifier(
                monotonic_cst=constraint, random_state=seed, **settings
            )
            head.fit(design, label)
            heads.append(head)
        return heads
