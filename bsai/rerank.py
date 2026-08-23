"""Order-only reranking: same risk mass, different batteries.

Two public results bound what a Task-1 change is allowed to do. V9 added risk
mass -- a learned head plus a x1.15 level scale -- and bought exactly one extra
swap per scenario for zero extra catches (public 2137.22 against V8's 2078.28,
misses unchanged at 2.28). V19 removed risk mass and lost 1.36 catches per
scenario (2113.43, late +403). The remaining direction is the one that changes
neither: keep the per-scenario distribution of predicted risk and change *which*
battery carries which value.

``RankRemapModel`` does that literally. It asks the base model for the whole
scenario's CDF grid, sorts those rows by the 42-day column, sorts the batteries
by a candidate score, and hands the highest CDF row to the highest-scoring
battery. The *multiset of CDF rows* is unchanged, so:

* ``sum(p)`` per scenario is identical to the base model's, and any expected-due
  budget computed from it is identical;
* the set of probability levels the planner's cost tables ever see is identical;
* every curve is still a real curve the base model produced, so monotonicity in
  the horizon and the contract's sum-to-one check survive by construction.

The permutation is applied inside groups of equal remaining-observation days.
``HazardForecaster`` caps each row's curve at its own censoring horizon, so
moving a curve between two devices with different windows would not preserve
mass; grouping makes the invariant exact. It costs almost nothing, because 96 %
of the devices in a scenario share one export date.

The scorer is deliberately not a probability. It never has to be calibrated, it
cannot inflate or deflate the budget, and a monotone transform of it is the same
model -- which is what makes this the lowest-capacity way to spend a new signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID

DECISION_HORIZON = 42
COMPENSATED_FEATURE = FEATURE_NAMES.index("voltage_compensated")


def remap(
    grid: np.ndarray,
    remaining: np.ndarray,
    score: np.ndarray,
    decision_column: int,
) -> np.ndarray:
    """Reassign the grid's rows to the batteries in ``score`` order.

    Ties in the score keep the base model's own ordering, so a scorer that is
    constant is exactly the identity.
    """
    out = grid.copy()
    level = grid[:, decision_column]
    key = np.round(np.asarray(remaining, dtype=float), 6)
    finite = np.where(np.isfinite(score), score, -np.inf)
    for value in np.unique(key):
        rows = np.flatnonzero(key == value)
        if rows.size < 2:
            continue
        # ``-level`` as the secondary key so equal scores keep the incumbent
        # order and a constant scorer is exactly the identity.
        source = rows[np.argsort(-level[rows], kind="stable")]
        target = rows[np.lexsort((-level[rows], -finite[rows]))]
        out[target] = grid[source]
    return out


@dataclass
class RankRemapModel:
    """Presents the ``WienerModel`` interface; only the ordering changes."""

    base: object
    scorer: object
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_NAMES))
    model_version: str = "bsai-rankremap/v1"

    @property
    def decision_column(self) -> int:
        return list(self.horizons).index(DECISION_HORIZON)

    # -- passthroughs so every existing tool keeps working ------------------
    @property
    def volatility_scale(self) -> float:
        return getattr(self.base, "volatility_scale", 1.0)

    @volatility_scale.setter
    def volatility_scale(self, value: float) -> None:
        self.base.volatility_scale = float(value)

    @property
    def calibration(self):
        return getattr(self.base, "calibration", None)

    @calibration.setter
    def calibration(self, value) -> None:
        self.base.calibration = value

    def context(self) -> FeatureContext:
        return self.base.context()

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        return self.base.cdf_at(grid_values, days)

    # -- the one thing this class does --------------------------------------
    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        grid = self.base.predict_grid(features, remaining, devices)
        if grid.shape[0] < 2:
            return grid
        score = np.asarray(
            self.scorer.score(features, remaining, devices, grid), dtype=float
        )
        return remap(grid, remaining, score, self.decision_column)


@dataclass
class LinearScorer:
    """``logit(p) + sum_j weight_j * clipped(signal_j)``.

    One weight per named signal and nothing else: no interactions, no learned
    probability, no building term. ``signals`` returns a dict of named columns
    for the rows it is handed, so the same object serves the cached-frame screen
    and the live forecaster.
    """

    weights: dict[str, float]
    signals: object
    clip: float = 3.0

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        column = list(HORIZON_GRID).index(DECISION_HORIZON)
        p = np.clip(grid[:, column], 1e-9, 1 - 1e-9)
        out = np.log(p / (1.0 - p))
        computed = self.signals(features, remaining, devices, grid)
        for name, weight in self.weights.items():
            values = np.asarray(computed[name], dtype=float)
            values = np.where(np.isfinite(values), values, 0.0)
            out = out + weight * np.clip(values, -self.clip, self.clip)
        return out


def centred_rank(value: np.ndarray) -> np.ndarray:
    """Percentile rank in [-0.5, +0.5]; non-finite values sink to the bottom."""
    value = np.asarray(value, dtype=float)
    if value.size < 2:
        return np.zeros(value.size)
    filled = np.where(np.isfinite(value), value, -np.inf)
    order = np.argsort(np.argsort(filled, kind="stable"), kind="stable")
    return order / (value.size - 1) - 0.5


def decision_level(grid: np.ndarray, remaining: np.ndarray, horizons) -> np.ndarray:
    """The 42-day number the planner sees: the curve, capped at its own censoring
    horizon. This is ``HazardForecaster``'s own arithmetic, repeated here so the
    scorer ranks exactly what the planner will act on."""
    xs = np.concatenate([[0.0], np.asarray(horizons, dtype=float)])
    out = np.empty(grid.shape[0])
    for row in range(grid.shape[0]):
        ys = np.concatenate([[0.0], grid[row]])
        days = max(float(remaining[row]), 0.0)
        out[row] = min(
            float(np.interp(float(DECISION_HORIZON), xs, ys)),
            float(np.interp(days, xs, ys)),
        )
    return np.where(np.asarray(remaining, dtype=float) < 0.0, 0.0, np.clip(out, 0.0, 1.0))


@dataclass
class CompensatedBarrierScorer:
    """Average the model's own ordering with distance to a temperature-corrected
    barrier.

    The first-passage law takes the *measured* margin ``smooth_v - 2.4`` as the
    distance to the barrier. Within a device, residual voltage tracks residual
    temperature at +0.00463 V/degC, positive in 100 % of 454 train devices, so a
    cell measured five degrees warm reads about 0.023 V high -- and near the knee
    0.02 V is roughly two weeks of remaining life. The barrier is a chemical
    state; the reading is not. Measured on the 19,890 scenario rows out of fold
    by building, among the batteries a scenario might plausibly touch, the
    model's own probability orders the real deaths at concordance 0.615 and
    ``voltage_compensated`` orders them at 0.641 -- the raw physical state,
    temperature-corrected, is a *better* within-scenario ranker than the whole
    model built on it.

    Two orderings of comparable quality and different construction average by
    rank, which is the assumption-free combination: no fitted weight, no learned
    level, nothing to calibrate, and ``weight`` exists only so the control at
    zero reproduces the incumbent exactly.
    """

    weight: float = 1.0
    horizons: tuple[int, ...] = HORIZON_GRID

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        level = decision_level(grid, remaining, self.horizons)
        compensated = features[:, COMPENSATED_FEATURE].astype(float)
        return centred_rank(level) + self.weight * centred_rank(-compensated)


@dataclass
class OracleScorer:
    """Diagnostic only: rank by the answer.

    Not shippable and not meant to be. It exists to answer one question the
    top-k screen cannot -- whether the order-only deployment path converts a
    ranking gain into cost through the real planner at all, or whether the
    planner's own economics absorb it. Without that control a screen-level win
    of -129 and a planner-level +23 are indistinguishable from a bug.

    The scenario start is recovered from the row's own remaining-observation
    window, which is exactly ``end_time - origin``, so no extra plumbing is
    needed to line the labels up.
    """

    end_ordinal: dict
    eol_ordinal: dict
    horizon: float = float(DECISION_HORIZON)

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        if devices is None:
            raise ValueError("the oracle needs the device of each row")
        out = np.zeros(len(devices))
        for position, device in enumerate(devices):
            end = self.end_ordinal.get(str(device))
            eol = self.eol_ordinal.get(str(device))
            if end is None or eol is None:
                continue
            days = eol - (end - float(remaining[position]))
            if 0.0 < days <= self.horizon:
                out[position] = 1.0 + (self.horizon - days) / self.horizon
        return out


@dataclass
class SecondModelScorer:
    """Rank by a second forecaster's ordering, keeping the first one's levels.

    This is the deployable form of the v13 ensemble's central idea: a different
    model may order the fleet better than the incumbent while its *level* is not
    to be trusted (``docs/ENSEMBLE_FINDINGS.md`` calls that the budget trap, and
    the censored-drift model's own submission was +179 on public from a -43.5
    local read). Handing over only the order is the part that cannot repeat that.
    """

    other: object
    weight: float = 1.0
    horizons: tuple[int, ...] = HORIZON_GRID

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        mine = decision_level(grid, remaining, self.horizons)
        theirs = decision_level(
            self.other.predict_grid(features, remaining, devices),
            remaining,
            getattr(self.other, "horizons", self.horizons),
        )
        return centred_rank(mine) + self.weight * centred_rank(theirs)


@dataclass
class SumScorer:
    """Add several scorers' outputs, each already on the centred-rank scale."""

    parts: list

    def score(self, features, remaining, devices, grid) -> np.ndarray:
        total = np.zeros(features.shape[0])
        for part in self.parts:
            total = total + np.asarray(
                part.score(features, remaining, devices, grid), dtype=float
            )
        return total
