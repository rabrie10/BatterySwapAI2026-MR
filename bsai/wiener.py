"""First-passage probability from a Wiener process with learned drift.

The margin ``m(t) = smooth_v(t) - 2.4`` is a non-monotonic degradation signal:
it trends down over a battery's life but rises and falls with the seasons, so
the monotone processes (gamma, inverse Gaussian) do not apply and the Wiener
process does. Its first passage to a barrier has a closed-form distribution, and
covariates enter through the parameters rather than through the probability --
the standard construction in the degradation literature.

Why this rather than predicting the probability directly:

* **Sample efficiency.** The drift and volatility are estimated from every
  observed window on every device -- hundreds of thousands of them -- while a
  classifier of "does it cross" only ever sees 82 events on the train split.
* **The horizon axis comes for free.** V6 stacked twenty-four horizons and had
  to learn the shape of the curve from data, under a monotone constraint bolted
  on to stop it going backwards. Here ``P(cross by h)`` is a formula in ``h``,
  automatically monotone and automatically consistent between horizons.
* **The parameters transfer.** Drift and volatility are physical quantities. A
  probability level is not, which is why V6's calibration could not be repaired
  across buildings: it over-predicted by 2.21x in the closing scenarios and no
  calibrator fitted on the other four folds could see it.

For a process starting at margin ``m > 0`` with downward drift and an absorbing
barrier at zero, with ``drop`` the expected fall over ``h`` days and ``s`` the
standard deviation of that fall::

    P(cross within h) = PHI((-m + drop) / s)
                      + exp(2 * drop * m / s^2) * PHI((-m - drop) / s)

The second term is the reflection correction: it counts the paths that dip below
the barrier and come back up. Without it the probability would only describe
where the process *ends*, not whether it ever touched zero, and end of life is a
first-passage event.

One caveat is handled explicitly. ``smooth_v`` is a seven-day trailing median of
daily medians, so brief excursions are suppressed and the observed running
minimum is less extreme than a Brownian path with the same increment variance
would give. A single scalar, ``volatility_scale``, absorbs that; it is fitted
out of fold and is the only free parameter added on top of the two regressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID, TrainingFrame
from .margin import EOL_THRESHOLD
from .smoothing import SmoothingCache

VOLTAGE_FEATURE = FEATURE_NAMES.index("voltage")

# Windows used to fit the drift and volatility. Every one of these must fit
# entirely inside a device's pre-crossing history, so long windows are rarer;
# the monotone constraint on the horizon column extrapolates the rest.
FIT_HORIZONS = (7, 14, 21, 28, 42, 63, 91, 126)

# Below this the barrier crossing is certain enough that the formula is skipped.
MIN_MARGIN = 1e-4
MIN_SIGMA = 1e-4
MAX_EXPONENT = 50.0

DEFAULT_PARAMS = dict(
    max_iter=250,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=20260821,
)


def first_passage_probability(
    margin: np.ndarray, drop: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """P(a Wiener path from ``margin`` reaches zero within the window).

    ``drop`` is the expected fall over the window and ``sigma`` the standard
    deviation of that fall, both already in margin units.
    """
    margin = np.asarray(margin, dtype=float)
    drop = np.asarray(drop, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), MIN_SIGMA)

    already = margin <= MIN_MARGIN
    safe_margin = np.where(already, MIN_MARGIN, margin)

    first = norm.cdf((-safe_margin + drop) / sigma)
    # The reflection term is a large exponential times a tiny tail, so it is
    # evaluated in log space; without the clip the exponential overflows long
    # before the product does.
    exponent = np.clip(2.0 * drop * safe_margin / sigma**2, -MAX_EXPONENT, MAX_EXPONENT)
    second = np.exp(exponent + norm.logcdf((-safe_margin - drop) / sigma))

    out = np.clip(first + second, 0.0, 1.0)
    return np.where(already, 1.0, out)


def build_increment_targets(
    frame: TrainingFrame,
    cache: SmoothingCache,
    horizons: tuple[int, ...] = FIT_HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """How far the margin actually fell over each window that we can observe.

    A window is used only if it lies wholly before the device's crossing, so the
    fitted dynamics describe a battery that is still alive -- which is the only
    state the first-passage model is ever asked about.
    """
    designs: list[np.ndarray] = []
    drops: list[np.ndarray] = []

    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")

    for horizon in horizons:
        rows: list[int] = []
        values: list[float] = []
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            margin = margins.get(device_id)
            start = stop
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            limit = last if crossing < 0 else min(last, crossing - 1)
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= limit) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            rows.append(chosen[finite])
            values.append(here[finite] - there[finite])
        if not rows:
            continue
        index = np.concatenate(rows)
        designs.append(
            np.hstack(
                [
                    frame.features[index],
                    np.full((index.size, 1), horizon, dtype=np.float32),
                ]
            )
        )
        drops.append(np.concatenate(values))

    if not designs:
        raise ValueError("no observable windows; check the horizons and the cache")
    return np.vstack(designs), np.concatenate(drops)


@dataclass
class WienerModel:
    """Drift and volatility regressors plus the closed-form passage law.

    Presents the same interface as ``HazardModel`` so the forecaster, the
    out-of-fold dispatcher and the whole Task 2 planner are unchanged.
    """

    drift: HistGradientBoostingRegressor
    scatter: HistGradientBoostingRegressor
    climatology: np.ndarray
    volatility_scale: float = 1.0
    # Correction along the remaining-observation axis; see bsai/calibrate.py.
    # Applied here rather than in the forecaster so the out-of-fold dispatcher
    # routes each device to its own fold's correction automatically.
    calibration: object | None = None
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-wiener/v1"

    @staticmethod
    def _constraints(n_features: int, sign: int) -> np.ndarray:
        # Both the expected fall and its spread grow with the window length.
        out = np.zeros(n_features + 1, dtype=int)
        out[-1] = sign
        return out

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        drop: np.ndarray,
        climatology: np.ndarray,
        *,
        params: dict | None = None,
    ) -> "WienerModel":
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        n_features = design.shape[1] - 1

        drift = HistGradientBoostingRegressor(
            monotonic_cst=cls._constraints(n_features, 1), **settings
        )
        drift.fit(design, drop)

        # Mean absolute residual, converted to a Gaussian sigma. Fitting the
        # spread rather than assuming one is what lets a quiet battery and a
        # noisy one at the same margin get different crossing probabilities.
        residual = np.abs(drop - drift.predict(design))
        scatter = HistGradientBoostingRegressor(
            monotonic_cst=cls._constraints(n_features, 1), **settings
        )
        scatter.fit(design, residual)

        return cls(
            drift=drift,
            scatter=scatter,
            climatology=np.asarray(climatology, dtype=float),
        )

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def probabilities(
        self, features: np.ndarray, horizon: np.ndarray
    ) -> np.ndarray:
        design = np.hstack([features, horizon[:, None].astype(np.float32)])
        drop = np.maximum(self.drift.predict(design), 0.0)
        sigma = (
            np.maximum(self.scatter.predict(design), MIN_SIGMA)
            * np.sqrt(np.pi / 2.0)
            * self.volatility_scale
        )
        margin = features[:, VOLTAGE_FEATURE].astype(float) - EOL_THRESHOLD
        return first_passage_probability(margin, drop, sigma)

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
        grid = np.asarray(self.horizons, dtype=float)
        # No record can be filed after observation ends, so the window that
        # matters is the part of the horizon we would still be watching.
        effective = np.clip(np.minimum(grid[:, None], remaining[None, :]), 0.0, None)
        tall = np.tile(features, (len(self.horizons), 1))
        probability = self.probabilities(tall, effective.reshape(-1))
        probability = np.where(effective.reshape(-1) <= 0.0, 0.0, probability)
        out = probability.reshape(len(self.horizons), rows).T
        if self.calibration is not None:
            out = self.calibration.apply(out, remaining)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        grid = np.asarray(self.horizons, dtype=float)
        days = np.asarray(days, dtype=float)
        if grid_values.shape[0] == 0:
            return np.zeros((0, days.shape[0]))
        anchored_x = np.concatenate([[0.0], grid])
        anchored_y = np.hstack([np.zeros((grid_values.shape[0], 1)), grid_values])
        out = np.empty((grid_values.shape[0], days.shape[0]))
        for row in range(grid_values.shape[0]):
            out[row] = np.interp(days, anchored_x, anchored_y[row])
        return out
