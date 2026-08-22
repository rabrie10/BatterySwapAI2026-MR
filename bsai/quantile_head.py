"""Multi-quantile increment head: distribution-free first passage.

The incumbent (``bsai/wiener.py``) prices the 42-day margin drop as a Gaussian:
GBDT drift plus GBDT |residual| scatter. Near knee onset that distribution is
the wrong SHAPE, not just the wrong scale. Cells at margin 0.05-0.10 V with an
elevated IR channel (beta_30 >= 0.014) realise a 0.275 due rate while the
Gaussian says 0.14, and the missed dues sit at margin 0.12-0.18 failing ~25
days later -- a population that either stays on the plateau (small drop) or
plunges (drop >= the whole margin). A location-scale Gaussian cannot put 25%
of its mass past the margin without dragging the plateau mass with it, which
is exactly why every probability-floor patch on top of it died: the floor
reshaped p without reshaping the distribution the cost tables integrate.

This head drops the Gaussian. It fits K quantile regressors for the h-day
margin drop (same censor-aware increment targets as the incumbent, same
building-grouped folds) and reads

    P(cross within h | margin m) = P(drop >= m)

directly off the per-row quantile curve by monotone interpolation in q. A
bimodal drop distribution is representable: the low and mid quantiles sit on
the plateau, the top quantiles jump past the margin, and the interpolated
exceedance prices the plunge tail without inflating the plateau cells.

Mechanics that keep the incumbent's guarantees:

* **Quantile non-crossing** -- the K predictions are sorted per row before
  interpolation, so the implied CDF is a genuine CDF.
* **Horizon monotonicity** -- every quantile regressor carries the incumbent's
  monotone constraint on the horizon column. Each order statistic of a set of
  coordinate-wise non-decreasing curves is non-decreasing, so P(cross by h) is
  non-decreasing in h by construction; ``predict_grid`` finishes with the same
  ``maximum.accumulate`` backstop the incumbent uses.
* **Tail handling** -- beyond the fitted 0.05/0.95 quantiles the quantile
  function is extended linearly by half a segment (the local density
  continued to q=0 and q=1). A hard clip at q=0.95 would floor every battery
  at p=0.05 and add ~20 phantom dues per scenario; the linear tail lets a
  plateau cell whose whole predicted drop distribution sits below its margin
  price to exactly zero.

The class presents the same interface as ``WienerModel`` (``predict_grid``,
``cdf_at``, ``context``, ``horizons``, ``calibration``, ``volatility_scale``
as an inert attribute), so ``OofHazardModel`` and ``HazardForecaster``
dispatch it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID
from .margin import EOL_THRESHOLD
from .wiener import MIN_MARGIN, VOLTAGE_FEATURE

# Decile mid-points: dense enough to see a 10-25% plunge mode, coarse enough
# that each regressor still trains on every window.
QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)

# Same family as the incumbent's drift/scatter fits (bsai/wiener.py).
DEFAULT_PARAMS = dict(
    max_iter=250,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=20260821,
)


def crossing_probability(
    quantile_values: np.ndarray,
    margin: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> np.ndarray:
    """P(drop >= margin) from per-row quantile predictions of the drop.

    ``quantile_values`` is (rows, K) aligned with ``quantiles``. Rows are
    sorted (non-crossing), the quantile function is extended linearly to q=0
    and q=1 at the local tail density, and the exceedance probability is the
    monotone interpolation of q at the row's margin, clipped to [0, 1].
    """
    q = np.asarray(quantiles, dtype=float)
    margin = np.asarray(margin, dtype=float)
    values = np.sort(np.asarray(quantile_values, dtype=float), axis=1)

    # Extend to q=0 and q=1 by continuing the outermost segment's density.
    lo = values[:, 0] - (values[:, 1] - values[:, 0]) * (q[0] / (q[1] - q[0]))
    hi = values[:, -1] + (values[:, -1] - values[:, -2]) * (
        (1.0 - q[-1]) / (q[-1] - q[-2])
    )
    xs = np.concatenate([lo[:, None], values, hi[:, None]], axis=1)
    # Ties (a degenerate segment) would make the interpolation ill-defined;
    # nudge into strict increase at a scale far below any real voltage.
    xs = np.maximum.accumulate(xs, axis=1)
    xs = xs + np.arange(xs.shape[1], dtype=float) * 1e-9
    qs = np.concatenate([[0.0], q, [1.0]])

    # Vectorised piecewise-linear interpolation of F(margin) = q(xs = margin).
    segment = np.clip((xs < margin[:, None]).sum(axis=1), 1, xs.shape[1] - 1)
    x0 = np.take_along_axis(xs, (segment - 1)[:, None], axis=1)[:, 0]
    x1 = np.take_along_axis(xs, segment[:, None], axis=1)[:, 0]
    q0 = qs[segment - 1]
    q1 = qs[segment]
    t = np.clip((margin - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    f = q0 + (q1 - q0) * t
    f = np.where(margin <= xs[:, 0], 0.0, f)
    f = np.where(margin >= xs[:, -1], 1.0, f)
    return np.clip(1.0 - f, 0.0, 1.0)


@dataclass
class QuantileHeadModel:
    """K quantile regressors of the h-day drop plus the exceedance readout.

    Presents the same interface as ``WienerModel`` so the forecaster and the
    out-of-fold dispatcher are unchanged.
    """

    regressors: tuple
    quantiles: tuple[float, ...]
    climatology: np.ndarray
    # Interface parity with WienerModel: harmless no-op here, kept so tooling
    # that sets it (fit_calibration, validate_v6 --volatility-scale) works.
    volatility_scale: float = 1.0
    # Correction along the remaining-observation axis; see bsai/calibrate.py.
    # Applied inside predict_grid so the out-of-fold dispatcher routes each
    # device to its own fold's correction automatically.
    calibration: object | None = None
    # Optional fixed-horizon auxiliary quantile sets {horizon: regressors},
    # fitted on that horizon's windows only, WITHOUT the horizon column. Their
    # exceedance applies wherever the effective horizon covers them (a plunge
    # that completes in 14 days is certain by 42), taken as a pointwise max.
    aux: dict | None = None
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-qhead/v1"

    @staticmethod
    def _constraints(n_features: int) -> np.ndarray:
        # The drop distribution shifts up with the window length, quantile by
        # quantile: same monotone constraint as the incumbent's drift.
        out = np.zeros(n_features + 1, dtype=int)
        out[-1] = 1
        return out

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        drop: np.ndarray,
        climatology: np.ndarray,
        *,
        quantiles: tuple[float, ...] = QUANTILES,
        params: dict | None = None,
    ) -> "QuantileHeadModel":
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        constraints = cls._constraints(design.shape[1] - 1)
        regressors = []
        for quantile in quantiles:
            regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(quantile),
                monotonic_cst=constraints,
                **settings,
            )
            regressor.fit(design, drop)
            regressors.append(regressor)
        return cls(
            regressors=tuple(regressors),
            quantiles=tuple(float(q) for q in quantiles),
            climatology=np.asarray(climatology, dtype=float),
        )

    @staticmethod
    def fit_aux(
        features: np.ndarray,
        drop: np.ndarray,
        *,
        quantiles: tuple[float, ...] = QUANTILES,
        params: dict | None = None,
    ) -> tuple:
        """One fixed-horizon quantile set: features only, no horizon column."""
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        regressors = []
        for quantile in quantiles:
            regressor = HistGradientBoostingRegressor(
                loss="quantile", quantile=float(quantile), **settings
            )
            regressor.fit(features, drop)
            regressors.append(regressor)
        return tuple(regressors)

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def _quantile_matrix(self, design: np.ndarray, regressors) -> np.ndarray:
        out = np.empty((design.shape[0], len(regressors)))
        for column, regressor in enumerate(regressors):
            out[:, column] = regressor.predict(design)
        return out

    def probabilities(
        self, features: np.ndarray, horizon: np.ndarray
    ) -> np.ndarray:
        """P(margin reaches zero within ``horizon``) per row."""
        horizon = np.asarray(horizon, dtype=float)
        design = np.hstack(
            [features, horizon[:, None].astype(np.float32)]
        )
        margin = features[:, VOLTAGE_FEATURE].astype(float) - EOL_THRESHOLD
        values = self._quantile_matrix(design, self.regressors)
        probability = crossing_probability(values, margin, self.quantiles)
        aux = getattr(self, "aux", None)
        if aux:
            for aux_horizon in sorted(aux):
                covered = horizon >= float(aux_horizon)
                if not covered.any():
                    continue
                aux_values = self._quantile_matrix(
                    features[covered], aux[aux_horizon]
                )
                aux_probability = crossing_probability(
                    aux_values, margin[covered], self.quantiles
                )
                probability[covered] = np.maximum(
                    probability[covered], aux_probability
                )
        already = margin <= MIN_MARGIN
        return np.where(already, 1.0, probability)

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
