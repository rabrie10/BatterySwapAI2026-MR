"""Censored days-to-EOL regression. MEASURED AND REJECTED -- not shipped.

Kept for the same reason as ``bsai/margin.py``: the measurement should be
reproducible without rebuilding the model. Against the Wiener first-passage
model on the same features, the same folds and the same censoring clip, at
scenario cutoffs out of fold by building:

    swaps/scenario     12      15      18      21
    precision Wiener   0.370   0.349   0.325   0.302
    precision AFT      0.304   0.271   0.251   0.231

Worse on recall too, at every point. See ``docs/V8_HYPOTHESIS_TESTS.md`` section 2
for why, and for the one variant that has *not* been tried (a tail-weighted
loss rather than ``E[log T]``). Do not re-derive this one.

What follows is the design as built.

The shipped pipeline labels each (device, cutoff) with "did an EOL record appear
within h days, yes or no". That throws away *when* it appeared. There are only
82 events on train, so discarding the timing of each one is expensive: a device
that crossed 3 days after a cutoff and one that crossed 41 days after carry the
same label at h = 42, and at h = 7 the second is simply a negative.

An accelerated failure time model keeps the timing. Each row contributes either
an exact log-time (the device crossed while we were watching) or a lower bound
(it had not crossed by the end of its observation), and both enter the same
likelihood. The horizon axis then comes from the fitted distribution, exactly as
it does for the Wiener first-passage model:

    P(record within h) = PHI((log h - mu(x)) / sigma)

so this drops into the same forecaster, the same out-of-fold dispatcher and the
same planner without touching any of them.

Fitting is EM on a Tobit likelihood, which is what lets a gradient-boosted
regressor -- which has no censored loss -- estimate one anyway. Censored rows
get their target replaced by the conditional expectation of the latent log-time
given that it exceeds the censoring point, the regressor is refitted, and the
two steps alternate. Three or four passes are enough; the imputed values stop
moving.

``remaining_observation_days`` is deliberately **not** a covariate here. It is
the censoring time, and letting the mean function see it would let the model
explain censoring rather than degradation. It enters where it belongs, in the
effective horizon: no record can be filed after observation ends, so the window
that matters is ``min(h, remaining)`` -- the same clip ``WienerModel`` applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID

MIN_SIGMA = 0.05
MAX_SIGMA = 5.0
# Below one day the log-time target is dominated by the daily grid resolution.
MIN_TIME = 1.0

DEFAULT_PARAMS = dict(
    max_iter=250,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=20260822,
)


def _inverse_mills(a: np.ndarray) -> np.ndarray:
    """phi(a) / (1 - PHI(a)), stable in the far right tail.

    The naive ratio divides a vanishing density by a vanishing survival and goes
    to nan around a = 8, which is exactly where the long-lived censored rows sit.
    """
    return np.exp(norm.logpdf(a) - norm.logsf(a))


def _sigma_mle(
    residual_event: np.ndarray, residual_censored: np.ndarray
) -> float:
    """Scale that maximises the censored log-normal likelihood, given the mean."""

    def negative_log_likelihood(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        total = 0.0
        if residual_event.size:
            z = residual_event / sigma
            total -= float(np.sum(norm.logpdf(z) - np.log(sigma)))
        if residual_censored.size:
            total -= float(np.sum(norm.logsf(residual_censored / sigma)))
        return total

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(MIN_SIGMA), np.log(MAX_SIGMA)),
        method="bounded",
    )
    return float(np.clip(np.exp(result.x), MIN_SIGMA, MAX_SIGMA))


@dataclass
class AFTModel:
    """Log-normal AFT with a boosted mean. Same interface as ``WienerModel``."""

    mean: HistGradientBoostingRegressor
    sigma: float
    climatology: np.ndarray
    calibration: object | None = None
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-aft/v1"
    # Present so the tools that sweep the Wiener volatility can run unchanged.
    volatility_scale: float = 1.0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        climatology: np.ndarray,
        *,
        params: dict | None = None,
        iterations: int = 4,
    ) -> "AFTModel":
        """EM on the Tobit likelihood.

        ``time`` is the days to crossing for an event and the days of remaining
        observation for a censored row; ``event`` says which.
        """
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        event = np.asarray(event, dtype=bool)
        log_time = np.log(np.maximum(np.asarray(time, dtype=float), MIN_TIME))

        # Start censored rows a little past their bound so the first fit is not
        # told that every survivor died the day we stopped looking.
        target = np.where(event, log_time, log_time + 0.5)
        model = None
        sigma = 1.0
        for _ in range(max(int(iterations), 1)):
            model = HistGradientBoostingRegressor(**settings)
            model.fit(features, target)
            mu = model.predict(features)
            sigma = _sigma_mle(
                (log_time - mu)[event], (log_time - mu)[~event]
            )
            bound = (log_time[~event] - mu[~event]) / sigma
            target = target.copy()
            target[~event] = mu[~event] + sigma * _inverse_mills(bound)
        assert model is not None
        return cls(
            mean=model,
            sigma=float(sigma),
            climatology=np.asarray(climatology, dtype=float),
        )

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def log_time(self, features: np.ndarray) -> np.ndarray:
        return self.mean.predict(features)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        mu = self.log_time(features)
        remaining = np.asarray(remaining, dtype=float)
        grid = np.asarray(self.horizons, dtype=float)
        effective = np.clip(np.minimum(grid[None, :], remaining[:, None]), 0.0, None)
        with np.errstate(divide="ignore"):
            z = (np.log(np.maximum(effective, MIN_TIME)) - mu[:, None]) / self.sigma
        out = np.where(effective <= 0.0, 0.0, norm.cdf(z))
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
