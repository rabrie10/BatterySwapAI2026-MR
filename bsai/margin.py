"""First-passage probability by regressing the voltage margin, not classifying
the event.

End of life is defined as ``smooth_series(voltage) < 2.4``. A classifier sees
only whether that fired, so the train split offers it 82 examples however many
rows we stack. But the quantity that *decides* the event -- the running minimum
of the margin ``smooth_v - 2.4`` -- is observable on all 26,366 device-days.

For device ``b``, cutoff ``t`` and horizon ``k``::

    observable = min(k, E_b - t)          # E_b: last day a record can exist
    y(b, t, k) = min over j in 1..observable of (smooth_v(b, t + j) - 2.4)

    EOL recorded within k   <=>   y(b, t, k) < 0

That is an exact restatement of the label, not an approximation, and it buys
three things the classifier cannot have:

* roughly three hundred times the supervision, all of it continuous;
* the distinction between a battery that ended the window at margin 0.004 and
  one that ended at 0.31 -- identical negatives to a classifier;
* censoring for free. No record can be filed after ``E_b``, so truncating the
  minimum there *is* the definition. The V6 model needed a
  ``remaining_observation_days`` feature to patch this; here the effective
  horizon ``min(k, E_b - t)`` is the only horizon the model ever sees.

Probability comes from quantile regression: fit ``y`` at a ladder of quantiles
with the effective horizon as a feature, then read ``P(y < 0)`` off the fitted
quantile curve. The constraint is that ``y`` cannot rise as the horizon grows,
because a longer window can only push a running minimum down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID, TrainingFrame
from .smoothing import SmoothingCache

EOL_THRESHOLD = 2.4

# A margin this far above the threshold is "no crossing possible"; it only has
# to exceed any real margin, and cells with no observable future get it.
NO_CROSSING_MARGIN = 9.0

# Volts of margin worth one e-fold of risk, used only to extrapolate past the
# ends of the quantile ladder. The median margin on the day a battery crosses is
# -0.055 V and 2,023 surviving rows sit within 0.02 V of the threshold, so this
# is the scale on which "nearly crossed" is measured.
MARGIN_SCALE = 0.02

# Training horizons. Inference uses the full HORIZON_GRID -- the effective
# horizon is a continuous feature under a monotone constraint, so the model
# interpolates between these safely, and every extra training horizon multiplies
# the row count.
TRAIN_HORIZONS = (3, 7, 14, 21, 28, 35, 42, 56, 84, 126, 200, 300, 400)

# Dense where decisions are made. The break-even swap probability is about 0.26
# (a wasted swap costs roughly 87, a missed one 270), so resolution matters most
# between 0.05 and 0.45.
QUANTILES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.65)

DEFAULT_PARAMS = dict(
    max_iter=300,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=20260821,
)


def forward_running_minimum(
    margin: np.ndarray, horizon: int, observation_end: int
) -> np.ndarray:
    """``out[i] = min(margin[i+1 : i+1+min(horizon, observation_end - i)])``.

    NaN days carry no record -- the official detector cannot fire on them -- so
    they are treated as "no crossing" rather than propagated.
    """
    filled = np.where(np.isnan(margin), NO_CROSSING_MARGIN, margin)
    count = filled.shape[0]
    out = np.full(count, NO_CROSSING_MARGIN, dtype=np.float64)
    if count < 2:
        return out

    # min over margin[i+1 : i+1+horizon], via a rolling minimum on the reverse.
    reversed_min = (
        pd.Series(filled[::-1]).rolling(int(horizon), min_periods=1).min().to_numpy()
    )
    windowed = reversed_min[::-1][1:]  # windowed[i] = min(margin[i+1 : i+1+horizon])

    # min over margin[i+1 : observation_end+1], for rows whose window runs past
    # the last day a record could be filed.
    limit = min(int(observation_end), count - 1)
    suffix = np.full(count, NO_CROSSING_MARGIN, dtype=np.float64)
    if limit >= 0:
        suffix[: limit + 1] = np.minimum.accumulate(filled[: limit + 1][::-1])[::-1]

    index = np.arange(count - 1)
    truncated = (index + int(horizon)) > limit
    out[:-1] = np.where(truncated, suffix[1:], windowed)
    # Past the observation end there is no observable future at all.
    out[np.arange(count) >= limit] = NO_CROSSING_MARGIN
    return out


def build_margin_targets(
    frame: TrainingFrame,
    cache: SmoothingCache,
    horizons: tuple[int, ...] = TRAIN_HORIZONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack every cutoff against every horizon.

    Returns ``(design, y, row_index, effective_horizon)`` where ``design`` is
    the feature block with the effective horizon appended as its last column.
    """
    by_device: dict[str, np.ndarray] = {}
    for device_id, series in cache.devices.items():
        by_device[device_id] = series.smooth_voltage - EOL_THRESHOLD

    order = np.argsort(frame.device, kind="stable")
    parts_index: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_horizon: list[np.ndarray] = []

    for horizon in horizons:
        y = np.full(len(frame), NO_CROSSING_MARGIN, dtype=np.float64)
        effective = np.zeros(len(frame), dtype=np.float32)
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            rows = order[start:stop]
            margin = by_device.get(device_id)
            if margin is not None:
                observation_end = int(frame.observation_end[rows[0]])
                curve = forward_running_minimum(margin, horizon, observation_end)
                cutoffs = frame.cutoff[rows]
                inside = (cutoffs >= 0) & (cutoffs < curve.shape[0])
                y[rows[inside]] = curve[cutoffs[inside]]
                effective[rows] = np.clip(
                    np.minimum(horizon, observation_end - cutoffs), 0.0, None
                )
            start = stop
        parts_index.append(np.arange(len(frame)))
        parts_y.append(y)
        parts_horizon.append(effective)

    row_index = np.concatenate(parts_index)
    effective_horizon = np.concatenate(parts_horizon)
    design = np.hstack(
        [frame.features[row_index], effective_horizon[:, None].astype(np.float32)]
    )
    return design, np.concatenate(parts_y), row_index, effective_horizon


@dataclass
class MarginModel:
    """Quantile regressors on the running-minimum margin.

    Presents the same interface as ``HazardModel`` so the forecaster, the
    out-of-fold dispatcher and the Task 2 planner are all unchanged.
    """

    regressors: dict[float, HistGradientBoostingRegressor]
    climatology: np.ndarray
    horizons: tuple[int, ...] = HORIZON_GRID
    quantiles: tuple[float, ...] = QUANTILES
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-margin/v1"

    @staticmethod
    def _monotonic_constraints(n_features: int) -> np.ndarray:
        # A longer window can only push a running minimum down.
        constraints = np.zeros(n_features + 1, dtype=int)
        constraints[-1] = -1
        return constraints

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        y: np.ndarray,
        climatology: np.ndarray,
        *,
        quantiles: tuple[float, ...] = QUANTILES,
        params: dict | None = None,
    ) -> "MarginModel":
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        constraints = cls._monotonic_constraints(design.shape[1] - 1)
        regressors: dict[float, HistGradientBoostingRegressor] = {}
        for quantile in quantiles:
            regressor = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(quantile),
                monotonic_cst=constraints,
                **settings,
            )
            regressor.fit(design, y)
            regressors[float(quantile)] = regressor
        return cls(
            regressors=regressors,
            climatology=np.asarray(climatology, dtype=float),
            quantiles=tuple(quantiles),
        )

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def _crossing_probability(self, predicted: np.ndarray) -> np.ndarray:
        """Read ``P(y < 0)`` off a ladder of predicted quantiles.

        ``predicted`` is (rows, n_quantiles), ascending in quantile. Quantile
        crossing is repaired by sorting before interpolating, which is the
        standard remedy and cheaper than fitting a joint model.
        """
        levels = np.asarray(self.quantiles, dtype=float)
        values = np.sort(predicted, axis=1)
        rows = values.shape[0]
        out = np.zeros(rows, dtype=float)

        below = values < 0.0
        any_below = below.any(axis=1)
        # Highest quantile whose predicted margin is still negative.
        last_negative = np.where(any_below, below.sum(axis=1) - 1, -1)

        # Every quantile already negative: the crossing probability is above the
        # top of the ladder. Pinning them all at one value would erase the
        # ranking among our most confident calls, which are exactly the swaps we
        # most want ordered correctly, so extrapolate on how far below zero the
        # top quantile sits.
        saturated = last_negative == len(levels) - 1
        if saturated.any():
            depth = np.clip(-values[saturated, -1], 0.0, None)
            out[saturated] = 1.0 - (1.0 - levels[-1]) * np.exp(-depth / MARGIN_SCALE)

        # Straddles: interpolate between the bracketing quantiles on margin.
        straddle = any_below & ~saturated
        if straddle.any():
            index = last_negative[straddle]
            low_value = values[straddle, index]
            high_value = values[straddle, index + 1]
            low_level = levels[index]
            high_level = levels[index + 1]
            span = np.where(
                np.abs(high_value - low_value) < 1e-12, 1e-12, high_value - low_value
            )
            weight = np.clip((0.0 - low_value) / span, 0.0, 1.0)
            out[straddle] = low_level + weight * (high_level - low_level)

        # Nothing negative: below the bottom of the ladder. Scale by how close
        # the lowest quantile came to zero, so ranking survives down there --
        # this is the region that decides whether we swap 13 or 21 batteries.
        none_below = ~any_below
        if none_below.any():
            closest = values[none_below, 0]
            out[none_below] = levels[0] * np.exp(
                -np.clip(closest, 0.0, None) / MARGIN_SCALE
            )
        return np.clip(out, 0.0, 1.0)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        remaining = np.asarray(remaining, dtype=np.float32)
        horizons = np.asarray(self.horizons, dtype=np.float32)
        # One tall design over every horizon rather than one pass per horizon:
        # seven regressors times twenty-four horizons is 168 calls per scenario
        # otherwise, and the ensembles are small enough that the per-call
        # overhead dominates.
        effective = np.clip(
            np.minimum(horizons[:, None], remaining[None, :]), 0.0, None
        ).reshape(-1, 1)
        design = np.hstack([np.tile(features, (len(self.horizons), 1)), effective])
        predicted = np.column_stack(
            [self.regressors[q].predict(design) for q in self.quantiles]
        )
        probability = self._crossing_probability(predicted)
        # No observable future means no record can be filed.
        probability = np.where(effective[:, 0] <= 0.0, 0.0, probability)
        out = probability.reshape(len(self.horizons), rows).T
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        grid = np.asarray(self.horizons, dtype=float)
        days = np.asarray(days, dtype=float)
        if grid_values.shape[0] == 0:
            return np.zeros((0, days.shape[0]))
        anchored_x = np.concatenate([[0.0], grid])
        anchored_y = np.hstack(
            [np.zeros((grid_values.shape[0], 1)), grid_values]
        )
        out = np.empty((grid_values.shape[0], days.shape[0]))
        for row in range(grid_values.shape[0]):
            out[row] = np.interp(days, anchored_x, anchored_y[row])
        return out
