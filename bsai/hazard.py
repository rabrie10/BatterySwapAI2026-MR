"""Multi-horizon first-passage model.

The evaluator's cost is driven by a distribution, not a point: swapping one day
late costs twenty times as much as swapping one day early, and deferring a
battery that turns out to be due costs between 60 and 480 hours. So the model
predicts P(EOL recorded within h days) for a grid of h and lets the decision
layer integrate, rather than predicting a remaining-useful-life number.

A mean regressor is actively wrong here. Measured on train, one has MAE 25 days
overall but is optimistic by +17 days exactly where it matters (true RUL under
14 days) and pessimistic by -21 days at 70-120 days: regression to the mean, in
the direction that makes you late.

Horizon enters as a feature with a monotone constraint, so one model serves
every horizon and the predicted CDF cannot decrease as the horizon grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .features import (
    FEATURE_NAMES,
    DeviceView,
    FeatureContext,
    feature_row,
    fleet_climatology,
)
from .smoothing import SmoothingCache

# Dense through the 42-day planning window, then coarse out to the longest
# remaining observation window a scenario can face (about 334 days on train).
HORIZON_GRID = (
    3, 7, 10, 14, 17, 21, 25, 28, 32, 35, 39, 42,
    49, 56, 70, 84, 105, 126, 150, 180, 220, 270, 330, 400,
)

DEFAULT_PARAMS = dict(
    max_iter=400,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=20260821,
)


@dataclass
class TrainingFrame:
    """One row per (device, cutoff): features plus what the future did."""

    features: np.ndarray
    device: np.ndarray
    building: np.ndarray
    cutoff: np.ndarray
    crossing: np.ndarray  # grid index of EOL, or -1 when never recorded
    last_observed: np.ndarray
    observation_end: np.ndarray  # grid index of the device's last possible record

    def __len__(self) -> int:
        return int(self.features.shape[0])


def build_training_frame(
    cache: SmoothingCache,
    eol_index: dict[str, int | None],
    building_of: dict[str, str],
    observation_end_index: dict[str, int],
    *,
    stride: int = 3,
    warmup_days: int = 45,
    cutoff_days: np.ndarray | None = None,
) -> TrainingFrame:
    """Sample cutoffs along each device's history.

    Cutoffs stop at the crossing. Device-days after EOL are not decision points
    and, left in, they dominate the top of any ranking with guaranteed
    negatives -- which is exactly how a first attempt at this measured
    precision@50 = 0.000 before the bug was found.

    ``cutoff_days`` replaces the per-device stride with a shared grid of day
    ordinals. That is how a scenario actually samples -- every alive device on
    one date -- and the two populations are not interchangeable for
    calibration: fitted on the stride population, the model's top probability
    bucket predicted 0.87 against a realised 0.39 at scenario cutoffs.
    """
    context = FeatureContext(
        climatology=fleet_climatology(
            {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
        )
    )
    rows: list[list[float]] = []
    device: list[str] = []
    building: list[str] = []
    cutoff: list[int] = []
    crossing: list[int] = []
    last_observed: list[int] = []
    observation_end: list[int] = []

    for device_id, series in cache.devices.items():
        valid = np.flatnonzero(~np.isnan(series.smooth_voltage))
        if valid.size == 0:
            continue
        first, last = int(valid[0]), int(valid[-1])
        cross = eol_index.get(device_id)
        stop = last if cross is None else min(last, int(cross))
        view = DeviceView(series.smooth_voltage, series.smooth_temperature)
        if cutoff_days is None:
            indices = range(first + warmup_days, stop + 1, stride)
        else:
            local = cutoff_days - series.origin
            indices = local[(local >= first + warmup_days) & (local <= stop)]
        for index in indices:
            index = int(index)
            row = feature_row(view, index, series.origin + index, context)
            if row is None:
                continue
            rows.append(row)
            device.append(device_id)
            building.append(building_of.get(device_id, ""))
            cutoff.append(index)
            crossing.append(-1 if cross is None else int(cross))
            last_observed.append(last)
            observation_end.append(int(observation_end_index.get(device_id, last)))

    return TrainingFrame(
        features=np.asarray(rows, dtype=np.float32),
        device=np.asarray(device),
        building=np.asarray(building),
        cutoff=np.asarray(cutoff, dtype=np.int64),
        crossing=np.asarray(crossing, dtype=np.int64),
        last_observed=np.asarray(last_observed, dtype=np.int64),
        observation_end=np.asarray(observation_end, dtype=np.int64),
    )


REMAINING_CLIP = (-60.0, 500.0)


def stack_horizons(
    frame: TrainingFrame, horizons=HORIZON_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand each cutoff into every horizon. Nothing is censored.

    The scored quantity is not "will this battery cross 2.4 V" but "will an EOL
    *record* exist by then", and a record can only exist while the device is
    still being observed. That makes every row fully labelled: a device that has
    not crossed by its observation end is a genuine negative at any horizon, not
    a censored unknown.

    Dropping those rows -- the previous behaviour -- removed exactly the
    population that dominates the closing scenarios, and the model over-predicted
    there by 2.2x while being slightly *under*-confident in the opening ones. So
    the remaining observation window enters as a feature: it is known at plan
    time from ``locations.end_time``, and without it the model cannot express
    "no time left for a record to be filed".
    """
    remaining = np.clip(
        (frame.observation_end - frame.cutoff).astype(np.float32), *REMAINING_CLIP
    )
    parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    horizon_values: list[np.ndarray] = []
    all_rows = np.arange(len(frame))
    for horizon in horizons:
        crossed = (frame.crossing >= 0) & ((frame.crossing - frame.cutoff) <= horizon)
        parts.append(all_rows)
        labels.append(crossed.astype(np.int8))
        horizon_values.append(np.full(len(frame), horizon, dtype=np.float32))

    index = np.concatenate(parts)
    horizon_column = np.concatenate(horizon_values)
    design = np.hstack(
        [frame.features[index], remaining[index][:, None], horizon_column[:, None]]
    )
    return design, np.concatenate(labels), index, horizon_column


@dataclass
class HazardModel:
    """Fitted classifier plus the per-band calibration and split constants."""

    classifier: HistGradientBoostingClassifier
    climatology: np.ndarray
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    calibrators: dict[int, IsotonicRegression] = field(default_factory=dict)
    model_version: str = "bsai-hazard/v1"

    @staticmethod
    def _monotonic_constraints(n_features: int) -> np.ndarray:
        # Two appended columns: remaining observation window, then horizon.
        constraints = np.zeros(n_features + 2, dtype=int)
        constraints[-2] = 1  # more observation time can only add records
        constraints[-1] = 1  # the CDF cannot fall as the horizon grows
        return constraints

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        labels: np.ndarray,
        climatology: np.ndarray,
        *,
        params: dict | None = None,
    ) -> "HazardModel":
        settings = dict(DEFAULT_PARAMS)
        settings.update(params or {})
        classifier = HistGradientBoostingClassifier(
            monotonic_cst=cls._monotonic_constraints(design.shape[1] - 2), **settings
        )
        classifier.fit(design, labels)
        return cls(classifier=classifier, climatology=np.asarray(climatology, dtype=float))

    def fit_calibration(
        self, probabilities: np.ndarray, labels: np.ndarray, horizons: np.ndarray
    ) -> None:
        """Isotonic per horizon band, on out-of-fold predictions only.

        With 82 events, isotonic on its own would carve the probability axis into
        near-empty steps, so bands pool neighbouring horizons.
        """
        for low, high in ((0, 21), (21, 42), (42, 120), (120, 10**6)):
            mask = (horizons > low) & (horizons <= high)
            if mask.sum() < 500 or labels[mask].sum() < 20:
                continue
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(probabilities[mask], labels[mask])
            self.calibrators[high] = calibrator

    def _calibrate(self, probabilities: np.ndarray, horizon: float) -> np.ndarray:
        for high in sorted(self.calibrators):
            if horizon <= high:
                return self.calibrators[high].predict(probabilities)
        return probabilities

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        """CDF at every horizon in the grid, shape (n_devices, n_horizons).

        ``devices`` is unused here; it exists so that a fold-dispatching model
        can share the forecaster's call site during out-of-fold validation.
        """
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        # One tall design rather than one call per horizon: the horizon is just
        # another column, and a single pass over 24x the rows is markedly
        # cheaper than 24 passes over the tree ensemble.
        horizon_column = np.repeat(
            np.asarray(self.horizons, dtype=np.float32), rows
        )[:, None]
        remaining_column = np.tile(
            np.clip(np.asarray(remaining, dtype=np.float32), *REMAINING_CLIP),
            len(self.horizons),
        )[:, None]
        design = np.hstack(
            [np.tile(features, (len(self.horizons), 1)), remaining_column, horizon_column]
        )
        raw = self.classifier.predict_proba(design)[:, 1].reshape(len(self.horizons), rows)
        out = np.empty((rows, len(self.horizons)))
        for column, horizon in enumerate(self.horizons):
            out[:, column] = self._calibrate(raw[column], horizon)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        """Interpolate the grid CDF onto arbitrary day offsets."""
        xs = np.concatenate([[0.0], np.asarray(self.horizons, dtype=float)])
        out = np.empty((grid_values.shape[0], len(days)))
        for row in range(grid_values.shape[0]):
            ys = np.concatenate([[0.0], grid_values[row]])
            out[row] = np.interp(days, xs, ys)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)
