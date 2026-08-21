"""Gradient-boosted discrete hazard with a censoring-aware AFT tail.

The planning window and the observed tail ask different statistical questions.
Inside 42 days we need fine ranking at a very low event rate, so a boosted
conditional hazard is fitted on the same scenario-cutoff population used by
the evaluator.  Beyond 42 days the data are sparse and censoring dominates; a
penalized Weibull AFT model supplies a smooth continuation instead of asking a
tree ensemble to extrapolate hundreds of nearly empty daily intervals.

The two curves meet continuously at ``TAIL_ANCHOR``::

    F(t) = F_hazard(42) + (1 - F_hazard(42))
           * P_aft(42 < T <= t | T > 42)

This keeps near-term ranking under the boosted model while letting the AFT
tail determine how much probability is genuinely observable before each
device's known observation end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID, TrainingFrame

HAZARD_BINS = (3, 7, 10, 14, 17, 21, 25, 28, 32, 35, 39, 42)
TAIL_ANCHOR = 42

# The AFT model is deliberately small.  Its job is extrapolation and censoring,
# not near-term ranking; feeding all 64 correlated columns to 82 physical events
# makes the tail unstable across buildings.
AFT_FEATURE_NAMES = (
    "voltage",
    "voltage_compensated",
    "staleness",
    "slope_14",
    "slope_30",
    "slope_90",
    "crossing_30",
    "crossing_comp_30",
    "age_days",
    "beta_30",
    "v_std_30",
    "beta_rise",
)
AFT_FEATURE_INDEX = tuple(FEATURE_NAMES.index(name) for name in AFT_FEATURE_NAMES)

DEFAULT_HAZARD_PARAMS = dict(
    max_iter=220,
    learning_rate=0.055,
    max_leaf_nodes=31,
    min_samples_leaf=55,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=20,
    random_state=20260821,
)


def take_frame(frame: TrainingFrame, index: np.ndarray) -> TrainingFrame:
    """Take rows without weakening the typed training-frame contract."""
    index = np.asarray(index)
    return TrainingFrame(
        features=frame.features[index],
        device=frame.device[index],
        building=frame.building[index],
        cutoff=frame.cutoff[index],
        crossing=frame.crossing[index],
        last_observed=frame.last_observed[index],
        observation_end=frame.observation_end[index],
    )


def _device_weights(frame: TrainingFrame) -> np.ndarray:
    """Give every physical battery equal total landmark weight."""
    _, inverse, counts = np.unique(frame.device, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(float)


def event_duration(frame: TrainingFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return duration, event flag, and known follow-up from each cutoff."""
    followup = np.maximum(frame.observation_end - frame.cutoff, 0).astype(float)
    raw = frame.crossing - frame.cutoff
    event = (
        (frame.crossing >= 0)
        & (raw > 0)
        & (frame.crossing <= frame.observation_end)
    )
    duration = np.where(event, raw, followup).astype(float)
    return np.maximum(duration, 0.5), event.astype(np.int8), followup


def build_hazard_table(
    frame: TrainingFrame,
    bins: tuple[int, ...] = HAZARD_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand landmarks into known conditional-hazard intervals.

    An observed EOL contributes one positive interval.  A censored landmark
    contributes negatives only for intervals fully inside its known follow-up.
    Each landmark's weight is divided over its known intervals, after first
    normalizing over repeated cutoffs of the same physical battery.
    """
    duration, event, followup = event_duration(frame)
    # The evaluator samples one active battery per scenario.  The frame is
    # already that target population, so every landmark gets equal total
    # likelihood weight.  Per-device normalization belongs in the AFT fit,
    # where repeated cutoffs would otherwise masquerade as physical events;
    # using it here underweights long-lived devices that genuinely occur in
    # more scored scenarios and measurably damages near-term ranking.
    base_weight = np.ones(len(frame), dtype=float)
    pieces: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    known_count = np.zeros(len(frame), dtype=np.int32)
    start = 0.0
    for end in bins:
        positive = (event == 1) & (duration > start) & (duration <= end)
        negative = (duration > end) & (followup >= end)
        known = positive | negative
        index = np.flatnonzero(known)
        pieces.append((index, positive[index].astype(np.int8), start, float(end)))
        known_count[index] += 1
        start = float(end)

    designs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for index, label, start, end in pieces:
        if index.size == 0:
            continue
        interval = np.column_stack(
            [
                np.full(index.size, start, dtype=np.float32),
                np.full(index.size, end, dtype=np.float32),
                np.full(index.size, end - start, dtype=np.float32),
                np.full(index.size, np.log1p(end), dtype=np.float32),
            ]
        )
        designs.append(np.hstack([frame.features[index], interval]))
        labels.append(label)
        weights.append(base_weight[index] / np.maximum(known_count[index], 1))

    if not designs:
        raise ValueError("no known hazard intervals")
    return (
        np.vstack(designs).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(weights).astype(np.float64),
    )


@dataclass(frozen=True)
class PlattMap:
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        return 1.0 / (1.0 + np.exp(-np.clip(self.slope * logit + self.intercept, -40, 40)))


def fit_platt(probability: np.ndarray, label: np.ndarray) -> PlattMap:
    valid = np.isfinite(probability)
    p = np.asarray(probability, dtype=float)[valid]
    y = np.asarray(label, dtype=np.int8)[valid]
    if p.size < 100 or np.unique(y).size < 2 or y.sum() < 5:
        return PlattMap()
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    x = np.log(clipped / (1.0 - clipped))[:, None]
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
    model.fit(x, y)
    slope = float(model.coef_[0, 0])
    if not np.isfinite(slope) or slope <= 0:
        return PlattMap()
    return PlattMap(slope=slope, intercept=float(model.intercept_[0]))


@dataclass
class AFTTail:
    fitter: WeibullAFTFitter
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, frame: TrainingFrame, *, penalizer: float = 0.5) -> "AFTTail":
        raw = frame.features[:, AFT_FEATURE_INDEX].astype(float)
        medians = np.nanmedian(raw, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(raw), raw, medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        scales = np.where(scales > 1e-8, scales, 1.0)
        transformed = (filled - means) / scales

        duration, event, followup = event_duration(frame)
        usable = followup > 0
        columns = list(AFT_FEATURE_NAMES)
        table = pd.DataFrame(transformed[usable], columns=columns)
        table["duration"] = duration[usable]
        table["event"] = event[usable]
        table["weight"] = _device_weights(frame)[usable]
        fitter = WeibullAFTFitter(penalizer=float(penalizer))
        fitter.fit(
            table,
            duration_col="duration",
            event_col="event",
            weights_col="weight",
            formula=" + ".join(columns),
        )
        return cls(fitter=fitter, medians=medians, means=means, scales=scales)

    def transform(self, features: np.ndarray) -> pd.DataFrame:
        raw = np.asarray(features, dtype=float)[:, AFT_FEATURE_INDEX]
        filled = np.where(np.isfinite(raw), raw, self.medians)
        values = (filled - self.means) / self.scales
        return pd.DataFrame(values, columns=list(AFT_FEATURE_NAMES))

    def cdf(self, features: np.ndarray, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if features.shape[0] == 0:
            return np.zeros((0, times.size))
        safe_times = np.maximum(times, 1e-3)
        survival = self.fitter.predict_survival_function(
            self.transform(features), times=safe_times
        ).to_numpy(dtype=float).T
        return np.maximum.accumulate(np.clip(1.0 - survival, 0.0, 1.0), axis=1)


@dataclass
class HybridHazardAFTModel:
    hazard: HistGradientBoostingClassifier
    ranker: HistGradientBoostingClassifier
    aft: AFTTail
    climatology: np.ndarray
    hazard_all_missing: np.ndarray
    rank_blend: float = 0.65
    calibrators: dict[int, PlattMap] = field(default_factory=dict)
    bins: tuple[int, ...] = HAZARD_BINS
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)
    model_version: str = "bsai-gbdt-hazard-aft/v1"

    @classmethod
    def fit(
        cls,
        frame: TrainingFrame,
        climatology: np.ndarray,
        *,
        hazard_params: dict | None = None,
        aft_penalizer: float = 0.5,
    ) -> "HybridHazardAFTModel":
        settings = dict(DEFAULT_HAZARD_PARAMS)
        settings.update(hazard_params or {})
        design, label, weight = build_hazard_table(frame)
        all_missing = ~np.isfinite(design).any(axis=0)
        if all_missing.any():
            design = design.copy()
            design[:, all_missing] = 0.0
        hazard = HistGradientBoostingClassifier(**settings)
        hazard.fit(design, label, sample_weight=weight)

        # A direct 42-day head supplies the operating-point ranking that the
        # interval likelihood can dilute across twelve mostly-negative bins.
        # It does not define a CDF: it only tilts the hazard curve's terminal
        # odds, while the conditional hazards retain the learned time shape.
        rank_design = frame.features.copy()
        feature_all_missing = all_missing[: frame.features.shape[1]]
        if feature_all_missing.any():
            rank_design[:, feature_all_missing] = 0.0
        rank_label = horizon_labels(frame, horizons=(TAIL_ANCHOR,))[:, 0]
        rank_settings = dict(settings)
        rank_settings["max_iter"] = max(120, int(settings["max_iter"]))
        ranker = HistGradientBoostingClassifier(**rank_settings)
        ranker.fit(rank_design, rank_label)
        aft = AFTTail.fit(frame, penalizer=aft_penalizer)
        return cls(
            hazard=hazard,
            ranker=ranker,
            aft=aft,
            climatology=np.asarray(climatology, dtype=float),
            hazard_all_missing=all_missing,
        )

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def _near_cdf(self, features: np.ndarray) -> np.ndarray:
        rows = features.shape[0]
        hazards: list[np.ndarray] = []
        start = 0.0
        for end in self.bins:
            interval = np.column_stack(
                [
                    np.full(rows, start, dtype=np.float32),
                    np.full(rows, end, dtype=np.float32),
                    np.full(rows, end - start, dtype=np.float32),
                    np.full(rows, np.log1p(end), dtype=np.float32),
                ]
            )
            design = np.hstack([features, interval])
            if self.hazard_all_missing.any():
                design[:, self.hazard_all_missing] = 0.0
            hazards.append(self.hazard.predict_proba(design)[:, 1])
            start = float(end)
        conditional = np.column_stack(hazards)
        raw = 1.0 - np.exp(np.cumsum(np.log1p(-np.clip(conditional, 0.0, 1.0 - 1e-9)), axis=1))

        rank_design = features.copy()
        feature_all_missing = self.hazard_all_missing[: features.shape[1]]
        if feature_all_missing.any():
            rank_design[:, feature_all_missing] = 0.0
        direct = np.clip(self.ranker.predict_proba(rank_design)[:, 1], 1e-6, 1.0 - 1e-6)
        anchor = np.clip(raw[:, -1], 1e-6, 1.0 - 1e-6)
        direct_logit = np.log(direct / (1.0 - direct))
        anchor_logit = np.log(anchor / (1.0 - anchor))
        blended_logit = self.rank_blend * direct_logit + (1.0 - self.rank_blend) * anchor_logit
        blended_anchor = 1.0 / (1.0 + np.exp(-np.clip(blended_logit, -40, 40)))
        raw = raw / np.maximum(raw[:, [-1]], 1e-9) * blended_anchor[:, None]
        raw = np.maximum.accumulate(np.clip(raw, 0.0, 1.0), axis=1)

        calibrated = np.empty_like(raw)
        for column, end in enumerate(self.bins):
            calibrated[:, column] = self.calibrators.get(end, PlattMap()).apply(raw[:, column])
        return np.maximum.accumulate(np.clip(calibrated, 0.0, 1.0), axis=1)

    def _full_cdf(self, features: np.ndarray) -> np.ndarray:
        near = self._near_cdf(features)
        grid = np.asarray(self.horizons, dtype=float)
        aft = self.aft.cdf(features, grid)
        out = np.empty((features.shape[0], grid.size), dtype=float)
        anchor_column = self.bins.index(TAIL_ANCHOR)
        anchor = near[:, anchor_column]
        aft_anchor = aft[:, list(self.horizons).index(TAIL_ANCHOR)]
        for column, horizon in enumerate(self.horizons):
            if horizon <= TAIL_ANCHOR:
                near_column = self.bins.index(horizon)
                out[:, column] = near[:, near_column]
            else:
                conditional = np.clip(
                    (aft[:, column] - aft_anchor) / np.maximum(1.0 - aft_anchor, 1e-8),
                    0.0,
                    1.0,
                )
                out[:, column] = anchor + (1.0 - anchor) * conditional
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        del devices
        features = np.asarray(features, dtype=np.float32)
        if features.shape[0] == 0:
            return np.zeros((0, len(self.horizons)))
        full = self._full_cdf(features)
        xs = np.concatenate([[0.0], np.asarray(self.horizons, dtype=float)])
        out = np.empty_like(full)
        for row in range(features.shape[0]):
            ys = np.concatenate([[0.0], full[row]])
            effective = np.minimum(np.asarray(self.horizons, dtype=float), max(float(remaining[row]), 0.0))
            out[row] = np.interp(effective, xs, ys)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        xs = np.concatenate([[0.0], np.asarray(self.horizons, dtype=float)])
        days = np.asarray(days, dtype=float)
        out = np.empty((grid_values.shape[0], days.size), dtype=float)
        for row in range(grid_values.shape[0]):
            ys = np.concatenate([[0.0], grid_values[row]])
            out[row] = np.interp(days, xs, ys)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)


def horizon_labels(frame: TrainingFrame, horizons: tuple[int, ...] = HAZARD_BINS) -> np.ndarray:
    """Recorded-EOL labels at each effective (censor-capped) horizon."""
    duration, event, followup = event_duration(frame)
    columns = []
    for horizon in horizons:
        effective = np.minimum(float(horizon), followup)
        columns.append((event == 1) & (duration <= effective))
    return np.column_stack(columns).astype(np.int8)


def fit_horizon_calibrators(
    probability: np.ndarray,
    labels: np.ndarray,
    bins: tuple[int, ...] = HAZARD_BINS,
) -> dict[int, PlattMap]:
    return {
        horizon: fit_platt(probability[:, column], labels[:, column])
        for column, horizon in enumerate(bins)
    }


__all__ = [
    "AFTTail",
    "HAZARD_BINS",
    "HybridHazardAFTModel",
    "PlattMap",
    "build_hazard_table",
    "event_duration",
    "fit_horizon_calibrators",
    "horizon_labels",
    "take_frame",
]
