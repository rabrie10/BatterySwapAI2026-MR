"""Task 1 survival model: causal AFT hazard model, calibration, and the v1 contract producer.

Model family: a single censoring-aware parametric AFT model (Weibull /
LogNormal / LogLoglogistic, selected by out-of-fold Brier score) over a small,
curated covariate set. AFT is chosen over a discrete-time hazard classifier
because it extrapolates smoothly and monotonically to arbitrary horizons
(needed for the observed-tail / unobserved-EOL split, which can require
surviving several hundred days past the 42-day planning horizon for early
scenarios) without any special-casing, and because the dataset has only 82
observed physical events — far too few to safely support the ~100+ covariates
a fully-featured discrete model would want. The curated feature set below (6
covariates) is a deliberately conservative choice given that event count —
even smaller than an earlier 14-covariate version, after integration testing
against the real Task 2 planner showed the extra covariates were diluting
signal without buying identification (see docs/TASK1_IMPLEMENTATION.md Sec
5.2). The AFT curve is also not used alone: `Task1Forecaster.predict()`
blends it with a sharp deterministic physical-extrapolation prior (Sec 4.5 of
that document) that was needed to fix under-prediction of near-term risk
for a specific, common failure pattern.

Everything Task 1 needs to reproduce evaluator-aligned outcome probabilities
from one calibrated conditional survival curve is:

    failure_cdf(d)               = G(min(d, C))
    prob_observed_after_horizon  = max(G(C) - G(horizon_end), 0)
    prob_unobserved_eol          = max(1 - G(C), 0)

where G is the calibrated conditional CDF and C is evaluation_observation_end
(both measured as day offsets from prediction_origin). These three sum to one
by construction for any monotone G, which is why `predict()` below evaluates
G at capped time arguments rather than post-hoc patching the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression

from batteryswap_solution.forecast import CONTRACT_VERSION, ForecastMetadata, RiskForecast

from .cutoffs import assign_building_folds, build_example_table
from .features import (
    build_daily_panels,
    build_feature_series,
    compute_features_asof,
    leave_one_out_building_features,
)

MODEL_VERSION = "task1-aft-survival-blended/v1"

AFT_FAMILIES = {
    "weibull": WeibullAFTFitter,
    "lognormal": LogNormalAFTFitter,
    "loglogistic": LogLogisticAFTFitter,
}
PENALIZER_GRID: tuple[float, ...] = (0.1, 0.5)
HORIZONS_FOR_METRICS: tuple[int, ...] = (7, 14, 21, 28, 35, 42)

CURATED_FEATURES: tuple[str, ...] = (
    "latest_voltage",
    "voltage_slope_28d",
    "frac_low_voltage_28d",
    "age_days",
    "not_yet_deployed",
    "cold_start",
)

MIN_DURATION_DAYS = 0.5


def derive_design(table: pd.DataFrame) -> pd.DataFrame:
    """Curated, transformed covariate frame (unscaled, NaNs kept for imputation)."""

    out = pd.DataFrame(index=table.index)
    out["latest_voltage"] = table["latest_voltage"]
    out["voltage_slope_28d"] = table["voltage_slope_28d"]
    out["voltage_slope_90d"] = table["voltage_slope_90d"]
    out["voltage_std_28d"] = table["voltage_std_28d"]
    out["crossing_days_log"] = np.log1p(table["crossing_days_extrapolated"])
    out["temp_mean_28d"] = table["temp_mean_28d"]
    out["frac_low_voltage_28d"] = table["frac_low_voltage_28d"]
    out["completeness_90d"] = table["completeness_90d"].fillna(0.0)
    out["age_days"] = table["age_days"]
    out["days_since_last_reading"] = table["days_since_last_reading"]
    out["building_loo_latest_voltage"] = table["building_loo_latest_voltage"]
    out["building_loo_voltage_slope_28d"] = table["building_loo_voltage_slope_28d"]
    out["not_yet_deployed"] = table["not_yet_deployed"].astype(float)
    out["cold_start"] = table["cold_start"].astype(float)
    return out


@dataclass(frozen=True)
class FeatureTransform:
    """Median-impute + z-score, with statistics frozen at fit time (train-fold only)."""

    columns: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    def transform(self, design: pd.DataFrame) -> pd.DataFrame:
        subset = design.reindex(columns=list(self.columns)).astype(float)
        filled = subset.fillna(pd.Series(self.medians))
        scaled = (filled - pd.Series(self.means)) / pd.Series(self.stds)
        return scaled


def fit_feature_transform(design: pd.DataFrame, columns: Iterable[str]) -> FeatureTransform:
    columns = list(columns)
    subset = design.reindex(columns=columns).astype(float)
    medians = {c: (float(subset[c].median()) if subset[c].notna().any() else 0.0) for c in columns}
    filled = subset.fillna(pd.Series(medians))
    means = filled.mean().to_dict()
    stds = {c: (v if v > 1e-9 else 1.0) for c, v in filled.std(ddof=0).to_dict().items()}
    return FeatureTransform(columns=tuple(columns), medians=medians, means=means, stds=stds)


@dataclass(frozen=True)
class PlattCalibrator:
    """1-D logistic recalibration of raw CDF values; monotone by construction (slope > 0)."""

    slope: float
    intercept: float

    def apply(self, raw_cdf: np.ndarray) -> np.ndarray:
        eps = 1e-6
        clipped = np.clip(raw_cdf, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + np.exp(-(self.slope * logit + self.intercept)))


def fit_platt_calibrator(raw_cdf: np.ndarray, label: np.ndarray, weight: np.ndarray) -> PlattCalibrator:
    if len(raw_cdf) == 0:
        return PlattCalibrator(slope=1.0, intercept=0.0)
    eps = 1e-6
    clipped = np.clip(raw_cdf, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if len(np.unique(label)) < 2:
        return PlattCalibrator(slope=1.0, intercept=0.0)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, label, sample_weight=weight)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if not np.isfinite(slope) or slope <= 0:
        return PlattCalibrator(slope=1.0, intercept=0.0)
    return PlattCalibrator(slope=slope, intercept=intercept)


def _build_lifelines_frame(scaled_design: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    frame = scaled_design.copy()
    frame["duration_days"] = table["duration_days"].to_numpy()
    frame["duration_days"] = frame["duration_days"].clip(lower=MIN_DURATION_DAYS)
    frame["event"] = table["event"].to_numpy()
    frame["sample_weight"] = table["sample_weight"].to_numpy()
    return frame


def fit_aft(family: str, frame: pd.DataFrame, penalizer: float):
    model = AFT_FAMILIES[family](penalizer=penalizer)
    model.fit(
        frame,
        duration_col="duration_days",
        event_col="event",
        weights_col="sample_weight",
        robust=True,
        show_progress=False,
    )
    return model


def masked_labels_at_horizon(table: pd.DataFrame, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact masked multi-horizon label rule (spec Sec 5.1): mask ambiguous censoring."""

    duration = table["duration_days"].to_numpy()
    event = table["event"].to_numpy()
    observed_within = (event == 1) & (duration <= horizon_days)
    known_alive = duration >= horizon_days
    mask = observed_within | known_alive
    label = observed_within.astype(float)
    return mask, label


def evaluate_predictions(table: pd.DataFrame, predicted_cdf_by_horizon: dict[int, np.ndarray]) -> dict:
    weight = table["sample_weight"].to_numpy()
    brier: dict[int, float] = {}
    log_loss: dict[int, float] = {}
    for horizon, predicted in predicted_cdf_by_horizon.items():
        mask, label = masked_labels_at_horizon(table, horizon)
        valid = mask & np.isfinite(predicted)
        if valid.sum() == 0:
            continue
        p = np.clip(predicted[valid], 1e-6, 1.0 - 1e-6)
        y = label[valid]
        w = weight[valid]
        brier[horizon] = float(np.average((p - y) ** 2, weights=w))
        log_loss[horizon] = float(np.average(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), weights=w))
    return {"brier": brier, "log_loss": log_loss}


def run_cross_validation(
    table: pd.DataFrame,
    design: pd.DataFrame,
    family: str,
    penalizer: float,
    n_folds: int = 5,
    seed: int = 20260818,
) -> dict:
    """Causal grouped (unseen-building) out-of-fold CV for one model configuration."""

    folds = assign_building_folds(table, n_folds=n_folds, seed=seed).to_numpy()
    oof_predictions = {h: np.full(len(table), np.nan) for h in HORIZONS_FOR_METRICS}
    fold_concordances: list[float] = []

    for fold in range(n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0 or table.loc[train_mask, "event"].sum() == 0:
            continue

        transform = fit_feature_transform(design.loc[train_mask], CURATED_FEATURES)
        train_scaled = transform.transform(design.loc[train_mask])
        test_scaled = transform.transform(design.loc[test_mask])
        frame = _build_lifelines_frame(train_scaled, table.loc[train_mask])
        model = fit_aft(family, frame, penalizer)

        times = np.array(sorted(HORIZONS_FOR_METRICS), dtype=float)
        survival = model.predict_survival_function(test_scaled, times=times)
        raw_cdf = 1.0 - survival.to_numpy()
        test_positions = np.where(test_mask)[0]
        for row_index, horizon in enumerate(sorted(HORIZONS_FOR_METRICS)):
            oof_predictions[horizon][test_positions] = raw_cdf[row_index, :]

        median_survival = model.predict_median(test_scaled).to_numpy()
        median_survival = np.nan_to_num(median_survival, nan=1e6, posinf=1e6, neginf=0.0)
        fold_concordances.append(
            concordance_index(
                table.loc[test_mask, "duration_days"].to_numpy(),
                median_survival,
                table.loc[test_mask, "event"].to_numpy(),
            )
        )

    metrics = evaluate_predictions(table, oof_predictions)
    concordance = float(np.mean(fold_concordances)) if fold_concordances else float("nan")
    return {
        "concordance": concordance,
        "brier": metrics["brier"],
        "log_loss": metrics["log_loss"],
        "oof_predictions": oof_predictions,
        "n_folds_used": len(fold_concordances),
    }


def build_calibration_pool(
    table: pd.DataFrame, oof_predictions: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = table["sample_weight"].to_numpy()
    raws, labels, weights = [], [], []
    for horizon, predicted in oof_predictions.items():
        mask, label = masked_labels_at_horizon(table, horizon)
        valid = mask & np.isfinite(predicted)
        if valid.sum() == 0:
            continue
        raws.append(predicted[valid])
        labels.append(label[valid])
        weights.append(weight[valid])
    if not raws:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(raws), np.concatenate(labels), np.concatenate(weights)


def _conditional_logistic_cdf(days: np.ndarray, location: np.ndarray, scale: float) -> np.ndarray:
    """P(T <= days | T > 0) under a logistic location-scale model, vectorized.

    Same functional form as ``forecast.VoltageTrendForecaster``'s physical
    extrapolation. Used as a sharp, low-variance safety floor on top of the
    AFT model's calibrated curve (see ``Task1Forecaster.predict`` docstring
    for why): with only 82 physical events, the AFT's covariate coefficients
    are necessarily heavily shrunk (Sec 1/4 of docs/TASK1_IMPLEMENTATION.md),
    which under-predicts near-term risk specifically for batteries whose
    voltage has plateaued just above the EOL threshold rather than declining
    smoothly — exactly the physical situation ``crossing_days_extrapolated``
    is designed to flag directly.
    """

    raw = 1.0 / (1.0 + np.exp(-np.clip((days - location) / scale, -40.0, 40.0)))
    at_origin = 1.0 / (1.0 + np.exp(-np.clip((-location) / scale, -40.0, 40.0)))
    conditioned = (raw - at_origin) / np.maximum(1.0 - at_origin, 1e-9)
    return np.clip(conditioned, 0.0, 1.0)


@dataclass
class Task1Forecaster:
    """Fitted Task 1 artifact implementing ``RiskForecaster.predict()`` (contract v1).

    The final curve is ``max(calibrated AFT CDF, physical crossing-day CDF)``
    at every evaluated time, not the AFT curve alone. This blend is a
    deliberate correction, not an afterthought: with only 82 physical EOL
    events in the whole dataset, the AFT model's per-covariate coefficients
    are necessarily heavily shrunk toward the population baseline (every
    covariate has p > 0.25 even alone — see docs/TASK1_IMPLEMENTATION.md Sec
    1/4), which was measured (via the real Task 2 planner, not just
    Brier/concordance) to under-predict near-term risk for batteries whose
    voltage has plateaued just above the 2.4V threshold rather than declined
    smoothly — precisely because a plateau produces a near-zero trailing
    slope and therefore a large, physically-wrong crossing-day extrapolation
    if left to the AFT's weak covariate fit alone. Taking the pointwise
    maximum with the sharp, low-variance physical extrapolation
    (``crossing_days_extrapolated``, the same estimator this module derives
    for the AFT covariate, evaluated as a direct logistic location) recovers
    the sharpness the AFT cannot supply from 82 events, while the AFT model
    still supplies calibration, the observed/unobserved tail split, and
    monotonicity everywhere the physical estimate is less informative (e.g.
    genuinely cold-start batteries, where crossing_days_extrapolated falls
    back to the population median and stops dominating).
    """

    model_family: str
    penalizer: float
    aft_model: object
    transform: FeatureTransform
    calibrator: PlattCalibrator
    model_version: str = MODEL_VERSION
    max_integration_points: int = 220
    physical_uncertainty_days: float = 20.0

    def _scenario_table(
        self, battery_data: pd.DataFrame, locations: pd.DataFrame, origin: pd.Timestamp
    ) -> tuple[pd.DataFrame, list[str]]:
        id_column = "battery_id" if "battery_id" in locations else "battery"
        building_column = "building_id" if "building_id" in locations else "building"
        loc = locations.copy()
        loc[id_column] = loc[id_column].astype(str)
        battery_ids = loc[id_column].tolist()
        loc_indexed = loc.set_index(id_column, drop=False)

        # Inference needs exactly one row per battery ("as of" this scenario's
        # origin), not the full per-date rolling series build_feature_series
        # computes for training's many-cutoff-per-device reuse. Using the
        # single-date path here cut per-scenario feature computation from
        # ~40s to well under a second (see docs/TASK1_IMPLEMENTATION.md Sec 7).
        panels = build_daily_panels(battery_data)
        rows = []
        for battery_id in battery_ids:
            panel = panels.get(battery_id, pd.DataFrame())
            start_time = pd.Timestamp(loc_indexed.loc[battery_id, "start_time"])
            if start_time.tzinfo is not None:
                start_time = start_time.tz_localize(None)
            features = compute_features_asof(panel, origin)
            row = {
                "device_id": battery_id,
                "building_id": str(loc_indexed.loc[battery_id, building_column]),
                "cutoff": origin,
                "age_days": float((origin - start_time.normalize()) / pd.Timedelta(days=1)),
            }
            row.update(features)
            rows.append(row)

        table = pd.DataFrame(rows)
        table["not_yet_deployed"] = (table["age_days"] < 0).astype(float)
        table["cold_start"] = table["n_readings_total"].fillna(0.0) < 3
        table = leave_one_out_building_features(table)
        return table, battery_ids

    def predict(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        *,
        prediction_origin: pd.Timestamp,
        horizon_days: int,
        evaluation_observation_end: pd.Timestamp,
    ) -> RiskForecast:
        origin = pd.Timestamp(prediction_origin)
        if origin.tzinfo is not None:
            origin = origin.tz_localize(None)
        origin = origin.normalize()

        observation_end = pd.Timestamp(evaluation_observation_end)
        if observation_end.tzinfo is not None:
            observation_end = observation_end.tz_localize(None)
        observation_end = max(observation_end.normalize(), origin)

        horizon_days = int(horizon_days)
        dates = pd.date_range(origin, origin + pd.Timedelta(days=horizon_days), freq="D")

        table, battery_ids = self._scenario_table(battery_data, locations, origin)
        design = derive_design(table)
        scaled = self.transform.transform(design)
        n_batteries = len(battery_ids)

        c_offset = float((observation_end - origin) / pd.Timedelta(days=1))
        day_offsets = np.arange(horizon_days + 1, dtype=float)
        eval_offsets_curve = np.clip(np.minimum(day_offsets, c_offset), MIN_DURATION_DAYS, None)

        span = max(c_offset - horizon_days, 0.0)
        n_extra = int(np.clip(span, 0, self.max_integration_points))
        if n_extra >= 1:
            tail_grid = np.linspace(horizon_days, c_offset, n_extra + 1)
        else:
            tail_grid = np.array([horizon_days, horizon_days])
        tail_grid = np.clip(tail_grid, MIN_DURATION_DAYS, None)
        c_point = np.clip(np.array([c_offset]), MIN_DURATION_DAYS, None)

        all_times = np.unique(np.concatenate([eval_offsets_curve, tail_grid, c_point]))
        survival = self.aft_model.predict_survival_function(scaled, times=all_times)
        raw_cdf_matrix = 1.0 - survival.to_numpy()
        calibrated_matrix = self.calibrator.apply(raw_cdf_matrix)

        # Cold-start batteries (no crossing-day estimate available) get a huge
        # placeholder distance so the physical term contributes ~0 and the
        # AFT+calibration curve alone determines their forecast, which is the
        # right behavior when there is no voltage history to extrapolate from.
        crossing_days = table["crossing_days_extrapolated"].fillna(1.0e4).to_numpy(dtype=float)
        physical_cdf_matrix = _conditional_logistic_cdf(
            all_times[:, None], crossing_days[None, :], self.physical_uncertainty_days
        )
        combined_matrix = np.maximum(calibrated_matrix, physical_cdf_matrix)
        calibrated_matrix = np.maximum.accumulate(combined_matrix, axis=0)

        time_index = {round(float(t), 6): i for i, t in enumerate(all_times)}

        def rows_for(times_array: np.ndarray) -> np.ndarray:
            return np.array([time_index[round(float(t), 6)] for t in times_array])

        curve_rows_idx = rows_for(eval_offsets_curve)
        curve_cdf = calibrated_matrix[curve_rows_idx, :]  # (n_dates, n_batteries)

        horizon_row = rows_for(np.array([eval_offsets_curve[-1]]))[0]
        c_row = rows_for(c_point)[0]
        cdf_at_horizon = calibrated_matrix[horizon_row, :]
        cdf_at_c = calibrated_matrix[c_row, :]

        prob_observed_after_horizon = np.clip(cdf_at_c - cdf_at_horizon, 0.0, 1.0)
        prob_unobserved_eol = np.clip(1.0 - cdf_at_c, 0.0, 1.0)

        tail_positions = rows_for(tail_grid)
        survival_tail = 1.0 - calibrated_matrix[tail_positions, :]
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        integral = trapezoid(survival_tail, x=tail_grid, axis=0)
        survival_at_c = survival_tail[-1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_excess = (integral - span * survival_at_c) / prob_observed_after_horizon
        mean_excess = np.nan_to_num(mean_excess, nan=0.0, posinf=0.0, neginf=0.0)
        mean_excess = np.where(prob_observed_after_horizon > 1e-9, mean_excess, 0.0)
        mean_excess = np.clip(mean_excess, 0.0, None)

        curve_rows = pd.DataFrame(
            {
                "battery_id": np.repeat(np.asarray(battery_ids, dtype=object), len(dates)),
                "forecast_date": np.tile(dates.to_numpy(), n_batteries),
                "failure_cdf": curve_cdf.T.reshape(-1),
            }
        )
        tail_rows = pd.DataFrame(
            {
                "battery_id": battery_ids,
                "prob_observed_after_horizon": prob_observed_after_horizon,
                "mean_excess_rul_days_given_observed_after_horizon": mean_excess,
                "prob_unobserved_eol": prob_unobserved_eol,
                "prob_no_observed_eol_by_horizon": prob_observed_after_horizon + prob_unobserved_eol,
            }
        )

        reached_50 = curve_cdf >= 0.5
        first_reach = np.argmax(reached_50, axis=0).astype(float)
        never_reached = ~reached_50.any(axis=0)
        q50_days = np.where(never_reached, np.nan, first_reach)
        data_quality = np.clip(
            0.5 * table["completeness_90d"].fillna(0.0).to_numpy()
            + 0.5 * np.clip(table["n_readings_total"].fillna(0.0).to_numpy() / 30.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        summaries = pd.DataFrame(
            {
                "battery_id": battery_ids,
                "q50_days": q50_days,
                "data_quality": data_quality,
                "cold_start": table["cold_start"].to_numpy(),
            }
        )

        metadata = ForecastMetadata(
            contract_version=CONTRACT_VERSION,
            model_version=self.model_version,
            prediction_origin=origin,
            forecast_end_date=dates[-1],
            horizon_days=horizon_days,
            evaluation_observation_end=observation_end,
        )
        return RiskForecast(metadata, curve_rows, tail_rows, summaries)


def select_and_fit(
    table: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 20260818,
    families: Iterable[str] = tuple(AFT_FAMILIES),
    penalizers: Iterable[float] = PENALIZER_GRID,
) -> tuple[Task1Forecaster, dict]:
    """Model-family/penalizer selection by causal grouped OOF Brier, then final refit."""

    design = derive_design(table)
    grid_results: dict[tuple[str, float], dict] = {}
    for family in families:
        for penalizer in penalizers:
            try:
                cv = run_cross_validation(table, design, family, penalizer, n_folds=n_folds, seed=seed)
            except Exception:  # noqa: BLE001 - some configurations may fail to converge
                continue
            if not cv["brier"]:
                continue
            cv["mean_brier"] = float(np.mean(list(cv["brier"].values())))
            grid_results[(family, penalizer)] = cv

    if not grid_results:
        raise RuntimeError("No AFT family/penalizer configuration converged during model selection")

    best_key = min(grid_results, key=lambda key: grid_results[key]["mean_brier"])
    best_family, best_penalizer = best_key
    best_cv = grid_results[best_key]

    raw_pool, label_pool, weight_pool = build_calibration_pool(table, best_cv["oof_predictions"])
    calibrator = fit_platt_calibrator(raw_pool, label_pool, weight_pool)

    final_transform = fit_feature_transform(design, CURATED_FEATURES)
    final_scaled = final_transform.transform(design)
    final_frame = _build_lifelines_frame(final_scaled, table)
    final_model = fit_aft(best_family, final_frame, best_penalizer)

    forecaster = Task1Forecaster(
        model_family=best_family,
        penalizer=best_penalizer,
        aft_model=final_model,
        transform=final_transform,
        calibrator=calibrator,
    )
    report = {
        "selected_family": best_family,
        "selected_penalizer": best_penalizer,
        "cv_n_folds_used": best_cv["n_folds_used"],
        "cv_concordance": best_cv["concordance"],
        "cv_brier_by_horizon": best_cv["brier"],
        "cv_log_loss_by_horizon": best_cv["log_loss"],
        "n_examples": int(len(table)),
        "n_devices": int(table["device_id"].nunique()),
        "n_events": int(table["event"].sum()),
        "calibrator_slope": calibrator.slope,
        "calibrator_intercept": calibrator.intercept,
        "grid_search": {
            f"{family}/{penalizer}": {
                "mean_brier": result["mean_brier"],
                "concordance": result["concordance"],
            }
            for (family, penalizer), result in grid_results.items()
        },
    }
    return forecaster, report


def fit_task1_forecaster(
    locations: pd.DataFrame,
    eol_times: pd.Series,
    timeseries: pd.DataFrame,
    cutoff_dates: pd.DatetimeIndex,
    n_folds: int = 5,
    seed: int = 20260818,
    families: Iterable[str] = tuple(AFT_FAMILIES),
    penalizers: Iterable[float] = PENALIZER_GRID,
    physical_uncertainty_days: float = 20.0,
) -> tuple[Task1Forecaster, dict]:
    """End-to-end: causal features -> example table -> model selection -> calibration."""

    feature_series = build_feature_series(timeseries)
    table = build_example_table(locations, eol_times, feature_series, cutoff_dates)
    if table.empty:
        raise RuntimeError("No causal (device, cutoff) examples were constructed")
    forecaster, report = select_and_fit(
        table, n_folds=n_folds, seed=seed, families=families, penalizers=penalizers
    )
    if physical_uncertainty_days != forecaster.physical_uncertainty_days:
        forecaster = replace(forecaster, physical_uncertainty_days=physical_uncertainty_days)
    report["table_rows"] = int(len(table))
    return forecaster, report
