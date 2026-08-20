"""Experimental H1 discrete-time daily hazard challenger for Task 1.

This module is intentionally separate from ``src.risk.model``.  It reuses the
same causal landmark table and forecast contract, but replaces the parametric
AFT likelihood with a gradient-boosted daily conditional hazard model.

An interval ``k`` represents ``(k - 1, k]`` days after a landmark.  Observed
events contribute one positive row in ``ceil(duration_days)``; censored
landmarks contribute negative rows only for fully observed intervals through
``floor(duration_days)``.  Thus an unknown partial interval after censoring is
never silently treated as a negative outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from batteryswap_solution.forecast import CONTRACT_VERSION, ForecastMetadata, RiskForecast

from .cutoffs import assign_building_folds, build_example_table, time_holdout_mask
from .features import (
    build_daily_panels,
    build_feature_series,
    compute_features_asof,
    leave_one_out_building_features,
)
from .model import (
    CALIBRATION_HORIZONS,
    HORIZONS_FOR_METRICS,
    HorizonIsotonicCalibrator,
    derive_design,
    fit_horizon_isotonic_calibrator,
    masked_labels_at_horizon,
    physical_blend,
)


MODEL_VERSION = "task1-discrete-hazard-hgb/v1"
MAX_HAZARD_HORIZON = 365
HAZARD_FEATURES: tuple[str, ...] = (
    "latest_voltage",
    "voltage_slope_28d",
    "frac_low_voltage_28d",
    "age_days",
    "not_yet_deployed",
    "cold_start",
    "horizon_day",
)


def interval_counts(duration_days: np.ndarray, event: np.ndarray, max_horizon: int) -> np.ndarray:
    """Number of fully known/event-containing daily intervals per landmark."""

    duration = np.maximum(np.asarray(duration_days, dtype=float), 0.0)
    observed = np.asarray(event, dtype=int) == 1
    counts = np.where(observed, np.ceil(duration), np.floor(duration)).astype(np.int64)
    return np.clip(counts, 0, int(max_horizon))


def build_hazard_table(
    landmarks: pd.DataFrame,
    *,
    max_horizon: int = MAX_HAZARD_HORIZON,
    weighting: str = "normalized",
) -> pd.DataFrame:
    """Expand causal landmarks into daily at-risk interval rows.

    ``interval`` retains the landmark's existing per-device cutoff weight on
    every interval. ``normalized`` additionally divides by the number of
    intervals, giving every landmark equal total mass and preventing long
    follow-up histories from dominating the fitted objective.
    """

    if weighting not in {"interval", "normalized"}:
        raise ValueError("weighting must be 'interval' or 'normalized'")
    if landmarks.empty:
        return pd.DataFrame(columns=[*HAZARD_FEATURES, "label", "weight", "landmark_index"])

    base = derive_design(landmarks)
    counts = interval_counts(
        landmarks["duration_days"].to_numpy(), landmarks["event"].to_numpy(), max_horizon
    )
    total = int(counts.sum())
    landmark_positions = np.repeat(np.arange(len(landmarks), dtype=np.int64), counts)
    horizon_day = np.concatenate(
        [np.arange(1, count + 1, dtype=np.float32) for count in counts if count > 0]
    ) if total else np.empty(0, dtype=np.float32)

    expanded = base.iloc[landmark_positions][list(HAZARD_FEATURES[:-1])].reset_index(drop=True)
    expanded["horizon_day"] = horizon_day
    labels = np.zeros(total, dtype=np.uint8)
    observed = landmarks["event"].to_numpy(dtype=int) == 1
    event_interval = np.ceil(landmarks["duration_days"].to_numpy(dtype=float)).astype(np.int64)
    positive_landmarks = observed & (event_interval >= 1) & (event_interval <= int(max_horizon))
    if positive_landmarks.any():
        starts = np.cumsum(np.r_[0, counts[:-1]])
        positive_rows = starts[positive_landmarks] + event_interval[positive_landmarks] - 1
        labels[positive_rows] = 1

    landmark_weight = landmarks["sample_weight"].to_numpy(dtype=float)
    weights = landmark_weight[landmark_positions]
    if weighting == "normalized":
        weights = weights / counts[landmark_positions]

    expanded["label"] = labels
    expanded["weight"] = weights.astype(np.float32)
    expanded["landmark_index"] = landmark_positions
    return expanded


def hazards_to_survival(hazards: np.ndarray, axis: int = 0) -> np.ndarray:
    """Convert conditional hazards to survival with stable log products."""

    h = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0 - 1e-12)
    return np.exp(np.cumsum(np.log1p(-h), axis=axis))


def hazards_to_cdf(hazards: np.ndarray, axis: int = 0) -> np.ndarray:
    """Convert conditional hazards to a bounded, monotone cumulative CDF."""

    cdf = 1.0 - hazards_to_survival(hazards, axis=axis)
    return np.clip(np.maximum.accumulate(cdf, axis=axis), 0.0, 1.0)


def _landmark_feature_matrix(table: pd.DataFrame) -> pd.DataFrame:
    return derive_design(table)[list(HAZARD_FEATURES[:-1])].astype(float)


def predict_raw_cdf(
    model: HistGradientBoostingClassifier,
    landmark_features: pd.DataFrame,
    times: Iterable[float],
    *,
    max_horizon: int = MAX_HAZARD_HORIZON,
) -> np.ndarray:
    """Predict raw cumulative failure probabilities at requested day offsets."""

    requested = np.asarray(list(times), dtype=float)
    if requested.size == 0:
        return np.empty((0, len(landmark_features)), dtype=float)
    max_day = min(int(np.ceil(np.maximum(requested, 0.0).max())), int(max_horizon))
    if max_day <= 0:
        return np.zeros((len(requested), len(landmark_features)), dtype=float)

    n_rows = len(landmark_features)
    repeated = pd.DataFrame(
        np.tile(landmark_features.to_numpy(dtype=float), (max_day, 1)),
        columns=list(HAZARD_FEATURES[:-1]),
    )
    repeated["horizon_day"] = np.repeat(np.arange(1, max_day + 1, dtype=float), n_rows)
    hazards = model.predict_proba(repeated[list(HAZARD_FEATURES)])[:, 1].reshape(max_day, n_rows)
    daily_cdf = hazards_to_cdf(hazards, axis=0)

    out = np.zeros((len(requested), n_rows), dtype=float)
    positive = requested > 0
    indices = np.clip(np.ceil(requested[positive]).astype(int), 1, max_day) - 1
    out[positive] = daily_cdf[indices]
    return out


def _fit_model(hazard_table: pd.DataFrame, params: dict, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=seed, **params)
    model.fit(
        hazard_table[list(HAZARD_FEATURES)],
        hazard_table["label"].to_numpy(),
        sample_weight=hazard_table["weight"].to_numpy(),
    )
    return model


def _weighted_metrics(
    table: pd.DataFrame, predicted_by_horizon: dict[int, np.ndarray]
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {
        name: {} for name in (
            "brier", "trivial_brier", "brier_ratio", "log_loss", "calibration_ratio",
            "mean_predicted", "observed_rate", "roc_auc", "top20_event_capture",
        )
    }
    weights = table["sample_weight"].to_numpy(dtype=float)
    for horizon, predicted in predicted_by_horizon.items():
        mask, label = masked_labels_at_horizon(table, horizon)
        valid = mask & np.isfinite(predicted)
        if not valid.any():
            continue
        p = np.clip(predicted[valid], 1e-6, 1.0 - 1e-6)
        y = label[valid]
        w = weights[valid]
        observed_rate = float(np.average(y, weights=w))
        mean_predicted = float(np.average(p, weights=w))
        brier = float(np.average((p - y) ** 2, weights=w))
        trivial = float(np.average((observed_rate - y) ** 2, weights=w))
        result["brier"][horizon] = brier
        result["trivial_brier"][horizon] = trivial
        result["brier_ratio"][horizon] = brier / trivial if trivial > 0 else float("nan")
        result["log_loss"][horizon] = float(
            np.average(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), weights=w)
        )
        result["calibration_ratio"][horizon] = (
            mean_predicted / observed_rate if observed_rate > 0 else float("nan")
        )
        result["mean_predicted"][horizon] = mean_predicted
        result["observed_rate"][horizon] = observed_rate
        result["roc_auc"][horizon] = (
            float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else float("nan")
        )
        event_count = int(y.sum())
        if event_count:
            top_n = min(20, len(p))
            chosen = np.argpartition(p, -top_n)[-top_n:]
            result["top20_event_capture"][horizon] = float(y[chosen].sum() / event_count)
        else:
            result["top20_event_capture"][horizon] = float("nan")
    return result


def run_hazard_cross_validation(
    table: pd.DataFrame,
    params: dict,
    *,
    n_folds: int = 4,
    seed: int = 20260818,
    max_horizon: int = MAX_HAZARD_HORIZON,
    weighting: str = "normalized",
    physical_uncertainty_days: float = 0.0,
) -> dict:
    folds = assign_building_folds(table, n_folds=n_folds, seed=seed).to_numpy()
    horizons = tuple(sorted(h for h in CALIBRATION_HORIZONS if h <= max_horizon))
    oof = {h: np.full(len(table), np.nan, dtype=float) for h in horizons}
    hazard_rows = 0
    event_rows = 0
    folds_used = 0
    features = _landmark_feature_matrix(table)

    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold
        if not train_mask.any() or not test_mask.any():
            continue
        hazard_table = build_hazard_table(
            table.loc[train_mask].reset_index(drop=True),
            max_horizon=max_horizon,
            weighting=weighting,
        )
        if hazard_table.empty or hazard_table["label"].nunique() < 2:
            continue
        hazard_rows = max(hazard_rows, len(hazard_table))
        event_rows = max(event_rows, int(hazard_table["label"].sum()))
        model = _fit_model(hazard_table, params, seed + fold)
        test_positions = np.flatnonzero(test_mask)
        raw = predict_raw_cdf(model, features.iloc[test_positions], horizons, max_horizon=max_horizon)
        if physical_uncertainty_days > 0:
            raw = physical_blend(
                raw,
                np.asarray(horizons, dtype=float),
                table.iloc[test_positions]["crossing_days_extrapolated"].to_numpy(dtype=float),
                physical_uncertainty_days,
            )
        for row, horizon in enumerate(horizons):
            oof[horizon][test_positions] = raw[row]
        folds_used += 1

    calibrator = fit_horizon_isotonic_calibrator(table, oof)
    calibrated = {}
    for horizon, raw in oof.items():
        mapped = np.full_like(raw, np.nan)
        valid = np.isfinite(raw)
        if valid.any():
            mapped[valid] = calibrator.apply(raw[valid][None, :], np.array([horizon]))[0]
        calibrated[horizon] = mapped
    planning_predictions = {h: calibrated[h] for h in HORIZONS_FOR_METRICS if h in calibrated}
    metrics = _weighted_metrics(table, planning_predictions)
    raw_metrics = _weighted_metrics(
        table, {h: oof[h] for h in HORIZONS_FOR_METRICS if h in oof}
    )
    return {
        "oof_predictions": oof,
        "calibrated_predictions": calibrated,
        "calibrator": calibrator,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "mean_brier": float(np.nanmean(list(metrics["brier"].values()))),
        "folds_used": folds_used,
        "max_train_hazard_rows": hazard_rows,
        "max_train_event_rows": event_rows,
    }


@dataclass
class DiscreteHazardForecaster:
    model: HistGradientBoostingClassifier
    calibrator: HorizonIsotonicCalibrator
    params: dict
    weighting: str
    physical_uncertainty_days: float = 0.0
    max_horizon: int = MAX_HAZARD_HORIZON
    model_version: str = MODEL_VERSION
    max_integration_points: int = 220

    def _scenario_table(
        self, battery_data: pd.DataFrame, locations: pd.DataFrame, origin: pd.Timestamp
    ) -> tuple[pd.DataFrame, list[str]]:
        id_column = "battery_id" if "battery_id" in locations else "battery"
        building_column = "building_id" if "building_id" in locations else "building"
        loc = locations.copy()
        loc[id_column] = loc[id_column].astype(str)
        ids = loc[id_column].tolist()
        indexed = loc.set_index(id_column, drop=False)
        panels = build_daily_panels(battery_data)
        rows = []
        for battery_id in ids:
            start = pd.Timestamp(indexed.loc[battery_id, "start_time"])
            if start.tzinfo is not None:
                start = start.tz_localize(None)
            row = {
                "device_id": battery_id,
                "building_id": str(indexed.loc[battery_id, building_column]),
                "cutoff": origin,
                "age_days": float((origin - start.normalize()) / pd.Timedelta("1D")),
            }
            row.update(compute_features_asof(panels.get(battery_id, pd.DataFrame()), origin))
            rows.append(row)
        table = pd.DataFrame(rows)
        table["not_yet_deployed"] = (table["age_days"] < 0).astype(float)
        table["cold_start"] = table["n_readings_total"].fillna(0.0) < 3
        return leave_one_out_building_features(table), ids

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
        features = _landmark_feature_matrix(table)
        c_offset = float((observation_end - origin) / pd.Timedelta("1D"))
        curve_times = np.minimum(np.arange(horizon_days + 1, dtype=float), c_offset)
        span = max(c_offset - horizon_days, 0.0)
        n_extra = int(np.clip(span, 0, self.max_integration_points))
        tail_grid = (
            np.linspace(horizon_days, c_offset, n_extra + 1)
            if n_extra >= 1 else np.array([float(horizon_days), float(horizon_days)])
        )
        all_times = np.unique(np.concatenate([curve_times, tail_grid, [c_offset]]))
        raw = predict_raw_cdf(self.model, features, all_times, max_horizon=self.max_horizon)
        if self.physical_uncertainty_days > 0:
            raw = physical_blend(
                raw, all_times, table["crossing_days_extrapolated"].to_numpy(dtype=float),
                self.physical_uncertainty_days,
            )
        mapped = self.calibrator.apply(raw, np.minimum(all_times, self.max_horizon))
        calibrated = np.clip(np.maximum.accumulate(mapped, axis=0), 0.0, 1.0)
        calibrated[all_times <= 0] = 0.0
        time_index = {round(float(t), 8): i for i, t in enumerate(all_times)}
        rows_for = lambda values: np.array([time_index[round(float(v), 8)] for v in values])
        curve_cdf = calibrated[rows_for(curve_times)]
        cdf_h = calibrated[rows_for(np.array([curve_times[-1]]))[0]]
        cdf_c = calibrated[rows_for(np.array([c_offset]))[0]]
        observed_tail = np.clip(cdf_c - cdf_h, 0.0, 1.0)
        unobserved = np.clip(1.0 - cdf_c, 0.0, 1.0)

        survival_tail = 1.0 - calibrated[rows_for(tail_grid)]
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        integral = trapezoid(survival_tail, x=tail_grid, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_excess = (integral - span * survival_tail[-1]) / observed_tail
        mean_excess = np.where(observed_tail > 1e-9, mean_excess, 0.0)
        mean_excess = np.clip(np.nan_to_num(mean_excess), 0.0, None)

        n_batteries = len(battery_ids)
        curves = pd.DataFrame({
            "battery_id": np.repeat(np.asarray(battery_ids, dtype=object), len(dates)),
            "forecast_date": np.tile(dates.to_numpy(), n_batteries),
            "failure_cdf": curve_cdf.T.reshape(-1),
        })
        tail = pd.DataFrame({
            "battery_id": battery_ids,
            "prob_observed_after_horizon": observed_tail,
            "mean_excess_rul_days_given_observed_after_horizon": mean_excess,
            "prob_unobserved_eol": unobserved,
            "prob_no_observed_eol_by_horizon": observed_tail + unobserved,
        })
        reached = curve_cdf >= 0.5
        q50 = np.where(reached.any(axis=0), np.argmax(reached, axis=0).astype(float), np.nan)
        quality = np.clip(
            0.5 * table["completeness_90d"].fillna(0.0).to_numpy()
            + 0.5 * np.clip(table["n_readings_total"].fillna(0.0).to_numpy() / 30.0, 0.0, 1.0),
            0.0, 1.0,
        )
        summaries = pd.DataFrame({
            "battery_id": battery_ids,
            "q50_days": q50,
            "data_quality": quality,
            "cold_start": table["cold_start"].to_numpy(),
        })
        metadata = ForecastMetadata(
            contract_version=CONTRACT_VERSION,
            model_version=self.model_version,
            prediction_origin=origin,
            forecast_end_date=dates[-1],
            horizon_days=horizon_days,
            evaluation_observation_end=observation_end,
        )
        return RiskForecast(metadata, curves, tail, summaries)


def select_and_fit_hazard(
    table: pd.DataFrame,
    *,
    parameter_grid: Iterable[dict],
    n_folds: int = 4,
    seed: int = 20260818,
    max_horizon: int = MAX_HAZARD_HORIZON,
    weighting: str = "normalized",
    physical_uncertainty_days: float = 0.0,
) -> tuple[DiscreteHazardForecaster, dict]:
    results = []
    for params in parameter_grid:
        cv = run_hazard_cross_validation(
            table, params, n_folds=n_folds, seed=seed, max_horizon=max_horizon,
            weighting=weighting, physical_uncertainty_days=physical_uncertainty_days,
        )
        results.append((dict(params), cv))
    if not results:
        raise RuntimeError("No discrete-hazard configuration was evaluated")
    best_params, best_cv = min(results, key=lambda item: item[1]["mean_brier"])
    hazard_table = build_hazard_table(table, max_horizon=max_horizon, weighting=weighting)
    final_model = _fit_model(hazard_table, best_params, seed)
    forecaster = DiscreteHazardForecaster(
        model=final_model,
        calibrator=best_cv["calibrator"],
        params=best_params,
        weighting=weighting,
        physical_uncertainty_days=physical_uncertainty_days,
        max_horizon=max_horizon,
    )
    unique_events = int(table.loc[table["event"] == 1, "device_id"].nunique())
    report = {
        "selected_params": best_params,
        "weighting": weighting,
        "physical_uncertainty_days": physical_uncertainty_days,
        "n_folds_used": best_cv["folds_used"],
        "oof_metrics": best_cv["metrics"],
        "raw_oof_metrics": best_cv["raw_metrics"],
        "mean_oof_brier": best_cv["mean_brier"],
        "n_unique_batteries": int(table["device_id"].nunique()),
        "n_unique_eol_events": unique_events,
        "n_landmarks": int(len(table)),
        "n_hazard_rows": int(len(hazard_table)),
        "n_event_rows": int(hazard_table["label"].sum()),
        "censoring_rate": float(1.0 - unique_events / table["device_id"].nunique()),
        "grid_search": [
            {"params": params, "mean_oof_brier": cv["mean_brier"]} for params, cv in results
        ],
    }
    latest_mask = time_holdout_mask(table).to_numpy()
    if latest_mask.any():
        latest_table = table.loc[latest_mask].reset_index(drop=True)
        latest_predictions = {
            horizon: values[latest_mask]
            for horizon, values in best_cv["calibrated_predictions"].items()
            if horizon in HORIZONS_FOR_METRICS
        }
        report["latest_time_oof_subset"] = {
            "note": (
                "Latest 20% of landmarks evaluated with building-grouped OOF predictions; "
                "validation buildings remain unseen, but this is a drift subset rather than "
                "a strict train-past/test-future split."
            ),
            "n_landmarks": int(latest_mask.sum()),
            "metrics": _weighted_metrics(latest_table, latest_predictions),
        }
    return forecaster, report


def fit_discrete_hazard_forecaster(
    locations: pd.DataFrame,
    eol_times: pd.Series,
    timeseries: pd.DataFrame,
    cutoff_dates: pd.DatetimeIndex,
    **kwargs,
) -> tuple[DiscreteHazardForecaster, dict]:
    feature_series = build_feature_series(timeseries)
    table = build_example_table(locations, eol_times, feature_series, cutoff_dates)
    if table.empty:
        raise RuntimeError("No causal landmarks were constructed")
    return select_and_fit_hazard(table, **kwargs)


__all__ = [
    "DiscreteHazardForecaster", "HAZARD_FEATURES", "MAX_HAZARD_HORIZON",
    "build_hazard_table", "fit_discrete_hazard_forecaster", "hazards_to_cdf",
    "hazards_to_survival", "interval_counts", "predict_raw_cdf", "select_and_fit_hazard",
]
