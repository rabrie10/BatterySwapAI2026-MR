"""Forecast contract consumed by the Task 2 optimizer.

The planner only depends on calibrated daily event probabilities. Task 1 can
therefore replace the fallback forecaster without touching any scheduling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


CONTRACT_VERSION = "batteryswap-risk-forecast/v1"


@dataclass(frozen=True)
class ForecastMetadata:
    contract_version: str
    model_version: str
    prediction_origin: pd.Timestamp
    forecast_end_date: pd.Timestamp
    horizon_days: int
    evaluation_observation_end: pd.Timestamp


@dataclass(frozen=True)
class RiskForecast:
    metadata: ForecastMetadata
    curves: pd.DataFrame
    tail: pd.DataFrame
    summaries: pd.DataFrame


class RiskForecaster(Protocol):
    def predict(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        *,
        prediction_origin: pd.Timestamp,
        horizon_days: int,
        evaluation_observation_end: pd.Timestamp,
    ) -> RiskForecast:
        ...


class ForecastContractError(ValueError):
    """Raised when Task 1 returns a forecast that Task 2 cannot safely use."""


def _normal_date(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


def validate_forecast(
    forecast: RiskForecast,
    battery_ids: list[str],
    candidate_dates: pd.DatetimeIndex,
    *,
    tolerance: float = 1e-6,
) -> RiskForecast:
    """Validate and canonically order a v1 forecast.

    Tiny floating point monotonicity errors are repaired with a cumulative
    maximum. Structural or semantic errors fail loudly so the planner can use
    its known-valid fallback forecast.
    """

    metadata = forecast.metadata
    if metadata.contract_version != CONTRACT_VERSION:
        raise ForecastContractError(
            f"Unsupported forecast contract: {metadata.contract_version!r}"
        )

    expected_ids = pd.Index([str(value) for value in battery_ids], name="battery_id")
    expected_dates = pd.DatetimeIndex([_normal_date(value) for value in candidate_dates])
    curves = forecast.curves.copy()
    required_curve_columns = ["battery_id", "forecast_date", "failure_cdf"]
    missing = set(required_curve_columns) - set(curves.columns)
    if missing:
        raise ForecastContractError(f"Forecast curves missing columns: {sorted(missing)}")

    curves = curves[required_curve_columns].copy()
    curves["battery_id"] = curves["battery_id"].astype(str)
    curves["forecast_date"] = pd.to_datetime(curves["forecast_date"]).map(_normal_date)
    if curves.duplicated(["battery_id", "forecast_date"]).any():
        raise ForecastContractError("Duplicate battery/date keys in forecast curves")
    if set(curves["battery_id"]) != set(expected_ids):
        raise ForecastContractError("Forecast battery IDs do not match active locations")

    pivot = curves.pivot(index="battery_id", columns="forecast_date", values="failure_cdf")
    pivot = pivot.reindex(index=expected_ids, columns=expected_dates)
    values = pivot.to_numpy(dtype=float)
    if values.shape != (len(expected_ids), len(expected_dates)) or not np.isfinite(values).all():
        raise ForecastContractError("Forecast curves are incomplete or non-finite")
    if ((values < -tolerance) | (values > 1.0 + tolerance)).any():
        raise ForecastContractError("failure_cdf must be bounded by [0, 1]")
    values = np.maximum.accumulate(np.clip(values, 0.0, 1.0), axis=1)

    tail = forecast.tail.copy()
    required_tail_columns = {
        "battery_id",
        "prob_observed_after_horizon",
        "mean_excess_rul_days_given_observed_after_horizon",
        "prob_unobserved_eol",
        "prob_no_observed_eol_by_horizon",
    }
    missing = required_tail_columns - set(tail.columns)
    if missing:
        raise ForecastContractError(f"Forecast tail missing columns: {sorted(missing)}")
    tail = tail[list(required_tail_columns)].copy()
    tail["battery_id"] = tail["battery_id"].astype(str)
    if tail["battery_id"].duplicated().any() or set(tail["battery_id"]) != set(expected_ids):
        raise ForecastContractError("Forecast tail must contain each active battery exactly once")
    tail = tail.set_index("battery_id").reindex(expected_ids)
    tail_values = tail.to_numpy(dtype=float)
    if not np.isfinite(tail_values).all():
        raise ForecastContractError("Forecast tail contains non-finite values")
    probability_columns = [
        "prob_observed_after_horizon",
        "prob_unobserved_eol",
        "prob_no_observed_eol_by_horizon",
    ]
    if ((tail[probability_columns] < -tolerance) | (tail[probability_columns] > 1 + tolerance)).any().any():
        raise ForecastContractError("Forecast tail probabilities must be bounded by [0, 1]")
    if (tail["mean_excess_rul_days_given_observed_after_horizon"] < 0).any():
        raise ForecastContractError("Observed-tail mean excess RUL cannot be negative")

    final_cdf = values[:, -1]
    observed_tail = tail["prob_observed_after_horizon"].to_numpy(dtype=float)
    unobserved = tail["prob_unobserved_eol"].to_numpy(dtype=float)
    if not np.allclose(final_cdf + observed_tail + unobserved, 1.0, atol=tolerance):
        raise ForecastContractError("Final CDF and tail probabilities do not sum to one")
    expected_no_event = observed_tail + unobserved
    reported_no_event = tail["prob_no_observed_eol_by_horizon"].to_numpy(dtype=float)
    if not np.allclose(expected_no_event, reported_no_event, atol=tolerance):
        raise ForecastContractError("Inconsistent prob_no_observed_eol_by_horizon")

    canonical_curves = pd.DataFrame(
        {
            "battery_id": np.repeat(expected_ids.to_numpy(), len(expected_dates)),
            "forecast_date": np.tile(expected_dates.to_numpy(), len(expected_ids)),
            "failure_cdf": values.reshape(-1),
        }
    )
    canonical_tail = tail.reset_index()
    summaries = forecast.summaries.copy()
    if not summaries.empty and "battery_id" in summaries:
        summaries["battery_id"] = summaries["battery_id"].astype(str)

    return RiskForecast(metadata, canonical_curves, canonical_tail, summaries)


class VoltageTrendForecaster:
    """Deterministic submission-safe fallback until the fitted Task 1 model lands.

    It estimates a robust recent voltage level and decline rate, maps projected
    threshold crossing to a conditional logistic event distribution, and
    separates observed and evaluator-unobserved probability mass. This is a
    fallback, not a substitute for a calibrated out-of-fold Task 1 model.
    """

    model_version = "voltage-trend-fallback/v1"

    def __init__(
        self,
        *,
        eol_voltage: float = 2.4,
        history_days: int = 45,
        minimum_decline_per_day: float = 0.0008,
        uncertainty_days: float = 18.0,
    ) -> None:
        self.eol_voltage = float(eol_voltage)
        self.history_days = int(history_days)
        self.minimum_decline_per_day = float(minimum_decline_per_day)
        self.uncertainty_days = float(uncertainty_days)

    @staticmethod
    def _conditional_logistic_cdf(days: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
        raw = 1.0 / (1.0 + np.exp(-np.clip((days - location) / scale, -40.0, 40.0)))
        at_origin = 1.0 / (1.0 + np.exp(-np.clip((-location) / scale, -40.0, 40.0)))
        conditioned = (raw - at_origin) / np.maximum(1.0 - at_origin, 1e-9)
        return np.clip(conditioned, 0.0, 1.0)

    def _features(
        self,
        battery_data: pd.DataFrame,
        battery_ids: pd.Index,
        prediction_origin: pd.Timestamp,
    ) -> pd.DataFrame:
        if isinstance(battery_data.index, pd.MultiIndex):
            recent = battery_data.groupby(
                level=0, sort=False, observed=False
            ).tail(self.history_days * 24)
            frame = recent.reset_index()
        else:
            frame = battery_data.copy()
        if "device_id" not in frame or "end_time" not in frame or "voltage" not in frame:
            return pd.DataFrame(index=battery_ids, columns=["latest_voltage", "decline_per_day"])

        frame = frame[["device_id", "end_time", "voltage"]].copy()
        frame["device_id"] = frame["device_id"].astype(str)
        frame["end_time"] = pd.to_datetime(frame["end_time"])
        cutoff = pd.Timestamp(prediction_origin) - pd.Timedelta(days=self.history_days)
        frame = frame[frame["end_time"] >= cutoff]
        frame["age_days"] = (
            pd.Timestamp(prediction_origin) - frame["end_time"]
        ) / pd.Timedelta(days=1)

        latest = (
            frame[frame["age_days"] <= 3.0]
            .groupby("device_id", sort=False)["voltage"]
            .median()
            .rename("latest_voltage")
        )
        previous = (
            frame[(frame["age_days"] >= 10.0) & (frame["age_days"] <= 28.0)]
            .groupby("device_id", sort=False)["voltage"]
            .median()
            .rename("previous_voltage")
        )
        features = pd.concat([latest, previous], axis=1).reindex(battery_ids)
        features["decline_per_day"] = (
            (features["previous_voltage"] - features["latest_voltage"]) / 17.5
        ).clip(lower=self.minimum_decline_per_day)
        return features

    def predict(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        *,
        prediction_origin: pd.Timestamp,
        horizon_days: int,
        evaluation_observation_end: pd.Timestamp,
    ) -> RiskForecast:
        origin = _normal_date(prediction_origin)
        observation_end = max(_normal_date(evaluation_observation_end), origin)
        dates = pd.date_range(origin, origin + pd.Timedelta(days=horizon_days), freq="D")
        location_id_column = "battery_id" if "battery_id" in locations else "battery"
        battery_ids = pd.Index(locations[location_id_column].astype(str), name="battery_id")
        features = self._features(battery_data, battery_ids, pd.Timestamp(prediction_origin))

        population_voltage = float(features["latest_voltage"].median())
        if not np.isfinite(population_voltage):
            population_voltage = 2.85
        latest_voltage = features["latest_voltage"].fillna(population_voltage).to_numpy(dtype=float)
        decline = features["decline_per_day"].fillna(self.minimum_decline_per_day).to_numpy(dtype=float)
        crossing_days = np.clip(
            (latest_voltage - self.eol_voltage) / np.maximum(decline, self.minimum_decline_per_day),
            1.0,
            730.0,
        )
        scale = np.clip(
            self.uncertainty_days + 0.16 * crossing_days,
            self.uncertainty_days,
            90.0,
        )

        day_offsets = np.arange(len(dates), dtype=float)[None, :]
        physical_cdf = self._conditional_logistic_cdf(
            day_offsets,
            crossing_days[:, None],
            scale[:, None],
        )
        observed_days = float((observation_end - origin) / pd.Timedelta(days=1))
        observed_cdf = self._conditional_logistic_cdf(
            np.array([[observed_days]]),
            crossing_days[:, None],
            scale[:, None],
        )[:, 0]
        horizon_cdf = physical_cdf[:, -1]
        if observed_days < horizon_days:
            horizon_cdf = np.minimum(horizon_cdf, observed_cdf)
            physical_cdf = np.minimum(physical_cdf, observed_cdf[:, None])
        observed_tail = np.clip(observed_cdf - horizon_cdf, 0.0, 1.0)
        unobserved = np.clip(1.0 - observed_cdf, 0.0, 1.0)

        tail_midpoint = np.maximum(
            crossing_days - float(horizon_days),
            0.5 * max(observed_days - horizon_days, 0.0),
        )
        tail_midpoint = np.minimum(tail_midpoint, max(observed_days - horizon_days, 0.0))
        tail_midpoint = np.where(observed_tail > 1e-12, tail_midpoint, 0.0)

        curves = pd.DataFrame(
            {
                "battery_id": np.repeat(battery_ids.to_numpy(), len(dates)),
                "forecast_date": np.tile(dates.to_numpy(), len(battery_ids)),
                "failure_cdf": physical_cdf.reshape(-1),
            }
        )
        tail = pd.DataFrame(
            {
                "battery_id": battery_ids.to_numpy(),
                "prob_observed_after_horizon": observed_tail,
                "mean_excess_rul_days_given_observed_after_horizon": tail_midpoint,
                "prob_unobserved_eol": unobserved,
                "prob_no_observed_eol_by_horizon": observed_tail + unobserved,
            }
        )
        summaries = pd.DataFrame(
            {
                "battery_id": battery_ids.to_numpy(),
                "q10_days": np.maximum(crossing_days - 2.197 * scale, 0.0),
                "q50_days": crossing_days,
                "data_quality": features["latest_voltage"].notna().to_numpy(dtype=float),
                "cold_start": features["latest_voltage"].isna().to_numpy(),
            }
        )
        metadata = ForecastMetadata(
            contract_version=CONTRACT_VERSION,
            model_version=self.model_version,
            prediction_origin=origin,
            forecast_end_date=dates[-1],
            horizon_days=int(horizon_days),
            evaluation_observation_end=observation_end,
        )
        return RiskForecast(metadata, curves, tail, summaries)
