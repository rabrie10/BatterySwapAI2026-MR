"""Scenario-incidence forecasting and conservative top-K portfolio selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from batteryswap_solution.forecast import RiskForecast


HORIZON_FRACTIONS = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)
PROBABILITY_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
PROBABILITY_THRESHOLDS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
TOP_COUNTS = (3, 5, 10, 15, 20)


def scenario_feature_names() -> tuple[str, ...]:
    names = ["fleet_size"]
    names.extend(f"expected_due_f{index}" for index in range(len(HORIZON_FRACTIONS)))
    names.extend(f"probability_q{int(100 * q):02d}" for q in PROBABILITY_QUANTILES)
    names.extend(f"count_above_{threshold:g}" for threshold in PROBABILITY_THRESHOLDS)
    names.extend(f"top_{count}_sum" for count in TOP_COUNTS)
    names.extend(("probability_mean", "probability_sd", "probability_entropy"))
    return tuple(names)


def scenario_features(forecast: RiskForecast) -> np.ndarray:
    """Summarise the whole fleet without using identities or scenario dates."""
    curves = forecast.curves.pivot(
        index="battery_id", columns="forecast_date", values="failure_cdf"
    ).sort_index(axis=1)
    values = curves.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Portfolio incidence features require a non-empty forecast")
    final = np.clip(values[:, -1], 0.0, 1.0)
    sorted_probability = np.sort(final)[::-1]

    features: list[float] = [float(values.shape[0])]
    last = values.shape[1] - 1
    for fraction in HORIZON_FRACTIONS:
        position = int(np.clip(round(last * fraction), 0, last))
        features.append(float(np.clip(values[:, position], 0.0, 1.0).sum()))
    features.extend(float(np.quantile(final, q)) for q in PROBABILITY_QUANTILES)
    features.extend(
        float(np.count_nonzero(final >= threshold))
        for threshold in PROBABILITY_THRESHOLDS
    )
    features.extend(
        float(sorted_probability[: min(count, len(sorted_probability))].sum())
        for count in TOP_COUNTS
    )
    clipped = np.clip(final, 1e-9, 1.0 - 1e-9)
    entropy = -np.sum(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    features.extend((float(final.mean()), float(final.std()), float(entropy)))
    return np.asarray(features, dtype=np.float64)


def horizon_probabilities(forecast: RiskForecast) -> pd.Series:
    final_date = pd.to_datetime(forecast.curves["forecast_date"]).max()
    final = forecast.curves.loc[
        pd.to_datetime(forecast.curves["forecast_date"]) == final_date,
        ["battery_id", "failure_cdf"],
    ]
    result = final.set_index(final["battery_id"].astype(str))["failure_cdf"].astype(float)
    result.index.name = "battery_id"
    return result


@dataclass
class ScenarioIncidenceModel:
    """Small-data aggregate model, kept independent of battery ranking."""

    estimator: Any
    feature_names: tuple[str, ...]
    service_multiplier: float = 1.25
    service_buffer: float = 6.0
    minimum_budget: int = 2
    maximum_budget: int = 24
    candidate_multiplier: float = 2.0
    candidate_buffer: float = 6.0
    maximum_candidate_pool: int = 40
    model_version: str = "bsai-portfolio-incidence/v1"

    def predict_due(self, forecast: RiskForecast) -> float:
        row = scenario_features(forecast)
        if len(row) != len(self.feature_names):
            raise ValueError("Scenario incidence feature schema mismatch")
        predicted = float(np.asarray(self.estimator.predict(row[None, :])).reshape(-1)[0])
        return max(predicted, 0.0)

    def service_budget(self, forecast: RiskForecast) -> int:
        raw = int(np.ceil(self.service_multiplier * self.predict_due(forecast) + self.service_buffer))
        return int(np.clip(raw, self.minimum_budget, self.maximum_budget))

    def candidate_pool(self, forecast: RiskForecast) -> int:
        raw = int(np.ceil(self.candidate_multiplier * self.predict_due(forecast) + self.candidate_buffer))
        return int(np.clip(raw, self.minimum_budget, self.maximum_candidate_pool))


@dataclass
class CrossFittedIncidenceModel:
    """Validation-only incidence predictions keyed by causal scenario origin."""

    predicted_due_by_origin: dict[str, float]
    service_multiplier: float
    service_buffer: float
    minimum_budget: int = 2
    maximum_budget: int = 24
    candidate_multiplier: float = 2.0
    candidate_buffer: float = 6.0
    maximum_candidate_pool: int = 40
    model_version: str = "bsai-portfolio-incidence/oof-v1"

    @staticmethod
    def _key(forecast: RiskForecast) -> str:
        return pd.Timestamp(forecast.metadata.prediction_origin).normalize().isoformat()

    def predict_due(self, forecast: RiskForecast) -> float:
        key = self._key(forecast)
        if key not in self.predicted_due_by_origin:
            raise KeyError(f"No cross-fitted incidence prediction for {key}")
        return max(float(self.predicted_due_by_origin[key]), 0.0)

    def service_budget(self, forecast: RiskForecast) -> int:
        raw = int(np.ceil(self.service_multiplier * self.predict_due(forecast) + self.service_buffer))
        return int(np.clip(raw, self.minimum_budget, self.maximum_budget))

    def candidate_pool(self, forecast: RiskForecast) -> int:
        raw = int(np.ceil(self.candidate_multiplier * self.predict_due(forecast) + self.candidate_buffer))
        return int(np.clip(raw, self.minimum_budget, self.maximum_candidate_pool))


class PortfolioForecaster:
    """Annotate a battery forecast with an independently estimated top-K budget."""

    def __init__(self, base_forecaster, incidence_model: Any) -> None:
        self.base_forecaster = base_forecaster
        self.incidence_model = incidence_model
        self.model_version = (
            f"{getattr(base_forecaster, 'model_version', 'unknown')}+"
            f"{incidence_model.model_version}"
        )
        self.last_expected_due = 0.0
        self.last_service_budget = 0
        self.last_probabilities: pd.Series | None = None

    @property
    def last_cold_start(self) -> int:
        return int(getattr(self.base_forecaster, "last_cold_start", 0))

    def predict(self, battery_data, locations, **kwargs) -> RiskForecast:
        forecast = self.base_forecaster.predict(battery_data, locations, **kwargs)
        result, predicted_due, budget, probability = annotate_portfolio(
            forecast, self.incidence_model
        )
        self.last_expected_due = predicted_due
        self.last_service_budget = budget
        self.last_probabilities = probability
        return result


def annotate_portfolio(
    forecast: RiskForecast, incidence_model: Any
) -> tuple[RiskForecast, float, int, pd.Series]:
    """Attach incidence and ranking fields to an already-computed forecast."""
    probability = horizon_probabilities(forecast)
    predicted_due = incidence_model.predict_due(forecast)
    budget = incidence_model.service_budget(forecast)
    candidate_pool = incidence_model.candidate_pool(forecast)
    ranks = probability.rank(method="first", ascending=False).astype(int)

    summaries = forecast.summaries.copy()
    if "battery_id" not in summaries:
        summaries = pd.DataFrame({"battery_id": probability.index})
    summaries["battery_id"] = summaries["battery_id"].astype(str)
    summaries = summaries.set_index("battery_id").reindex(probability.index)
    summaries["portfolio_rank"] = ranks.reindex(summaries.index).to_numpy(dtype=int)
    summaries["scenario_predicted_due"] = predicted_due
    summaries["scenario_service_budget"] = budget
    summaries["scenario_candidate_pool"] = candidate_pool

    result = RiskForecast(
        replace(forecast.metadata, model_version=f"{forecast.metadata.model_version}+portfolio"),
        forecast.curves,
        forecast.tail,
        summaries.reset_index(),
    )
    return result, predicted_due, budget, probability
