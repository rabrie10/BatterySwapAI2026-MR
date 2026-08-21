"""Adapter from the first-passage model to the Task 2 forecast contract.

``batteryswap_solution.forecast`` asks for three things per battery: the CDF of
a *recorded* EOL across the planning window, the probability the record lands
after the window but still inside the observation period, and the probability
there is no record at all. Those are exactly the three branches the evaluator
prices, so the split has to be made here rather than approximated later.

The censoring time is not a nuisance parameter: ``locations.end_time`` is handed
to ``plan()``, so for every battery we know the last day on which an EOL could
possibly be recorded, and the evaluator's substitute EOL for the unrecorded case
is the deterministic ``normalize(end_time + unobserved_eol_days)``. Two
consequences are wired in below:

* the CDF is capped at its own value at the censoring horizon, which is what
  makes late scenarios stop demanding service by themselves rather than through
  a hand-tuned survivor gate;
* a battery whose data ended before the scenario even started gets no
  probability mass at all, so the cost layer prices its substitute EOL in the
  past and the optimizer never touches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from batteryswap_solution.forecast import (
    CONTRACT_VERSION,
    ForecastMetadata,
    RiskForecast,
)

from .features import DeviceView, FeatureContext, feature_row, fleet_climatology
from .hazard import HazardModel
from .smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
_MAX_TAIL_DAYS = 400.0


def _normal_date(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


def _ordinal(value: pd.Timestamp) -> int:
    return int((value - _EPOCH) / pd.Timedelta(days=1))


class HazardForecaster:
    """Stateful across scenarios: the smoothing cache is the expensive part."""

    def __init__(
        self,
        model: HazardModel,
        *,
        use_split_climatology: bool = True,
        probability_scale: float = 1.0,
    ) -> None:
        """``probability_scale`` shrinks the whole CDF.

        The model ranks well but its probability *level* does not survive a
        change of buildings: measured out-of-fold at scenario cutoffs it
        predicts 12.86 due per scenario against a realised 9.46, and the top
        bucket predicts 0.87 against 0.39. Refitting the calibrator on the
        scenario population does not help, because the shift is between
        buildings and a calibrator fitted on the other four folds cannot see it
        -- which is exactly the situation on the public and private splits.

        So the correction is a deliberate, tunable under-confidence rather than
        a calibrator. It is one number, selected out-of-fold, and it errs in the
        cheaper direction: under-confidence costs late swaps on a handful of
        batteries, over-confidence costs early swaps on dozens.
        """
        self.model = model
        self.probability_scale = float(probability_scale)
        self.model_version = model.model_version
        self.cache = SmoothingCache()
        self.use_split_climatology = use_split_climatology
        self._context: FeatureContext | None = None
        self.last_cold_start = 0
        self.last_expected_due = 0.0
        self.last_probabilities: pd.Series | None = None

    def _refresh_context(self) -> FeatureContext:
        if not self.use_split_climatology:
            return self.model.context()
        profile = fleet_climatology(
            {d: (s.origin, s.smooth_temperature) for d, s in self.cache.devices.items()}
        )
        if not np.isfinite(profile).all() or np.allclose(profile, 0.0):
            return self.model.context()
        return FeatureContext(climatology=profile)

    def predict(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        *,
        prediction_origin: pd.Timestamp,
        horizon_days: int,
        evaluation_observation_end: pd.Timestamp,
    ) -> RiskForecast:
        self.cache.update(battery_data)
        self._context = self._refresh_context()

        origin = _normal_date(prediction_origin)
        origin_ordinal = _ordinal(origin)
        dates = pd.date_range(origin, origin + pd.Timedelta(days=horizon_days), freq="D")
        day_offsets = np.arange(horizon_days + 1, dtype=float)

        id_column = "battery_id" if "battery_id" in locations else "battery"
        battery_ids = pd.Index(locations[id_column].astype(str), name="battery_id")
        observation_end = pd.to_datetime(locations["end_time"])
        if getattr(observation_end.dt, "tz", None) is not None:
            observation_end = observation_end.dt.tz_localize(None)
        observation_end = observation_end.dt.normalize()
        remaining = ((observation_end - origin) / pd.Timedelta(days=1)).to_numpy(
            dtype=float
        )

        rows: list[list[float]] = []
        positions: list[int] = []
        row_devices: list[str] = []
        for position, device_id in enumerate(battery_ids):
            series = self.cache.devices.get(device_id)
            if series is None:
                continue
            index = series.index_of(origin_ordinal)
            if index < 0:
                continue
            index = min(index, len(series) - 1)
            view = DeviceView(series.smooth_voltage, series.smooth_temperature)
            row = feature_row(view, index, series.origin + index, self._context)
            if row is None:
                continue
            rows.append(row)
            positions.append(position)
            row_devices.append(device_id)

        count = len(battery_ids)
        self.last_cold_start = count - len(positions)
        grid = np.zeros((count, len(self.model.horizons)))
        if rows:
            predicted = self.model.predict_grid(
                np.asarray(rows, dtype=np.float32),
                remaining[np.asarray(positions, dtype=int)],
                np.asarray(row_devices),
            )
            grid[np.asarray(positions, dtype=int)] = np.clip(
                predicted * self.probability_scale, 0.0, 1.0
            )

        daily = self.model.cdf_at(grid, day_offsets)

        # No record can be filed after the observation period ends, so the CDF
        # flattens there. A device whose data ended before this scenario began
        # gets no mass at all.
        censor = self._cdf_at_scalar(grid, remaining)
        censor = np.where(remaining < 0.0, 0.0, censor)
        daily = np.minimum(daily, censor[:, None])

        horizon_cdf = daily[:, -1]
        observed_tail = np.clip(censor - horizon_cdf, 0.0, 1.0)
        unobserved = np.clip(1.0 - censor, 0.0, 1.0)
        # Absorb interpolation slack so the contract's sum-to-one check holds.
        unobserved = np.clip(1.0 - horizon_cdf - observed_tail, 0.0, 1.0)

        # Predicted expected number of due batteries. Comparing this against the
        # realised count per scenario is the calibration check that matters:
        # the shipped v3 model predicted about 20.6 against an actual 9.5, and
        # that same over-prediction is what produced 41 swaps per scenario on
        # the public leaderboard against 11 locally.
        self.last_expected_due = float(horizon_cdf.sum())
        self.last_probabilities = pd.Series(horizon_cdf, index=battery_ids.to_numpy())

        mean_excess = self._tail_mean_excess(
            grid, remaining, float(horizon_days), horizon_cdf, observed_tail
        )

        curves = pd.DataFrame(
            {
                "battery_id": np.repeat(battery_ids.to_numpy(), len(dates)),
                "forecast_date": np.tile(dates.to_numpy(), count),
                "failure_cdf": daily.reshape(-1),
            }
        )
        tail = pd.DataFrame(
            {
                "battery_id": battery_ids.to_numpy(),
                "prob_observed_after_horizon": observed_tail,
                "mean_excess_rul_days_given_observed_after_horizon": mean_excess,
                "prob_unobserved_eol": unobserved,
                "prob_no_observed_eol_by_horizon": observed_tail + unobserved,
            }
        )
        summaries = pd.DataFrame(
            {
                "battery_id": battery_ids.to_numpy(),
                "horizon_probability": horizon_cdf,
                "remaining_observation_days": remaining,
                "cold_start": ~np.isin(np.arange(count), positions),
            }
        )
        metadata = ForecastMetadata(
            contract_version=CONTRACT_VERSION,
            model_version=self.model_version,
            prediction_origin=origin,
            forecast_end_date=dates[-1],
            horizon_days=int(horizon_days),
            evaluation_observation_end=_normal_date(evaluation_observation_end),
        )
        return RiskForecast(metadata, curves, tail, summaries)

    def _cdf_at_scalar(self, grid: np.ndarray, days: np.ndarray) -> np.ndarray:
        xs = np.concatenate([[0.0], np.asarray(self.model.horizons, dtype=float)])
        out = np.empty(grid.shape[0])
        for row in range(grid.shape[0]):
            ys = np.concatenate([[0.0], grid[row]])
            out[row] = float(np.interp(max(days[row], 0.0), xs, ys))
        return np.clip(out, 0.0, 1.0)

    def _tail_mean_excess(
        self,
        grid: np.ndarray,
        remaining: np.ndarray,
        horizon: float,
        horizon_cdf: np.ndarray,
        observed_tail: np.ndarray,
    ) -> np.ndarray:
        """E[T - horizon] over the mass that lands after the window but on record."""
        xs = np.concatenate([[0.0], np.asarray(self.model.horizons, dtype=float)])
        out = np.zeros(grid.shape[0])
        for row in range(grid.shape[0]):
            if observed_tail[row] <= 1e-12:
                continue
            upper = min(float(remaining[row]), _MAX_TAIL_DAYS)
            if upper <= horizon:
                continue
            days = np.linspace(horizon, upper, 32)
            ys = np.concatenate([[0.0], grid[row]])
            values = np.clip(np.interp(days, xs, ys), 0.0, 1.0)
            values = np.maximum.accumulate(values)
            weights = np.diff(values)
            total = weights.sum()
            if total <= 1e-12:
                out[row] = 0.5 * (upper - horizon)
                continue
            centres = 0.5 * (days[:-1] + days[1:]) - horizon
            out[row] = float(np.clip((weights * centres).sum() / total, 0.0, upper - horizon))
        return out
