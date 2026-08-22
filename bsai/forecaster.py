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

from functools import partial

import numpy as np
import pandas as pd

from batteryswap_solution.forecast import (
    CONTRACT_VERSION,
    ForecastMetadata,
    RiskForecast,
)

from . import features as feature_lib
from .calibrate import RemainingCalibration
from .features import DeviceView, FeatureContext, feature_row, fleet_climatology
from .hazard import HazardModel
from .rawdaily import RawDailyCache
from .shape import ShapeCache, align_to
from .smoothing import SmoothingCache
from .v12_rawany import RawAnyCache

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
        calibration: RemainingCalibration | None = None,
    ) -> None:
        """Two corrections sit between the model and the planner.

        ``probability_scale`` shrinks the whole CDF uniformly. ``calibration``
        corrects along the remaining-observation axis, and that one is not
        optional in practice: pooled, the model looks well calibrated at 0.93
        predicted-to-actual, but that average is 0.54 in the opening scenarios
        and 1.64 in the closing ones. It predicts the most failures where there
        are the fewest. A uniform scale cannot fix a bias that changes sign,
        which is why every global knob tried in V6 traded one end against the
        other and landed inside the noise floor.
        """
        self.model = model
        self.probability_scale = float(probability_scale)
        # Within-scenario rank -> realised-rate recalibration; needs the whole
        # scenario's rows at once, so it lives here rather than in the model,
        # where the out-of-fold dispatcher only ever sees one building.
        self.rank_calibration = getattr(model, "rank_calibration", None)
        # Correction along the remaining-observation axis. The pooled calibration
        # ratio of 0.93 hides an under-prediction of 0.54 in the opening
        # scenarios and an over-prediction of 1.64 in the closing ones, and no
        # single scalar can fix a bias that changes sign.
        if calibration is not None:
            model.calibration = calibration
        self.calibration = getattr(model, "calibration", None)
        self.model_version = model.model_version
        self.cache = SmoothingCache()
        self.shape_cache = ShapeCache()
        # Raw-daily channels for the variant feature sets (10-30 degC filtered
        # and any-temperature). Gated on the active feature variant exactly
        # like the variant rows themselves: under the default "base" variant
        # these caches are never updated or read (unless a resurrection gate
        # asks for the filtered one), so the base path is byte-identical.
        self.raw_cache = RawDailyCache()
        self.raw_any_cache = RawAnyCache()
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
        self.shape_cache.update(battery_data)
        needs_raw = feature_lib.variant_needs_raw(feature_lib.active_feature_variant())
        gate = getattr(self.model, "resurrection_gate", None)
        # The raw channel always updates: the selection-exchange dip flag reads
        # raw_min3 regardless of the feature variant or gate attribute, and the
        # incremental update is one groupby per scenario step.
        self.raw_cache.update(battery_data)
        if needs_raw:
            self.raw_any_cache.update(battery_data)
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
        row_margin: list[float] = []
        row_staleness: list[float] = []
        row_dwell: list[float] = []
        row_raw_min3: list[float] = []
        for position, device_id in enumerate(battery_ids):
            series = self.cache.devices.get(device_id)
            if series is None:
                continue
            index = series.index_of(origin_ordinal)
            if index < 0:
                continue
            index = min(index, len(series) - 1)
            view = DeviceView(series.smooth_voltage, series.smooth_temperature)
            shape_view = align_to(
                self.shape_cache.devices.get(device_id), series.origin, len(series)
            )
            row = feature_row(
                view,
                index,
                series.origin + index,
                self._context,
                shape_view,
                raw=partial(self.raw_cache.features_at, device_id)
                if needs_raw
                else None,
                raw_any=partial(self.raw_any_cache.features_at, device_id)
                if needs_raw
                else None,
            )
            if row is None:
                continue
            rows.append(row)
            positions.append(position)
            row_devices.append(device_id)
            value, stale = view.value_at_or_before(index)
            row_margin.append(float(value) - 2.4)
            # Clamped staleness, exactly matching the frame the exchange gates
            # were measured on. An overhang-corrected variant was tried and
            # flooded the dark gate with long-ended devices (validated +520):
            # the measured class is in-series gaps on live observations, and
            # the flag below also requires remaining observation.
            row_staleness.append(float(stale))
            below = view.first_below.get(2.45, -1)
            row_dwell.append(float(index - below) if 0 <= below <= index else -1.0)
            raw3 = self.raw_cache.features_at(device_id, series.origin + index)[1]
            row_raw_min3.append(float(raw3))

        count = len(battery_ids)
        self.last_cold_start = count - len(positions)
        grid = np.zeros((count, len(self.model.horizons)))
        if rows:
            predicted = self.model.predict_grid(
                np.asarray(rows, dtype=np.float32),
                remaining[np.asarray(positions, dtype=int)],
                np.asarray(row_devices),
            )
            if self.rank_calibration is not None and predicted.shape[0] > 0:
                column = min(11, predicted.shape[1] - 1)
                factor = self.rank_calibration.factors(predicted[:, column])
                predicted = np.clip(predicted * factor[:, None], 0.0, 1.0)
            if gate is not None and predicted.shape[0] > 0:
                column = min(11, predicted.shape[1] - 1)
                p42 = predicted[:, column]
                floors = gate.floors(
                    p42,
                    np.asarray(row_margin, dtype=float),
                    np.asarray(row_staleness, dtype=float),
                    np.asarray(row_raw_min3, dtype=float),
                )
                lift = floors > p42
                if lift.any():
                    scale = np.ones_like(p42)
                    scale[lift] = floors[lift] / np.maximum(p42[lift], 1e-4)
                    predicted = np.clip(predicted * scale[:, None], 0.0, 1.0)
                    # A gated row whose CDF was uniformly tiny keeps its shape;
                    # make sure the floor value itself is reached at the window.
                    predicted[lift, column] = np.maximum(
                        predicted[lift, column], floors[lift]
                    )
                    predicted = np.maximum.accumulate(predicted, axis=1)
            index = np.asarray(positions, dtype=int)
            # The remaining-observation calibration is applied inside the model,
            # where the out-of-fold dispatcher has already chosen the right fold.
            grid[index] = np.clip(predicted * self.probability_scale, 0.0, 1.0)

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
        # Selection-exchange flags for the planner's post-pass. Measured on the
        # paired-incumbent harness (exact deltas, official scorer, 48
        # scenarios): the joint exchange -- defer zombie-fingerprint planned
        # batteries AND force-include dark-channel batteries AND refill to the
        # cap -- is worth -79.2/scenario (33 wins / 13 losses, sign-test
        # p=0.0045), while either side alone is neutral-to-harmful. Flags
        # only: probabilities are untouched, so the expected-due budget and
        # the cost tables are unchanged.
        demote = np.zeros(count, dtype=bool)
        gate_include = np.zeros(count, dtype=bool)
        if positions:
            index_array = np.asarray(positions, dtype=int)
            p42_rows = grid[index_array, min(11, grid.shape[1] - 1)]
            margins = np.asarray(row_margin)
            dwells = np.asarray(row_dwell)
            stales = np.asarray(row_staleness)
            raw3 = np.asarray(row_raw_min3)
            fingerprint = (margins < 0.05) & (dwells > 42.0) & (p42_rows > 0.4)
            demote[index_array[fingerprint]] = True
            remaining_rows = remaining[index_array]
            dark = (
                (stales > 30.0)
                & ((margins - 0.001 * stales) < 0.02)
                & (remaining_rows >= 30.0)
            )
            dip = (
                np.isfinite(raw3)
                & (raw3 < 2.40)
                & (margins > 0.03)
                & (p42_rows < 0.10)
            )
            gate_include[index_array[dark | dip]] = True

        summaries = pd.DataFrame(
            {
                "battery_id": battery_ids.to_numpy(),
                "horizon_probability": horizon_cdf,
                "remaining_observation_days": remaining,
                "cold_start": ~np.isin(np.arange(count), positions),
                "slot_demote": demote,
                "gate_include": gate_include,
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
