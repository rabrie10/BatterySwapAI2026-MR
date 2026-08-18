"""Causal daily feature engineering for the Task 1 risk model.

Design goal: compute each device's rolling feature history exactly once from
its own raw readings, then look features up "as of" any cutoff in O(log n).
This avoids re-scanning raw history per (battery, cutoff) example, which the
solution design spec flags as the dominant cost trap in this pipeline.

Every feature in this module is causal by construction: a rolling window
anchored at date ``d`` only ever aggregates rows with date ``<= d``, and the
as-of lookup only ever returns the row for the latest available date
``<= cutoff``. Cross-device (building) aggregates are computed later, in
``leave_one_out_building_features``, using a groupby-transform that only mixes
other devices' own causal features at the *same* cutoff — never future rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EOL_VOLTAGE = 2.4
NEAR_THRESHOLD_BAND = 0.1
MIN_DECLINE_PER_DAY = 0.0005
MAX_CROSSING_DAYS = 3650.0
ROLLING_WINDOWS_DAYS: tuple[int, ...] = (7, 14, 28, 56, 90, 180)
SLOPE_WINDOWS_DAYS: tuple[int, ...] = (14, 28, 90)


def _flatten_readings(battery_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the contract's battery_data (MultiIndex or flat) to flat columns."""

    if isinstance(battery_data.index, pd.MultiIndex):
        frame = battery_data.reset_index()
    else:
        frame = battery_data.reset_index(drop=True).copy()
    frame["device_id"] = frame["device_id"].astype(str)
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    if frame["end_time"].dt.tz is not None:
        frame["end_time"] = frame["end_time"].dt.tz_localize(None)
    return frame[["device_id", "end_time", "voltage", "temperature"]]


def build_daily_panels(battery_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One robust daily (voltage, temperature, n_readings) series per device.

    The panel is sparse: a calendar day with no valid hourly reading simply has
    no row. Downstream rolling windows use pandas' offset-based rolling
    (``'{n}D'``), which aggregates over whatever rows fall in the trailing
    calendar window regardless of gaps, so sparsity does not need explicit
    reindexing/NaN handling.
    """

    frame = _flatten_readings(battery_data)
    frame = frame.dropna(subset=["voltage", "temperature"])
    if frame.empty:
        return {}
    frame["date"] = frame["end_time"].dt.normalize()
    daily = (
        frame.groupby(["device_id", "date"], sort=True, observed=True)
        .agg(
            voltage=("voltage", "median"),
            temperature=("temperature", "median"),
            n_readings=("voltage", "size"),
        )
        .reset_index()
    )
    panels: dict[str, pd.DataFrame] = {}
    for device_id, group in daily.groupby("device_id", sort=False, observed=True):
        indexed = group.set_index("date").sort_index()[["voltage", "temperature", "n_readings"]]
        panels[str(device_id)] = indexed
    return panels


def _rolling_slope(t: pd.Series, y: pd.Series, window: str) -> pd.Series:
    """Closed-form trailing OLS slope of y on t over a calendar-time window."""

    n = y.rolling(window).count()
    sum_t = t.rolling(window).sum()
    sum_y = y.rolling(window).sum()
    sum_ty = (t * y).rolling(window).sum()
    sum_tt = (t * t).rolling(window).sum()
    denom = n * sum_tt - sum_t * sum_t
    numer = n * sum_ty - sum_t * sum_y
    slope = numer / denom.replace(0.0, np.nan)
    slope[n < 2] = np.nan
    return slope


def compute_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Trailing causal feature series for one device, indexed by calendar date."""

    if panel.empty:
        return pd.DataFrame()
    idx = panel.index
    epoch_days = (idx - pd.Timestamp("2000-01-01")).days.to_numpy(dtype=float)
    t = pd.Series(epoch_days, index=idx)
    voltage = panel["voltage"]
    temperature = panel["temperature"]

    out = pd.DataFrame(index=idx)
    out["latest_voltage"] = voltage
    out["latest_temperature"] = temperature
    out["distance_to_threshold"] = voltage - EOL_VOLTAGE

    gap_days = idx.to_series().diff().dt.days.fillna(0.0)
    gap_days.index = idx

    for window_days in ROLLING_WINDOWS_DAYS:
        window = f"{window_days}D"
        count = voltage.rolling(window).count()
        low = (voltage < (EOL_VOLTAGE + NEAR_THRESHOLD_BAND)).astype(float)

        out[f"voltage_mean_{window_days}d"] = voltage.rolling(window).mean()
        out[f"voltage_std_{window_days}d"] = voltage.rolling(window).std()
        out[f"voltage_min_{window_days}d"] = voltage.rolling(window).min()
        out[f"temp_mean_{window_days}d"] = temperature.rolling(window).mean()
        out[f"temp_std_{window_days}d"] = temperature.rolling(window).std()
        out[f"n_readings_{window_days}d"] = count
        out[f"completeness_{window_days}d"] = (count / float(window_days)).clip(upper=1.0)
        out[f"frac_low_voltage_{window_days}d"] = low.rolling(window).sum() / count.replace(0.0, np.nan)
        out[f"max_gap_{window_days}d"] = gap_days.rolling(window).max()

    for window_days in SLOPE_WINDOWS_DAYS:
        window = f"{window_days}D"
        out[f"voltage_slope_{window_days}d"] = _rolling_slope(t, voltage, window)

    long_slope = out[f"voltage_slope_{SLOPE_WINDOWS_DAYS[-1]}d"]
    decline = (-long_slope).clip(lower=MIN_DECLINE_PER_DAY)
    out["crossing_days_extrapolated"] = (out["distance_to_threshold"] / decline).clip(
        lower=0.0, upper=MAX_CROSSING_DAYS
    )

    return out


def build_feature_series(battery_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-device rolling feature series, computed once from raw readings."""

    panels = build_daily_panels(battery_data)
    return {
        device_id: compute_rolling_features(panel)
        for device_id, panel in panels.items()
        if not panel.empty
    }


INDIVIDUAL_FEATURE_COLUMNS = [
    "latest_voltage",
    "latest_temperature",
    "distance_to_threshold",
    "crossing_days_extrapolated",
    "days_since_last_reading",
    "history_days_available",
    "n_readings_total",
]
for _window in ROLLING_WINDOWS_DAYS:
    INDIVIDUAL_FEATURE_COLUMNS.extend(
        [
            f"voltage_mean_{_window}d",
            f"voltage_std_{_window}d",
            f"voltage_min_{_window}d",
            f"temp_mean_{_window}d",
            f"temp_std_{_window}d",
            f"n_readings_{_window}d",
            f"completeness_{_window}d",
            f"frac_low_voltage_{_window}d",
            f"max_gap_{_window}d",
        ]
    )
for _window in SLOPE_WINDOWS_DAYS:
    INDIVIDUAL_FEATURE_COLUMNS.append(f"voltage_slope_{_window}d")


def lookup_asof(
    feature_series: pd.DataFrame, as_of: pd.Timestamp, first_seen: pd.Timestamp | None = None
) -> dict[str, float]:
    """Feature row for the latest available date <= as_of (strictly causal).

    Always returns the full ``INDIVIDUAL_FEATURE_COLUMNS`` schema (as NaN where
    unavailable) so downstream code sees a consistent set of columns even for
    a battery with zero readings ever recorded — not just zero readings before
    ``as_of``. Callers impute NaNs and treat them as cold start.
    """

    as_of = pd.Timestamp(as_of).normalize()
    result: dict[str, float] = {column: np.nan for column in INDIVIDUAL_FEATURE_COLUMNS}

    if feature_series.empty:
        result["history_days_available"] = 0.0
        result["n_readings_total"] = 0.0
        return result

    position = int(feature_series.index.searchsorted(as_of, side="right")) - 1
    if position < 0:
        result["history_days_available"] = 0.0
        result["n_readings_total"] = 0.0
        return result

    row = feature_series.iloc[position]
    result.update(row.to_dict())
    last_date = feature_series.index[position]
    first_date = first_seen if first_seen is not None else feature_series.index[0]
    result["days_since_last_reading"] = float((as_of - last_date).days)
    result["history_days_available"] = float((last_date - pd.Timestamp(first_date)).days)
    result["n_readings_total"] = float(position + 1)
    return result


LOO_SOURCE_COLUMNS = [
    "latest_voltage",
    "voltage_slope_28d",
    "voltage_slope_90d",
    "frac_low_voltage_28d",
    "crossing_days_extrapolated",
    "history_days_available",
]


def leave_one_out_building_features(table: pd.DataFrame) -> pd.DataFrame:
    """Causal building-context features: peer aggregates excluding the device itself.

    Computed with a groupby-transform over (cutoff, building) so that unseen
    test buildings still work at inference: the aggregate only ever depends on
    *other* devices' own already-causal features at the identical cutoff, so
    it carries no future information and no device-identity leakage.
    """

    out = table.copy()
    group_key = [out["cutoff"], out["building_id"]]
    for column in LOO_SOURCE_COLUMNS:
        values = out[column]
        valid = values.notna()
        group_sum = values.where(valid, 0.0).groupby(group_key).transform("sum")
        group_count = valid.groupby(group_key).transform("sum")
        own = values.where(valid, 0.0)
        loo_count = (group_count - valid.astype(float)).clip(lower=0.0)
        loo_sum = group_sum - own
        loo_mean = loo_sum / loo_count.replace(0.0, np.nan)
        out[f"building_loo_{column}"] = loo_mean
    out["building_loo_device_count"] = (
        out["building_id"].groupby(group_key).transform("size") - 1
    ).clip(lower=0)
    return out
