"""Incremental, exact reimplementation of ``batteryswap_public.utils.smooth_series``.

The official helper costs about 26 seconds on a full split. ``make_submissions``
calls ``plan()`` once per scenario and hands it every measurement up to that
scenario's start, so recomputing the smoothing each time would cost roughly 40
minutes across the public and private splits alone.

Smoothing is causal: the daily resample is a calendar-day aggregate and the
rolling window trails. Smoothing a truncated series therefore equals truncating
a smoothed series, verified to 0.0 over 26,366 device-days. That licenses a
cache which only processes measurements newer than the previous scenario.

``tests/test_smoothing.py`` pins this module against the official function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Mirrors the defaults of batteryswap_public.utils.smooth_series.
MIN_TEMP = 10.0
MAX_TEMP = 30.0
DAILY_MIN_COUNT = 5
WINDOW_LENGTH = 7
WINDOW_MIN_PERIODS = 3  # int(0.5 * 7)

_EPOCH = pd.Timestamp("1970-01-01")


def _to_ordinal(days: pd.Series) -> np.ndarray:
    return ((days - _EPOCH) // pd.Timedelta(days=1)).to_numpy(dtype=np.int64)


def _rolling_median(values: np.ndarray) -> np.ndarray:
    """Trailing median over WINDOW_LENGTH grid days, skipping missing days."""
    return (
        pd.Series(values)
        .rolling(window=WINDOW_LENGTH, min_periods=WINDOW_MIN_PERIODS)
        .quantile(0.5)
        .to_numpy(dtype=float)
    )


@dataclass
class DeviceSeries:
    """Daily grid for one device. Index 0 is the day at ``origin``."""

    origin: int
    voltage: np.ndarray
    temperature: np.ndarray
    smooth_voltage: np.ndarray
    smooth_temperature: np.ndarray

    def __len__(self) -> int:
        return int(self.voltage.shape[0])

    def index_of(self, day_ordinal: int) -> int:
        return int(day_ordinal) - self.origin


@dataclass
class SmoothingCache:
    """Holds one daily grid per device and extends it as scenarios advance."""

    devices: dict[str, DeviceSeries] = field(default_factory=dict)
    watermark: pd.Timestamp | None = None

    def update(self, battery_data: pd.DataFrame) -> None:
        frame = _flatten(battery_data)
        if frame.empty:
            return

        if self.watermark is not None:
            # Re-read the boundary day in full. A scenario cut can land mid-day,
            # leaving that day's median based on a partial set of measurements.
            frame = frame[frame["end_time"] >= self.watermark]
            if frame.empty:
                return

        stable = frame[
            (frame["temperature"] > MIN_TEMP) & (frame["temperature"] < MAX_TEMP)
        ]
        latest = frame["end_time"].max()
        self.watermark = pd.Timestamp(latest).normalize()
        if stable.empty:
            return

        day = stable["end_time"].dt.normalize()
        # observed=True matters: device_id arrives from parquet as a Categorical
        # whose categories cover the whole file, so the default would build the
        # full category-by-day cartesian product.
        grouped = stable.assign(_day=_to_ordinal(day)).groupby(
            ["device_id", "_day"], sort=True, observed=True
        )
        daily = grouped[["voltage", "temperature"]].median()
        counts = grouped[["voltage", "temperature"]].count()
        # The official code masks values below the count threshold rather than
        # dropping the rows, and the resample grid still spans every day between
        # a device's first and last stable measurement. Keep both behaviours: a
        # thin day becomes a gap inside the grid, not a missing row.
        # The official resample returns float64 even though the parquet columns
        # are float32; keeping float64 here is what makes the values agree bit
        # for bit rather than only to float32 epsilon.
        daily = daily.mask(counts < DAILY_MIN_COUNT).astype(float)
        if daily.empty:
            return

        for device_id, block in daily.groupby(level=0, sort=False, observed=True):
            days = block.index.get_level_values(1).to_numpy(dtype=np.int64)
            if days.size == 0:
                # Filtering by count leaves unused level values behind.
                continue
            self._merge(
                str(device_id),
                days,
                block["voltage"].to_numpy(dtype=float),
                block["temperature"].to_numpy(dtype=float),
            )

    def _merge(
        self,
        device_id: str,
        days: np.ndarray,
        voltage: np.ndarray,
        temperature: np.ndarray,
    ) -> None:
        series = self.devices.get(device_id)
        first_new, last_new = int(days[0]), int(days[-1])

        if series is None:
            size = last_new - first_new + 1
            grid_v = np.full(size, np.nan)
            grid_t = np.full(size, np.nan)
            grid_v[days - first_new] = voltage
            grid_t[days - first_new] = temperature
            series = DeviceSeries(
                origin=first_new,
                voltage=grid_v,
                temperature=grid_t,
                smooth_voltage=np.full(size, np.nan),
                smooth_temperature=np.full(size, np.nan),
            )
            self.devices[device_id] = series
            dirty_from = 0
        else:
            # The official resample spans each device's own first to last stable
            # day, so the grid only ever grows forward.
            previous_length = len(series)
            if last_new >= series.origin + previous_length:
                pad = last_new - (series.origin + previous_length) + 1
                series.voltage = np.concatenate([series.voltage, np.full(pad, np.nan)])
                series.temperature = np.concatenate(
                    [series.temperature, np.full(pad, np.nan)]
                )
                series.smooth_voltage = np.concatenate(
                    [series.smooth_voltage, np.full(pad, np.nan)]
                )
                series.smooth_temperature = np.concatenate(
                    [series.smooth_temperature, np.full(pad, np.nan)]
                )
            positions = days - series.origin
            keep = positions >= 0
            if not keep.any():
                return
            positions = positions[keep]
            series.voltage[positions] = voltage[keep]
            series.temperature[positions] = temperature[keep]
            # Padding can introduce gap days that carry no stable measurement of
            # their own but still take a value from the trailing window, so the
            # dirty region has to start at the first padded slot too.
            dirty_from = min(int(positions.min()), previous_length)

        # A trailing window of WINDOW_LENGTH needs that much lead-in to be exact.
        start = max(0, dirty_from - (WINDOW_LENGTH - 1))
        series.smooth_voltage[dirty_from:] = _rolling_median(series.voltage[start:])[
            dirty_from - start :
        ]
        series.smooth_temperature[dirty_from:] = _rolling_median(
            series.temperature[start:]
        )[dirty_from - start :]

    def frame(self) -> pd.DataFrame:
        """Long-form smoothed output, matching ``smooth_series`` columns."""
        blocks = []
        for device_id, series in self.devices.items():
            days = series.origin + np.arange(len(series), dtype=np.int64)
            blocks.append(
                pd.DataFrame(
                    {
                        "device_id": device_id,
                        "end_time": _EPOCH + pd.to_timedelta(days, unit="D"),
                        "temperature": series.smooth_temperature,
                        "voltage": series.smooth_voltage,
                    }
                )
            )
        if not blocks:
            return pd.DataFrame(
                columns=["device_id", "end_time", "temperature", "voltage"]
            )
        return pd.concat(blocks, ignore_index=True)


def _flatten(battery_data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(battery_data.index, pd.MultiIndex):
        frame = battery_data.reset_index()
    else:
        frame = battery_data
    missing = {"device_id", "end_time", "voltage", "temperature"} - set(frame.columns)
    if missing:
        raise ValueError(f"battery_data missing columns: {sorted(missing)}")
    frame = frame[["device_id", "end_time", "voltage", "temperature"]]
    end_time = pd.to_datetime(frame["end_time"])
    if getattr(end_time.dt, "tz", None) is not None:
        end_time = end_time.dt.tz_localize(None)
    # The parquet columns are float32, but the official resample runs through
    # quantile(), which computes and returns float64. Widening here rather than
    # after aggregating is what preserves the low bits of the daily median.
    return frame.assign(
        end_time=end_time,
        voltage=frame["voltage"].astype(float),
        temperature=frame["temperature"].astype(float),
    )


def smooth_all(battery_data: pd.DataFrame) -> SmoothingCache:
    """Build a cache in one pass. Used by training, where there is no scenario loop."""
    cache = SmoothingCache()
    cache.update(battery_data)
    return cache
