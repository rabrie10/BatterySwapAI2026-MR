"""Raw daily medians: the information below the smoothing's lag floor.

Every feature the model has ever used lives on the official smoothed grid,
which is a seven-day trailing median of daily medians: it lags reality by
roughly 3.5 days by construction. The raw daily medians at the cutoff are a
distinct information channel: a battery whose last raw dailies read 2.41,
2.39, 2.38 will drag the trailing median under 2.4 within days regardless of
what the smoothed margin says, and roughly 2.2 dues per scenario fail inside
the first ten window days where this channel dominates.

Same 10-30 degC filter and >=5-readings rule as the official smoothing, so a
raw daily exists exactly where an official daily median exists; only the
seven-day rolling step is omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")

MIN_COUNT = 5
MIN_TEMP = 10.0
MAX_TEMP = 30.0


@dataclass
class DeviceRawDaily:
    """One raw daily median per calendar day, on the ordinal-day grid."""

    origin: int
    median: np.ndarray

    def __len__(self) -> int:
        return int(self.median.shape[0])


@dataclass
class RawDailyCache:
    """Per-device raw daily medians, extended as scenarios advance."""

    devices: dict[str, DeviceRawDaily] = field(default_factory=dict)
    watermark: pd.Timestamp | None = None

    def update(self, battery_data: pd.DataFrame) -> None:
        frame = battery_data
        if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        columns = {c.lower(): c for c in frame.columns}
        time_column = columns.get("end_time") or columns.get("timestamp")
        device_column = columns.get("device_id") or columns.get("battery")
        work = pd.DataFrame(
            {
                "end_time": pd.to_datetime(frame[time_column]),
                "device_id": frame[device_column].astype(str),
                "voltage": frame[columns["voltage"]].astype(float),
                "temperature": frame[columns["temperature"]].astype(float),
            }
        ).dropna(subset=["end_time", "voltage"])
        if self.watermark is not None:
            work = work[work["end_time"] >= self.watermark]
        if work.empty:
            return
        self.watermark = pd.Timestamp(work["end_time"].max()).normalize()

        stable = work[(work["temperature"] > MIN_TEMP) & (work["temperature"] < MAX_TEMP)]
        if stable.empty:
            return
        day = (
            (stable["end_time"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)
        ).to_numpy(dtype=np.int64)
        grouped = stable.assign(_day=day).groupby(["device_id", "_day"], sort=True)
        daily = grouped["voltage"].agg(["median", "size"])
        daily = daily[daily["size"] >= MIN_COUNT]
        if daily.empty:
            return
        for device_id, rows in daily.groupby(level=0, sort=False):
            days = rows.index.get_level_values(1).to_numpy(dtype=np.int64)
            values = rows["median"].to_numpy(dtype=float)
            self._merge(str(device_id), days, values)

    def _merge(self, device_id: str, days: np.ndarray, values: np.ndarray) -> None:
        first, last = int(days[0]), int(days[-1])
        existing = self.devices.get(device_id)
        if existing is None:
            series = DeviceRawDaily(
                origin=first, median=np.full(last - first + 1, np.nan)
            )
            self.devices[device_id] = series
        else:
            series = existing
            if last > series.origin + len(series) - 1:
                pad = last - (series.origin + len(series) - 1)
                series.median = np.concatenate([series.median, np.full(pad, np.nan)])
        positions = days - series.origin
        inside = (positions >= 0) & (positions < len(series))
        series.median[positions[inside]] = values[inside]

    def features_at(self, device_id: str, day_ordinal: int) -> list[float]:
        """Raw-channel features at one cutoff. Order matches RAW_FEATURE_NAMES."""
        series = self.devices.get(device_id)
        if series is None:
            return [np.nan] * len(RAW_FEATURE_NAMES)
        index = int(day_ordinal) - series.origin
        if index < 0:
            return [np.nan] * len(RAW_FEATURE_NAMES)
        index = min(index, len(series) - 1)
        window = series.median[max(0, index - 13) : index + 1]
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            return [np.nan] * len(RAW_FEATURE_NAMES)
        last = float(valid[-1])
        last3 = valid[-3:]
        last7 = valid[-7:]
        slope7 = (
            (float(last7[-1]) - float(last7[0])) / max(1, last7.size - 1)
            if last7.size >= 3
            else np.nan
        )
        return [
            last,
            float(np.min(last3)),
            float(np.mean(last3)),
            float(np.min(last7)),
            slope7,
            float(np.sum(last7 < 2.42)),
            float(np.sum(last7 < 2.45)),
        ]


RAW_FEATURE_NAMES = [
    "raw_last",
    "raw_min3",
    "raw_mean3",
    "raw_min7",
    "raw_slope7",
    "raw_days7_below_2.42",
    "raw_days7_below_2.45",
]
