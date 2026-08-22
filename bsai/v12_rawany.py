"""Any-temperature raw daily medians: the channel the 10-30 degC filter darkens.

Measured in outputs/roadblock_report.md (vii): of 1,430 scenario rows whose
official channel is dark >30 days, 651 (45.5%) have an any-temperature raw
daily median within 7 days of the cutoff, and 41 of 48 dues on dark rows (85%)
are raw-fresh in this channel. The official smoothing and bsai/rawdaily.py both
keep only 10-30 degC readings because that is how the competition defines end
of life; this cache drops the temperature filter entirely so a device that is
reporting -- just outside the band -- is observed instead of extrapolated.

Same shape as ``bsai.rawdaily.RawDailyCache`` (daily median voltage, >=5
readings per day) so the adapter pattern is identical; kept in a separate
V12-owned module because ``rawdaily.py`` is integrator-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")

MIN_COUNT = 5

RAW_ANY_FEATURE_NAMES = [
    "raw_any_last",
    "raw_any_min3",
]


@dataclass
class DeviceRawAny:
    """One any-temperature raw daily median per calendar day."""

    origin: int
    median: np.ndarray

    def __len__(self) -> int:
        return int(self.median.shape[0])


@dataclass
class RawAnyCache:
    """Per-device unfiltered raw daily medians, extended as scenarios advance."""

    devices: dict[str, DeviceRawAny] = field(default_factory=dict)
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
            }
        ).dropna(subset=["end_time", "voltage"])
        if self.watermark is not None:
            work = work[work["end_time"] >= self.watermark]
        if work.empty:
            return
        self.watermark = pd.Timestamp(work["end_time"].max()).normalize()

        day = (
            (work["end_time"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)
        ).to_numpy(dtype=np.int64)
        grouped = work.assign(_day=day).groupby(["device_id", "_day"], sort=True)
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
            series = DeviceRawAny(
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
        """Any-temperature features at one cutoff; order matches
        RAW_ANY_FEATURE_NAMES. Causal: reads days at or before the cutoff only.
        """
        series = self.devices.get(device_id)
        if series is None:
            return [np.nan] * len(RAW_ANY_FEATURE_NAMES)
        index = int(day_ordinal) - series.origin
        if index < 0:
            return [np.nan] * len(RAW_ANY_FEATURE_NAMES)
        index = min(index, len(series) - 1)
        window = series.median[max(0, index - 13) : index + 1]
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            return [np.nan] * len(RAW_ANY_FEATURE_NAMES)
        return [
            float(valid[-1]),
            float(np.min(valid[-3:])),
        ]
