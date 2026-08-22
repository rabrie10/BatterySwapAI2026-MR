"""Within-day shape of the raw hourly readings.

``smooth_series`` collapses each calendar day to one median and then takes a
seven-day trailing median of those. On the train split that turns 8,520,098
hourly readings into 360,847 numbers -- and every feature V6 had was some
function of that single collapsed series. Measured against a two-line control
(``margin / -slope``), the whole fifty-one-feature gradient-boosted model was
worth nothing: precision 0.300 against 0.309 at twelve swaps.

What the collapse throws away is the daily cycle, and that is where the early
warning lives. A battery's voltage responds to the daily temperature swing
through its internal resistance, and resistance rises as the cell approaches
collapse. Measured on the population the level-and-slope rule cannot rank at all
-- rows where the extrapolation says more than sixty days to crossing -- the
within-day sensitivity separates the batteries that cross within six weeks from
those that do not with AUC 0.871:

    within-day dV/dT   due median 0.01266   not due 0.00329   ratio 3.84
    daily voltage sd   due median 0.02682   not due 0.00686   ratio 3.91
    daily voltage range due median 0.08545  not due 0.02287   ratio 3.74

That is the "knee onset" surprise made visible: those batteries sit at a higher
voltage on a slower slope, so nothing in the smoothed series flags them, and
they cross anyway.

No temperature filter is applied here. ``smooth_series`` keeps only readings
between 10 and 30 degrees because that is how the competition defines end of
life, but narrowing the band would shrink the very temperature swing this module
measures against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01")

# A day needs enough readings for a within-day regression to mean anything.
MIN_READINGS = 6
# ... and enough temperature movement for the slope to be identifiable.
MIN_TEMPERATURE_SPREAD = 0.5


@dataclass
class DeviceShape:
    """One row per calendar day, on the same grid convention as DeviceSeries."""

    origin: int
    beta: np.ndarray  # within-day dV/dT, the internal-resistance proxy
    v_std: np.ndarray
    v_range: np.ndarray
    t_range: np.ndarray

    def __len__(self) -> int:
        return int(self.beta.shape[0])

    def index_of(self, day_ordinal: int) -> int:
        return int(day_ordinal) - self.origin


def _flatten(battery_data: pd.DataFrame) -> pd.DataFrame:
    frame = battery_data
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
        frame = frame.reset_index()
    columns = {c.lower(): c for c in frame.columns}
    time_column = columns.get("end_time") or columns.get("timestamp")
    device_column = columns.get("device_id") or columns.get("battery")
    out = pd.DataFrame(
        {
            "end_time": pd.to_datetime(frame[time_column]),
            "device_id": frame[device_column].astype(str),
            "voltage": frame[columns["voltage"]].astype(float),
            "temperature": frame[columns["temperature"]].astype(float),
        }
    )
    return out.dropna(subset=["end_time", "voltage", "temperature"])


@dataclass
class ShapeCache:
    """Daily within-day statistics per device, extended as scenarios advance.

    Mirrors ``SmoothingCache``: the boundary day is re-read in full on every
    update, because a scenario cut lands mid-day and a partial day would give a
    within-day slope fitted to half a temperature cycle.
    """

    devices: dict[str, DeviceShape] = field(default_factory=dict)
    watermark: pd.Timestamp | None = None

    def update(self, battery_data: pd.DataFrame) -> None:
        frame = _flatten(battery_data)
        if frame.empty:
            return
        if self.watermark is not None:
            frame = frame[frame["end_time"] >= self.watermark]
            if frame.empty:
                return

        latest = frame["end_time"].max()
        self.watermark = pd.Timestamp(latest).normalize()

        day = ((frame["end_time"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)).to_numpy(
            dtype=np.int64
        )
        work = frame.assign(
            _day=day,
            _vt=frame["voltage"] * frame["temperature"],
            _tt=frame["temperature"] ** 2,
        )
        grouped = work.groupby(["device_id", "_day"], sort=True, observed=True)
        daily = grouped.agg(
            v_mean=("voltage", "mean"),
            v_std=("voltage", "std"),
            v_min=("voltage", "min"),
            v_max=("voltage", "max"),
            t_mean=("temperature", "mean"),
            t_min=("temperature", "min"),
            t_max=("temperature", "max"),
            vt=("_vt", "mean"),
            tt=("_tt", "mean"),
            n=("voltage", "size"),
        )
        if daily.empty:
            return

        covariance = daily["vt"] - daily["v_mean"] * daily["t_mean"]
        variance = daily["tt"] - daily["t_mean"] ** 2
        spread = daily["t_max"] - daily["t_min"]
        usable = (
            (daily["n"] >= MIN_READINGS)
            & (spread >= MIN_TEMPERATURE_SPREAD)
            & (variance > 1e-9)
        )
        beta = np.where(usable, covariance / variance.where(variance > 1e-9, np.nan), np.nan)

        block = pd.DataFrame(
            {
                "beta": beta,
                "v_std": np.where(daily["n"] >= MIN_READINGS, daily["v_std"], np.nan),
                "v_range": np.where(
                    daily["n"] >= MIN_READINGS, daily["v_max"] - daily["v_min"], np.nan
                ),
                "t_range": spread.to_numpy(),
            },
            index=daily.index,
        )

        for device_id, rows in block.groupby(level=0, sort=False, observed=True):
            days = rows.index.get_level_values(1).to_numpy(dtype=np.int64)
            if days.size == 0:
                continue
            self._merge(str(device_id), days, rows)

    def _merge(self, device_id: str, days: np.ndarray, rows: pd.DataFrame) -> None:
        first, last = int(days[0]), int(days[-1])
        existing = self.devices.get(device_id)
        if existing is None:
            origin = first
            length = last - first + 1
            shape = DeviceShape(
                origin=origin,
                beta=np.full(length, np.nan),
                v_std=np.full(length, np.nan),
                v_range=np.full(length, np.nan),
                t_range=np.full(length, np.nan),
            )
            self.devices[device_id] = shape
        else:
            shape = existing
            origin = shape.origin
            if last > origin + len(shape) - 1:
                pad = last - (origin + len(shape) - 1)
                for name in ("beta", "v_std", "v_range", "t_range"):
                    setattr(
                        shape,
                        name,
                        np.concatenate([getattr(shape, name), np.full(pad, np.nan)]),
                    )

        positions = days - shape.origin
        inside = (positions >= 0) & (positions < len(shape))
        positions = positions[inside]
        for name in ("beta", "v_std", "v_range", "t_range"):
            getattr(shape, name)[positions] = rows[name].to_numpy(dtype=float)[inside]


class ShapeView:
    """Trailing means over the within-day statistics, cheap at any cutoff.

    Built by :func:`align_to`, which places a device's shape arrays on the same
    grid origin the smoothing cache uses -- the two can start on different days
    because the smoothing drops readings outside 10-30 degrees and this module
    does not.
    """

    __slots__ = ("beta", "v_std", "v_range", "t_range", "size", "_sums", "_counts")

    NAMES = ("beta", "v_std", "v_range", "t_range")

    def trailing_mean(self, name: str, index: int, window: int) -> float:
        """Mean of the last ``window`` days ending at ``index`` inclusive.

        Missing days are skipped rather than treated as zero, so a device that
        reports intermittently still gets a mean over the days it did report.
        """
        if index < 0 or index >= self.size:
            return float("nan")
        low = max(0, index - window + 1)
        high = index + 1
        count = int(self._counts[name][high] - self._counts[name][low])
        if count <= 0:
            return float("nan")
        return float(self._sums[name][high] - self._sums[name][low]) / count

    def prefix_median_iqr(
        self, name: str, index: int, min_count: int = 30
    ) -> tuple[float, float]:
        """Median and IQR of the daily values from the start through ``index``.

        A causal prefix statistic: at cutoff ``index`` it sees only days at or
        before the cutoff, so the same call means the same thing in training
        and at deployment. Used by the invariant feature variant to express a
        device's within-day sensitivity in units of its *own* history instead
        of the building-bound absolute scale.
        """
        if index < 0:
            return float("nan"), float("nan")
        values = getattr(self, name)[: min(index, self.size - 1) + 1]
        finite = values[np.isfinite(values)]
        if finite.size < min_count:
            return float("nan"), float("nan")
        q25, q50, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
        return float(q50), float(q75 - q25)


def align_to(shape: DeviceShape | None, grid_origin: int, size: int) -> ShapeView:
    """Build a view whose index 0 is ``grid_origin``, matching DeviceSeries."""
    view = ShapeView.__new__(ShapeView)
    view.size = int(size)
    arrays: dict[str, np.ndarray] = {}
    for name in ShapeView.NAMES:
        out = np.full(size, np.nan)
        if shape is not None:
            values = getattr(shape, name)
            offset = shape.origin - grid_origin
            source_start = max(0, -offset)
            target_start = max(0, offset)
            span = min(values.size - source_start, size - target_start)
            if span > 0:
                out[target_start : target_start + span] = values[
                    source_start : source_start + span
                ]
        arrays[name] = out
        setattr(view, name, out)
    view._sums = {}
    view._counts = {}
    for name, values in arrays.items():
        present = np.isfinite(values)
        view._sums[name] = np.concatenate([[0.0], np.cumsum(np.where(present, values, 0.0))])
        view._counts[name] = np.concatenate([[0], np.cumsum(present)])
    return view
