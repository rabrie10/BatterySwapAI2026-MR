"""Variant-aware training frame builder for the V12 feature sets.

Mirrors ``bsai.hazard.build_training_frame`` exactly -- same warmup, stride,
stop-at-crossing and eligibility rules, so the cutoff population is identical
row for row -- but passes each device's raw-daily adapter into ``feature_row``
so the variant rows carry the raw channel (bsai/rawdaily.py). ``hazard.py`` is
frozen; this lives in tools/ instead of modifying it.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import DeviceView, FeatureContext, feature_row, fleet_climatology
from bsai.hazard import TrainingFrame
from bsai.rawdaily import RawDailyCache
from bsai.shape import ShapeCache, align_to
from bsai.smoothing import SmoothingCache
from bsai.v12_rawany import RawAnyCache


def build_variant_training_frame(
    cache: SmoothingCache,
    eol_index: dict[str, int | None],
    building_of: dict[str, str],
    observation_end_index: dict[str, int],
    *,
    shape_cache: ShapeCache | None = None,
    raw_cache: RawDailyCache | None = None,
    raw_any_cache: RawAnyCache | None = None,
    variant: str = "extended",
    stride: int = 3,
    warmup_days: int = 45,
) -> TrainingFrame:
    """One row per (device, cutoff) on the requested feature variant."""
    context = FeatureContext(
        climatology=fleet_climatology(
            {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
        )
    )
    rows: list[list[float]] = []
    device: list[str] = []
    building: list[str] = []
    cutoff: list[int] = []
    crossing: list[int] = []
    last_observed: list[int] = []
    observation_end: list[int] = []

    for device_id, series in cache.devices.items():
        valid = np.flatnonzero(~np.isnan(series.smooth_voltage))
        if valid.size == 0:
            continue
        first, last = int(valid[0]), int(valid[-1])
        cross = eol_index.get(device_id)
        stop = last if cross is None else min(last, int(cross) - 1)
        view = DeviceView(series.smooth_voltage, series.smooth_temperature)
        shape_view = align_to(
            None if shape_cache is None else shape_cache.devices.get(device_id),
            series.origin,
            len(series),
        )
        raw = (
            partial(raw_cache.features_at, device_id)
            if raw_cache is not None
            else None
        )
        raw_any = (
            partial(raw_any_cache.features_at, device_id)
            if raw_any_cache is not None
            else None
        )
        for index in range(first + warmup_days, stop + 1, stride):
            row = feature_row(
                view,
                index,
                series.origin + index,
                context,
                shape_view,
                variant=variant,
                raw=raw,
                raw_any=raw_any,
            )
            if row is None:
                continue
            rows.append(row)
            device.append(device_id)
            building.append(building_of.get(device_id, ""))
            cutoff.append(index)
            crossing.append(-1 if cross is None else int(cross))
            last_observed.append(last)
            observation_end.append(int(observation_end_index.get(device_id, last)))

    return TrainingFrame(
        features=np.asarray(rows, dtype=np.float32),
        device=np.asarray(device),
        building=np.asarray(building),
        cutoff=np.asarray(cutoff, dtype=np.int64),
        crossing=np.asarray(crossing, dtype=np.int64),
        last_observed=np.asarray(last_observed, dtype=np.int64),
        observation_end=np.asarray(observation_end, dtype=np.int64),
    )
