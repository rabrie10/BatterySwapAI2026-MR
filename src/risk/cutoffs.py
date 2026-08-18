"""Causal (device, cutoff) example construction for Task 1 training.

An example is a `(device_id, cutoff)` pair where the device is "at risk"
(not yet at its terminal event/censoring time). Duration/event columns follow
standard right-censored survival notation, which already implements the
spec's masked multi-horizon label rule (Sec 5.1) for free: a censored
duration correctly contributes no information about horizons beyond the
censoring time to the likelihood, while an observed event contributes exact
information at every horizon. No feature may depend on data after `cutoff`;
this module only ever reads `feature_series[device_id]` through
`features.lookup_asof`, which enforces that boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import INDIVIDUAL_FEATURE_COLUMNS, leave_one_out_building_features, lookup_asof

DEFAULT_SYNTHETIC_STEP_DAYS = 21
MIN_WARMUP_DAYS = 0


def _normalize(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


def build_cutoff_grid(
    scenario_start_times: list[pd.Timestamp],
    timeline_start: pd.Timestamp,
    timeline_end: pd.Timestamp,
    step_days: int = DEFAULT_SYNTHETIC_STEP_DAYS,
) -> pd.DatetimeIndex:
    """Official scenario starts unioned with a regular synthetic grid.

    Official cutoffs are the highest-value examples because they match
    inference exactly (Sec 5.1 of the design spec); the synthetic grid adds
    event coverage across the full observed timeline.
    """

    official = pd.DatetimeIndex([_normalize(value) for value in scenario_start_times])
    synthetic = pd.date_range(_normalize(timeline_start), _normalize(timeline_end), freq=f"{step_days}D")
    combined = official.union(synthetic)
    return combined.sort_values()


def terminal_times(locations: pd.DataFrame, eol_times: pd.Series) -> pd.DataFrame:
    """Per-device (terminal_time, event) where event=1 means an observed EOL."""

    loc = locations.set_index(locations["battery"].astype(str))
    eol = eol_times.copy()
    eol.index = eol.index.astype(str)
    eol = eol.reindex(loc.index)
    window_end = pd.to_datetime(loc["end_time"])
    if window_end.dt.tz is not None:
        window_end = window_end.dt.tz_localize(None)
    eol_naive = pd.to_datetime(eol)
    if eol_naive.dt.tz is not None:
        eol_naive = eol_naive.dt.tz_localize(None)
    observed = eol_naive.notna()
    terminal = eol_naive.where(observed, window_end)
    return pd.DataFrame(
        {
            "terminal_time": terminal.map(_normalize),
            "observed_event": observed.astype(int),
        },
        index=loc.index,
    )


def build_example_table(
    locations: pd.DataFrame,
    eol_times: pd.Series,
    feature_series_by_device: dict[str, pd.DataFrame],
    cutoff_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble the full causal (device, cutoff) training table."""

    loc = locations.set_index(locations["battery"].astype(str))
    start_times = pd.to_datetime(loc["start_time"])
    if start_times.dt.tz is not None:
        start_times = start_times.dt.tz_localize(None)
    building_by_device = loc["building"].astype(str)
    terminal = terminal_times(locations, eol_times)

    rows: list[dict] = []
    for device_id in loc.index:
        device_terminal = terminal.loc[device_id, "terminal_time"]
        device_observed = int(terminal.loc[device_id, "observed_event"])
        device_start = start_times.loc[device_id]
        series = feature_series_by_device.get(device_id, pd.DataFrame())
        first_seen = series.index[0] if not series.empty else device_start

        alive_cutoffs = cutoff_dates[cutoff_dates < device_terminal]
        for cutoff in alive_cutoffs:
            duration_days = float((device_terminal - cutoff) / pd.Timedelta(days=1))
            if duration_days <= 0:
                continue
            features = lookup_asof(series, cutoff, first_seen=first_seen)
            row = {
                "device_id": device_id,
                "building_id": building_by_device.loc[device_id],
                "cutoff": cutoff,
                "duration_days": duration_days,
                "event": device_observed,
                "age_days": float((cutoff - device_start) / pd.Timedelta(days=1)),
            }
            row.update(features)
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["not_yet_deployed"] = (table["age_days"] < 0).astype(float)
    table["cold_start"] = table["n_readings_total"].fillna(0.0) < 3
    table = leave_one_out_building_features(table)

    counts = table.groupby("device_id")["device_id"].transform("size")
    table["sample_weight"] = 1.0 / counts

    return table


def assign_building_folds(table: pd.DataFrame, n_folds: int = 5, seed: int = 20260818) -> pd.Series:
    """Deterministic grouped folds: every cutoff from one building stays together.

    This is the "unseen buildings" axis of causal validation. Using a hash of
    the building id (rather than sklearn's GroupKFold order-dependent
    assignment) keeps fold membership stable across reruns and subsets.
    """

    buildings = sorted(table["building_id"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(buildings))
    fold_by_building = {
        building: int(fold_index % n_folds)
        for fold_index, building in zip(order, buildings)
    }
    return table["building_id"].map(fold_by_building)


def time_holdout_mask(table: pd.DataFrame, holdout_fraction: float = 0.2) -> pd.Series:
    """Boolean mask marking the temporally latest cutoffs as a held-out time slice.

    This is the "unknown time period" axis of causal validation: a model
    selected only on building-grouped folds could still be quietly overfit to
    the historical period all cutoffs are drawn from. This mask is used only
    as a secondary diagnostic (never for fitting or calibration) so it never
    needs to compose with the building folds.
    """

    threshold = table["cutoff"].quantile(1.0 - holdout_fraction)
    return table["cutoff"] >= threshold


__all__ = [
    "INDIVIDUAL_FEATURE_COLUMNS",
    "assign_building_folds",
    "build_cutoff_grid",
    "build_example_table",
    "terminal_times",
    "time_holdout_mask",
]
