"""Knee-weighted Wiener fitting for the V12 generation.

Measured basis (outputs/roadblock_report.md, L2 ii-iv): knee-regime windows are
under-represented 2.3-2.4x by uniform-history sampling, short-horizon drops
rest on ~105 crossing windows fleet-wide, and squared loss on a 0.8%-event
population shrinks the predicted 42-day knee drop to 0.036 V against a required
0.12-0.20 V (only 12.1% of knee rows are predicted to reach the barrier).
Weighting windows whose END lies within KNEE_DAYS of the device's crossing
counteracts the shrinkage at the fit level instead of patching probabilities
afterwards -- the probability-layer patches all died in the planner.

``bsai/wiener.py`` is frozen; the helpers here reproduce its exact fitting
recipe (same estimators, constraints, residual-scatter construction) with a
``sample_weight`` added, and its exact censor-aware target construction with a
window-end sidecar added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor

from bsai.margin import EOL_THRESHOLD
from bsai.wiener import DEFAULT_PARAMS, FIT_HORIZONS, WienerModel

# A window "ends at the knee" when its endpoint lies within this many days of
# the device's recorded crossing (either side: the approach carries the plunge,
# the just-crossed windows carry the censor-aware full-margin bump).
KNEE_DAYS = 21.0


def knee_weights(end_to_crossing: np.ndarray, w_knee: float) -> np.ndarray:
    """1 + w_knee on knee-regime windows, 1 elsewhere."""
    near = np.isfinite(end_to_crossing) & (np.abs(end_to_crossing) <= KNEE_DAYS)
    return 1.0 + float(w_knee) * near.astype(float)


def fit_wiener_weighted(
    design: np.ndarray,
    drop: np.ndarray,
    climatology: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    params: dict | None = None,
) -> WienerModel:
    """WienerModel.fit's exact recipe with an optional sample_weight."""
    settings = dict(DEFAULT_PARAMS)
    settings.update(params or {})
    n_features = design.shape[1] - 1

    drift = HistGradientBoostingRegressor(
        monotonic_cst=WienerModel._constraints(n_features, 1), **settings
    )
    drift.fit(design, drop, sample_weight=sample_weight)

    residual = np.abs(drop - drift.predict(design))
    scatter = HistGradientBoostingRegressor(
        monotonic_cst=WienerModel._constraints(n_features, 1), **settings
    )
    scatter.fit(design, residual, sample_weight=sample_weight)

    return WienerModel(
        drift=drift,
        scatter=scatter,
        climatology=np.asarray(climatology, dtype=float),
    )


def build_increment_targets_with_meta(
    frame, cache, horizons: tuple[int, ...] = FIT_HORIZONS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bsai.wiener.build_increment_targets plus a window-end sidecar.

    Returns (design, drop, end_to_crossing) where end_to_crossing is the
    window end minus the device's crossing index (+inf when the device never
    crossed), row-aligned with the design. The target values reproduce the
    production censor-aware rule exactly.
    """
    designs: list[np.ndarray] = []
    drops: list[np.ndarray] = []
    sidecars: list[np.ndarray] = []

    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")

    for horizon in horizons:
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            margin = margins.get(device_id)
            start = stop
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= last) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            drop = here[finite] - there[finite]
            end_fin = ends[usable][finite].astype(float)
            if crossing >= 0:
                crossed = (
                    (cutoffs[usable][finite] < crossing) & (end_fin >= crossing)
                )
                drop = np.where(crossed, np.maximum(drop, here[finite]), drop)
                sidecars.append(end_fin - crossing)
            else:
                sidecars.append(np.full(end_fin.size, np.inf))
            designs.append(
                np.hstack(
                    [
                        frame.features[chosen[finite]],
                        np.full((int(finite.sum()), 1), horizon, dtype=np.float32),
                    ]
                )
            )
            drops.append(drop)

    if not designs:
        raise ValueError("no observable windows; check the horizons and the cache")
    return np.vstack(designs), np.concatenate(drops), np.concatenate(sidecars)


def window_end_sidecar(
    frame, cache, horizons: tuple[int, ...] = FIT_HORIZONS
) -> tuple[np.ndarray, np.ndarray]:
    """(end_to_crossing, cens drop) row-aligned with ts.build_window_bank.

    Replicates the bank's window enumeration (horizon-outer, stable-device-
    order inner, identical usable/finite masks) without duplicating the design
    matrix. The returned cens drop exists purely so callers can ASSERT
    alignment against the bank's own target before trusting the sidecar.
    """
    margins = {
        device_id: series.smooth_voltage - EOL_THRESHOLD
        for device_id, series in cache.devices.items()
    }
    order = np.argsort(frame.device, kind="stable")
    sidecars: list[np.ndarray] = []
    drops: list[np.ndarray] = []

    for horizon in horizons:
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            margin = margins.get(device_id)
            start = stop
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= last) & (cutoffs >= 0)
            if not usable.any():
                continue
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            drop = here[finite] - there[finite]
            end_fin = ends[usable][finite].astype(float)
            if crossing >= 0:
                crossed = (cutoffs[usable][finite] < crossing) & (end_fin >= crossing)
                drop = np.where(crossed, np.maximum(drop, here[finite]), drop)
                sidecars.append(end_fin - crossing)
            else:
                sidecars.append(np.full(end_fin.size, np.inf))
            drops.append(drop)

    return np.concatenate(sidecars), np.concatenate(drops)
