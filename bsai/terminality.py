"""Trajectory dynamics near the barrier: is this a stable low state or a terminal one?

The first-passage law asks *P(the path ever touches 2.4)*, and as the margin goes
to zero that probability goes to one whatever the drift regressor says. The
training data contains devices that contradict it directly: `docs/V10_FINDINGS.md`
records four with per-device floors at 2.402-2.416 V that "kiss the threshold for
years and never cross", and this branch's own false-positive profile found six
devices swapped in 29 to 48 of 48 scenarios that never die. A Brownian passage
reads small margin plus volatility as certainty; the survival evidence says the
opposite.

So the quantity the model needs is not the distance to the barrier. It is whether
the device has *entered* a terminal decline or is sitting in a stable low state
it has occupied for a long time. Those look identical in `(margin, drift, sigma)`
and different in the trajectory:

* a stable low device has an old floor it keeps returning to, many rebounds, a
  high fraction of recovery days, and little secular decline once temperature is
  taken out;
* a terminal device has recently *arrived* at its low, keeps setting new lows,
  rebounds weakly, and its decline survives temperature removal.

Everything here reads one device's smoothed daily grid up to one cutoff index and
nothing after it, so training and inference compute the same thing. Nothing here
is a probability; these are candidate signals to be judged by within-scenario
concordance on a margin-matched population, out of fold by building.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES
from .hazard import HORIZON_GRID

EOL_THRESHOLD = 2.4
NEW_LOW_TOLERANCE = 1e-4
# bsai/features.py: within a device, residual voltage tracks residual
# temperature at +0.00463 V/degC, positive in 100 % of 454 train devices.
TEMPERATURE_BETA = 0.00463
REFERENCE_TEMPERATURE = 20.0

NAMES = (
    # -- how long has it already been down there ---------------------------
    "frac_below_245_180",
    "frac_below_250_180",
    "longest_run_below_245_180",
    "current_run_below_245",
    # -- the device's own floor, and whether it is still moving -------------
    "floor_gap",
    "days_since_new_low",
    "new_lows_90",
    "floor_age",
    # -- rebound behaviour --------------------------------------------------
    "rebound_count_90",
    "rebound_mean_90",
    "rebound_max_90",
    "frac_up_days_90",
    # -- shape of the recent record ----------------------------------------
    "median_minus_min_90",
    "std_ratio_30_180",
    "slope_sign_consistency_90",
    # -- decline against the device's own early life ------------------------
    "plateau_drop",
    "plateau_drop_rate",
    # -- what survives temperature removal ----------------------------------
    "detrended_slope_180",
    "thermal_share_90",
    "residual_new_low_90",
)


def _valid_prefix(voltage: np.ndarray, index: int) -> np.ndarray:
    """The device's observed voltages up to and including ``index``."""
    window = voltage[: index + 1]
    return window[~np.isnan(window)]


def _window(voltage: np.ndarray, index: int, days: int) -> np.ndarray:
    lo = max(0, index - days + 1)
    window = voltage[lo : index + 1]
    return window[~np.isnan(window)]


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _trailing_run(mask: np.ndarray) -> int:
    run = 0
    for value in mask[::-1]:
        if not value:
            break
        run += 1
    return run


def _new_lows(values: np.ndarray, days: int) -> int:
    """Days in the last ``days`` that undercut every earlier day of the record."""
    if values.size < 3:
        return 0
    running = np.minimum.accumulate(values)
    previous = np.concatenate([[np.inf], running[:-1]])
    tail = slice(max(0, values.size - days), values.size)
    return int((values[tail] < previous[tail] - NEW_LOW_TOLERANCE).sum())


def _slope(values: np.ndarray) -> float:
    if values.size < 3:
        return np.nan
    x = np.arange(values.size, dtype=float)
    x = x - x.mean()
    denominator = float((x * x).sum())
    if denominator <= 0:
        return np.nan
    return float((x * (values - values.mean())).sum() / denominator)


def features_at(
    voltage: np.ndarray, temperature: np.ndarray, index: int
) -> list[float]:
    """The twenty trajectory signals at one cutoff, or NaN where undefined."""
    out: list[float] = []
    prefix = _valid_prefix(voltage, index)
    if prefix.size < 20:
        return [np.nan] * len(NAMES)
    current = float(prefix[-1])

    w180 = _window(voltage, index, 180)
    w90 = _window(voltage, index, 90)
    w30 = _window(voltage, index, 30)

    # -- time already spent near the barrier -------------------------------
    below245 = w180 < 2.45
    out.append(float(below245.mean()) if w180.size else np.nan)
    out.append(float((w180 < 2.50).mean()) if w180.size else np.nan)
    out.append(float(_longest_run(below245)))
    out.append(float(_trailing_run(below245)))

    # -- the device's own floor --------------------------------------------
    running_min = float(prefix.min())
    floor_position = int(np.argmin(prefix))
    out.append(current - running_min)
    out.append(float(prefix.size - 1 - floor_position))
    # A new low is a day whose value undercuts every *earlier* one. The running
    # minimum has to be shifted by one before slicing, or the first day of the
    # window is compared against nothing and always counts.
    out.append(float(_new_lows(prefix, 90)))
    # How much of the device's life the floor has stood, as a fraction.
    out.append(float(prefix.size - 1 - floor_position) / max(prefix.size, 1))

    # -- rebounds -----------------------------------------------------------
    if w90.size >= 5:
        steps = np.diff(w90)
        ups = steps[steps > 0]
        out.append(float(ups.size))
        out.append(float(ups.mean()) if ups.size else 0.0)
        out.append(float(ups.max()) if ups.size else 0.0)
        out.append(float((steps >= 0).mean()))
    else:
        out += [np.nan] * 4

    # -- shape of the recent record ----------------------------------------
    out.append(float(np.median(w90) - w90.min()) if w90.size else np.nan)
    std180 = float(w180.std()) if w180.size > 5 else np.nan
    std30 = float(w30.std()) if w30.size > 5 else np.nan
    out.append(std30 / std180 if std180 and np.isfinite(std180) and std180 > 1e-9 else np.nan)
    # Sign consistency: of the 7-day differences in the last 90 days, what
    # fraction point down? A terminal decline is consistent; a stable low state
    # wanders.
    if w90.size >= 14:
        steps7 = w90[7:] - w90[:-7]
        out.append(float((steps7 < 0).mean()))
    else:
        out.append(np.nan)

    # -- decline against the device's own early life ------------------------
    plateau = float(np.median(prefix[:90])) if prefix.size >= 120 else np.nan
    out.append(plateau - current if np.isfinite(plateau) else np.nan)
    out.append(
        (plateau - current) / max(prefix.size - 90, 1)
        if np.isfinite(plateau) else np.nan
    )

    # -- what survives temperature removal ----------------------------------
    temps = temperature[: index + 1]
    both = ~np.isnan(voltage[: index + 1]) & ~np.isnan(temps)
    lo = max(0, index - 180 + 1)
    mask = np.zeros(index + 1, dtype=bool)
    mask[lo:] = True
    use = both & mask
    if use.sum() >= 30:
        v = voltage[: index + 1][use]
        t = temps[use]
        # The *measured* physical constant, not a fitted one. Regressing voltage
        # on temperature inside a 180-day window is not safe here: the seasonal
        # cycle itself trends across the window, so a free coefficient happily
        # absorbs a chemical decline into the "thermal" term and reports a
        # detrended slope of zero for a device that is plainly dying.
        residual = v - TEMPERATURE_BETA * (t - REFERENCE_TEMPERATURE)
        out.append(_slope(residual))
        variance = float(v.var())
        explained = (TEMPERATURE_BETA ** 2) * float(t.var())
        out.append(explained / variance if variance > 1e-12 else np.nan)
        # Is the temperature-corrected series still making new lows?
        out.append(float(_new_lows(residual, 90)))
    else:
        out += [np.nan] * 3

    return out


BAND_LOW, BAND_HIGH = 0.0, 0.10
STD_RATIO_INDEX = NAMES.index("std_ratio_30_180")
VOLTAGE_COLUMN = FEATURE_NAMES.index("voltage")


def std_ratio(voltage: np.ndarray, index: int) -> float:
    """std of the smoothed daily series over 30 days, against its own 180."""
    w180 = _window(voltage, index, 180)
    w30 = _window(voltage, index, 30)
    if w180.size <= 5 or w30.size <= 5:
        return np.nan
    long_run = float(w180.std())
    if not np.isfinite(long_run) or long_run <= 1e-9:
        return np.nan
    return float(w30.std()) / long_run


@dataclass
class NearThresholdScorer:
    """Reorder only the near-threshold band, only by volatility expansion.

    The matched study behind this is in ``docs/FINAL_TERMINALITY.md``. Among
    batteries at the *same* low margin in the *same* scenario, the incumbent
    separates the ones that die within 42 days from the ones still alive after
    90 at concordance 0.574, and the ratio of the last 30 days' trajectory
    volatility to the same device's own 180-day baseline separates them at
    0.670 -- above chance in **five of five** building folds, worst fold 0.542.
    The absolute version of the same quantity, ``v_std_30``, manages two of five
    and a worst fold of 0.190, which is the building-fragility of a scale
    feature that ``docs/V11_TRANSFER_FINDINGS.md`` already documented for
    ``beta_30`` against ``beta_rise``.

    Two restrictions keep this honest. It is an equal-weight rank average, so
    there is no fitted parameter to overfit. And it is applied *only inside the
    band it was measured in*: rows outside 0 to 0.10 V of margin keep the
    incumbent's order exactly, and band rows are permuted only among
    themselves, so the curves that were on band rows stay there.
    """

    series: dict
    end_ordinal: dict
    weight: float = 0.5
    band: tuple[float, float] = (BAND_LOW, BAND_HIGH)
    bin_width: float = 0.01
    horizons: tuple[int, ...] = ()

    def _cutoff(self, device: str, remaining: float) -> tuple[np.ndarray, int] | None:
        entry = self.series.get(str(device))
        end = self.end_ordinal.get(str(device))
        if entry is None or end is None:
            return None
        voltage, origin = entry
        index = int(round(end - float(remaining) - origin))
        if index < 0:
            return None
        return voltage, min(index, voltage.size - 1)

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        from .rerank import centred_rank, decision_level

        horizons = self.horizons or HORIZON_GRID
        level = decision_level(grid, remaining, horizons)
        base = centred_rank(level)
        if devices is None:
            return base
        margin = features[:, VOLTAGE_COLUMN].astype(float) - EOL_THRESHOLD
        band = np.flatnonzero((margin >= self.band[0]) & (margin <= self.band[1]))
        if band.size < 3:
            return base
        ratio = np.full(band.size, np.nan)
        for position, row in enumerate(band):
            found = self._cutoff(devices[row], remaining[row])
            if found is not None:
                ratio[position] = std_ratio(found[0], found[1])
        if np.isfinite(ratio).sum() < 3:
            return base
        # The signal was measured *at matched margin*, so it is spent at matched
        # margin: rows are reordered only against others in the same margin bin.
        # Letting it compete across bins is what the band-wide version does, and
        # that reads better on the matched metric while costing 147 points
        # end to end -- the metric only ever compares within a bin, so it cannot
        # see the cross-bin damage. See docs/FINAL_TERMINALITY.md.
        out = base.copy()
        bins = np.floor(margin[band] / self.bin_width).astype(int)
        for value in np.unique(bins):
            rows = band[bins == value]
            if rows.size < 2 or np.isfinite(ratio[bins == value]).sum() < 2:
                continue
            blended = centred_rank(base[rows]) + self.weight * centred_rank(
                ratio[bins == value]
            )
            source = rows[np.argsort(-base[rows], kind="stable")]
            target = rows[np.lexsort((-base[rows], -blended))]
            out[target] = base[source]
        return out
