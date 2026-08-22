"""Causal features for the first-passage model.

Everything here reads a device's smoothed daily grid up to one cutoff index and
nothing beyond it. Training and inference call the same code, so a feature
cannot silently mean two different things on the two sides.

Three groups earn their place from measurement rather than habit:

* Temperature-compensated level and slopes. Within-device, residual voltage
  tracks residual temperature with beta = +0.00463 V/degC, positive in 100% of
  454 train devices. A 4.87 degC indoor annual swing is 0.023 V, which near the
  knee is about two weeks of remaining life.
* Seasonal temperature outlook. EOL incidence is 1.76x higher in Nov-Mar than
  in May-Sep, and the calendar is known at plan time. The expected temperature
  change across the planning window converts into an expected voltage shift
  through beta.
* Knee-onset statistics. The batteries the first prototype missed sat *higher*
  on the curve (median 2.503 V vs 2.457 V) and were declining *more slowly*
  (-0.00172 V/day vs -0.00210) yet crossed anyway. Level and slope alone cannot
  separate them; acceleration against a device's own baseline can.

Stale devices are kept rather than gated out. At a 21-day staleness cut, 15.9%
of alive device-scenarios had no usable row, and those carried 11% of all due
batteries -- auto-deferred, which is the most expensive mistake available. Their
due rate (0.012-0.019) is close to the fresh rate (0.024), so the honest move is
to hand the model the stale row along with its staleness and let it discount.

``DeviceView`` precomputes the prefix quantities once per device so that each
cutoff costs roughly constant time instead of rescanning the whole history.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rawdaily import RAW_FEATURE_NAMES
from .v12_rawany import RAW_ANY_FEATURE_NAMES

EOL_VOLTAGE = 2.4
TEMPERATURE_BETA = 0.00463
REFERENCE_TEMPERATURE = 20.0

SLOPE_WINDOWS = (7, 14, 30, 60, 90, 120, 180)
LEVEL_THRESHOLDS = (2.90, 2.80, 2.70, 2.65, 2.60, 2.55, 2.50, 2.45, 2.42)
CROSSING_WINDOWS = (14, 30, 60)
OUTLOOK_DAYS = (21, 42)

# Trailing windows for the within-day statistics. Short enough to catch a knee
# turning on, long enough that a device reporting intermittently still has days
# to average over.
SHAPE_WINDOWS = (7, 30)
SHAPE_BASELINE_WINDOW = 180

MIN_OBSERVATIONS = 20
FAR_CROSSING_DAYS = 900.0
RECENT_WINDOW = 90
TREND_WINDOW = 180

_EPOCH = pd.Timestamp("1970-01-01")
_MONTH_OF_DAY = None


def _month_lookup(day_ordinal: int) -> int:
    return (_EPOCH + pd.Timedelta(days=int(day_ordinal))).month - 1


def _build_names() -> list[str]:
    names = ["voltage", "voltage_compensated", "staleness", "observations"]
    names += [f"slope_{w}" for w in SLOPE_WINDOWS]
    names += [f"slope_comp_{w}" for w in SLOPE_WINDOWS]
    names += ["curvature_7_60", "curvature_30_120", "slope_ratio_14_90"]
    names += ["voltage_max", "drawdown", "voltage_min", "range_90"]
    names += [f"days_below_{t:.2f}" for t in LEVEL_THRESHOLDS]
    names += [f"crossing_{w}" for w in CROSSING_WINDOWS]
    names += ["crossing_comp_30"]
    names += ["temp_recent", "temp_lifetime", "temp_std", "temp_now"]
    names += [f"temp_outlook_{d}" for d in OUTLOOK_DAYS]
    names += [f"voltage_outlook_shift_{d}" for d in OUTLOOK_DAYS]
    names += ["season_sin", "season_cos"]
    names += ["gap_fraction_90", "age_days"]
    names += [
        "knee_slope_vs_history",
        "knee_trend_residual",
        "knee_worst_14d_drop",
        "knee_recent_vs_baseline",
    ]
    # Within-day shape. Everything above is some function of one number per day;
    # these are the only features that see inside the day, where the internal
    # resistance signal lives. Measured on the population the smoothed series
    # cannot rank at all, the within-day dV/dT separates due from not-due with
    # AUC 0.871 and a 3.8x median ratio.
    names += [f"beta_{w}" for w in SHAPE_WINDOWS]
    names += [f"v_std_{w}" for w in SHAPE_WINDOWS]
    names += [f"v_range_{w}" for w in SHAPE_WINDOWS]
    names += ["t_range_30"]
    # Ratios against the device's own long baseline. Absolute sensitivity varies
    # a lot between devices and buildings, so a rise relative to the unit's own
    # history transfers where a raw level does not -- which is exactly where V6's
    # calibration failed across buildings.
    names += ["beta_rise", "v_std_rise", "v_range_rise"]
    return names


FEATURE_NAMES = _build_names()
N_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# V12 "invariant" feature variant.
#
# Measured basis (docs/TRANSFER_STRESS.md): the absolute within-day SCALE
# features are building-bound (per-building median dispersion beta_30 x5.79,
# v_std_30 x14.6) while their rise ratios against the device's own baseline
# are not (beta_rise x1.32, v_std_rise x1.42), and the worst-transferring
# holdouts shift exactly the scale features (KS 0.63-0.66 on v_std_30/beta_30
# plus temp_lifetime/temp_now). The variant below replaces every absolute
# scale/temperature feature with a within-device contrast and adds the
# seasonal-recovery residual: how much of the voltage move over the last 30/60
# days is NOT explained by the device's own early-life dV/dT response. A
# healthy cell's voltage recovers with warming; a dying cell's does not.
#
# Everything here is additive. ``FEATURE_NAMES`` and the base ``feature_row``
# output are byte-identical to before; the variant is a column selection over
# an extended row, activated per call or through the module registry.
# ---------------------------------------------------------------------------

MID_BASELINE_WINDOW = 90
RECOVERY_WINDOWS = (30, 60)
# The per-device dV/dT estimate uses the first this-many pair-valid (voltage
# and temperature both present) days of the smoothed grid, causally: before
# the freeze point it uses whatever prefix exists, after it the estimate is
# frozen so late-life degradation cannot contaminate the "healthy response".
EARLY_BETA_WINDOW_DAYS = 180
EARLY_BETA_MIN_DAYS = 60
# Residual voltage tracks residual temperature at +0.00463 V/degC, positive in
# 100% of 454 train devices; a per-device estimate outside this band is noise.
EARLY_BETA_CLIP = (0.0, 0.02)
# Guard: with less than this within-window temperature variance (degC^2) the
# slope is unidentifiable and the fleet value is the honest estimate.
EARLY_BETA_MIN_TEMP_VAR = 0.25
BETA_Z_MIN_IQR = 1e-4


def _build_extra_names() -> list[str]:
    names = ["temp_now_delta", "temp_recent_delta"]
    names += [f"{stem}_rise_7" for stem in ("beta", "v_std", "v_range")]
    names += [f"{stem}_rise_mid" for stem in ("beta", "v_std", "v_range")]
    names += ["beta_z_30"]
    names += [f"recovery_residual_{w}" for w in RECOVERY_WINDOWS]
    return names


EXTRA_FEATURE_NAMES = _build_extra_names()
# The raw-daily channel (bsai/rawdaily.py): daily medians WITHOUT the 7-day
# rolling step, i.e. the ~3.5 days of lag the official smoothing imposes by
# construction. ~2.2 dues/scenario fail in window days 1-10 where this channel
# dominates. Plus the any-temperature channel (bsai/v12_rawany.py): 85% of
# dues on officially-dark rows have fresh readings OUTSIDE the 10-30 degC
# filter (outputs/roadblock_report.md, vii). Variant rows only; the base set
# stays pickle-compatible.
_RAW_GROUP = list(RAW_FEATURE_NAMES) + list(RAW_ANY_FEATURE_NAMES)
FEATURE_NAMES_EXTENDED = FEATURE_NAMES + EXTRA_FEATURE_NAMES + _RAW_GROUP

# Dropped from the invariant variant: the absolute-scale and absolute-
# temperature columns the transfer harness measured as fragile, plus
# ``observations`` (a building/export property, not a health signal).
INVARIANT_DROPPED = (
    "observations",
    "temp_recent",
    "temp_lifetime",
    "temp_std",
    "temp_now",
    "beta_7",
    "beta_30",
    "v_std_7",
    "v_std_30",
    "v_range_7",
    "v_range_30",
    "t_range_30",
)
# Frozen exactly as measured in transfer round 1 (outputs/v12_transfer.md):
# no raw channels, absolute scales dropped. Kept for the paper trail.
FEATURE_NAMES_INVARIANT = [
    name
    for name in FEATURE_NAMES_EXTENDED
    if name not in INVARIANT_DROPPED and name not in _RAW_GROUP
]

# Measured iteration (outputs/v12_transfer.md): dropping the absolute
# within-day scales cost ranking on 4/5 hard holdouts (mean PR-AUC 0.3975 vs
# cens 0.4278) and worsened the out-of-building level (LOO raw x1.447) -- the
# scales are building-bound AND load-bearing. ``invariant2`` keeps them
# alongside the ratios, replaces only the absolute-temperature group (highest
# KS shift on the worst-transferring holdouts, top-5 drift importance) with
# within-device contrasts, and carries the raw-daily channel.
INVARIANT2_DROPPED = (
    "observations",
    "temp_recent",
    "temp_lifetime",
    "temp_std",
    "temp_now",
)
FEATURE_NAMES_INVARIANT2 = [
    name for name in FEATURE_NAMES_EXTENDED if name not in INVARIANT2_DROPPED
]

FEATURE_VARIANTS: dict[str, list[str]] = {
    "base": FEATURE_NAMES,
    "extended": FEATURE_NAMES_EXTENDED,
    "invariant": FEATURE_NAMES_INVARIANT,
    "invariant2": FEATURE_NAMES_INVARIANT2,
}
_VARIANT_INDICES: dict[str, tuple[int, ...]] = {
    name: tuple(FEATURE_NAMES_EXTENDED.index(feature) for feature in features)
    for name, features in FEATURE_VARIANTS.items()
    if name not in ("base", "extended")
}


def variant_needs_raw(name: str) -> bool:
    """Does this variant's feature list include any raw-daily channel?"""
    return any(raw in FEATURE_VARIANTS[name] for raw in _RAW_GROUP)

# Module registry so call sites that cannot pass a parameter (the forecaster,
# ``build_training_frame``) still produce variant rows: set it once per
# process, or export BSAI_FEATURE_VARIANT before launching an existing tool.
_ACTIVE_VARIANT = os.environ.get("BSAI_FEATURE_VARIANT", "base")
if _ACTIVE_VARIANT not in FEATURE_VARIANTS:
    raise ValueError(
        f"BSAI_FEATURE_VARIANT={_ACTIVE_VARIANT!r} is not one of "
        f"{sorted(FEATURE_VARIANTS)}"
    )


def set_feature_variant(name: str) -> None:
    """Select which feature set ``feature_row`` emits by default."""
    global _ACTIVE_VARIANT
    if name not in FEATURE_VARIANTS:
        raise ValueError(f"unknown feature variant {name!r}")
    _ACTIVE_VARIANT = name


def active_feature_variant() -> str:
    return _ACTIVE_VARIANT


def active_feature_names() -> list[str]:
    return FEATURE_VARIANTS[_ACTIVE_VARIANT]


def fleet_climatology(
    temperature_by_device: dict[str, tuple[int, np.ndarray]],
) -> np.ndarray:
    """Mean smoothed temperature by calendar month, centred on its own mean.

    Built from whichever split is in hand, so the seasonal outlook adapts to the
    buildings actually being planned rather than to the training fleet.
    """
    totals = np.zeros(12)
    counts = np.zeros(12)
    for origin, values in temperature_by_device.values():
        valid = ~np.isnan(values)
        if not valid.any():
            continue
        days = origin + np.flatnonzero(valid)
        months = (_EPOCH + pd.to_timedelta(days, unit="D")).month.to_numpy() - 1
        np.add.at(totals, months, values[valid])
        np.add.at(counts, months, 1.0)
    profile = np.divide(totals, counts, out=np.full(12, np.nan), where=counts > 0)
    if np.isnan(profile).all():
        return np.zeros(12)
    profile = np.where(np.isnan(profile), np.nanmean(profile), profile)
    return profile - profile.mean()


@dataclass(frozen=True)
class FeatureContext:
    """Split-level constants shared by every feature row."""

    climatology: np.ndarray

    def outlook(self, day_ordinal: int, ahead: int) -> float:
        now = _month_lookup(day_ordinal)
        then = _month_lookup(day_ordinal + ahead)
        return float(self.climatology[then] - self.climatology[now])


class DeviceView:
    """Prefix quantities for one device, so each cutoff is cheap to evaluate."""

    __slots__ = (
        "voltage",
        "temperature",
        "compensated",
        "last_valid",
        "count",
        "running_max",
        "running_min",
        "first_below",
        "first_index",
        "temp_sum",
        "temp_sqsum",
        "temp_count",
        "temp_last_valid",
        "size",
        "_early_beta_sums",
        "_early_beta_freeze",
    )

    def __init__(self, voltage: np.ndarray, temperature: np.ndarray) -> None:
        self.size = int(voltage.shape[0])
        self.voltage = voltage
        self.temperature = temperature

        filled_temp = np.where(
            np.isnan(temperature), REFERENCE_TEMPERATURE, temperature
        )
        self.compensated = voltage - TEMPERATURE_BETA * (
            filled_temp - REFERENCE_TEMPERATURE
        )

        valid = ~np.isnan(voltage)
        indices = np.where(valid, np.arange(self.size), -1)
        self.last_valid = np.maximum.accumulate(indices)
        self.count = np.cumsum(valid)
        self.first_index = int(np.argmax(valid)) if valid.any() else -1

        big = np.where(valid, voltage, -np.inf)
        small = np.where(valid, voltage, np.inf)
        self.running_max = np.maximum.accumulate(big)
        self.running_min = np.minimum.accumulate(small)

        # A level crossing is permanent for feature purposes: once the device
        # has been below a threshold, it stays "has been below".
        self.first_below = {}
        for threshold in LEVEL_THRESHOLDS:
            hit = valid & (voltage < threshold)
            self.first_below[threshold] = int(np.argmax(hit)) if hit.any() else -1

        temp_valid = ~np.isnan(temperature)
        temp_values = np.where(temp_valid, temperature, 0.0)
        self.temp_sum = np.concatenate([[0.0], np.cumsum(temp_values)])
        self.temp_sqsum = np.concatenate([[0.0], np.cumsum(temp_values**2)])
        self.temp_count = np.concatenate([[0], np.cumsum(temp_valid)])
        temp_indices = np.where(temp_valid, np.arange(self.size), -1)
        self.temp_last_valid = np.maximum.accumulate(temp_indices)

        # Prefix sums for the per-device early-life dV/dT regression
        # (voltage ~ 1 + temperature + day, over days where both are present).
        # The day regressor absorbs the aging trend so the temperature slope is
        # the seasonal response, not degradation aliased onto season.
        pair = valid & temp_valid
        day = np.arange(self.size, dtype=float)
        t = np.where(pair, filled_temp, 0.0)
        v = np.where(pair, np.where(valid, voltage, 0.0), 0.0)  # NaN-safe select
        d = np.where(pair, day, 0.0)

        def prefix(values: np.ndarray) -> np.ndarray:
            return np.concatenate([[0.0], np.cumsum(values)])

        self._early_beta_sums = {
            "n": prefix(pair.astype(float)),
            "t": prefix(t),
            "v": prefix(v),
            "d": prefix(d),
            "tt": prefix(t * t),
            "tv": prefix(t * v),
            "td": prefix(t * d),
            "dd": prefix(d * d),
            "dv": prefix(d * v),
        }
        counts = self._early_beta_sums["n"][1:]
        freeze = int(
            np.searchsorted(counts, float(EARLY_BETA_WINDOW_DAYS), side="left")
        )
        self._early_beta_freeze = min(freeze, self.size - 1)

    def value_at_or_before(self, index: int) -> tuple[float, int]:
        if index < 0:
            return float("nan"), 999
        index = min(index, self.size - 1)
        position = int(self.last_valid[index])
        if position < 0:
            return float("nan"), 999
        return float(self.voltage[position]), index - position

    def compensated_at_or_before(self, index: int) -> float:
        if index < 0:
            return float("nan")
        index = min(index, self.size - 1)
        position = int(self.last_valid[index])
        if position < 0:
            return float("nan")
        return float(self.compensated[position])

    def temperature_at_or_before(self, index: int) -> float:
        if index < 0:
            return float("nan")
        index = min(index, self.size - 1)
        position = int(self.temp_last_valid[index])
        if position < 0:
            return float("nan")
        return float(self.temperature[position])

    def early_beta(self, index: int) -> float:
        """The device's own dV/dT, estimated causally from its early life.

        Regression of smoothed voltage on smoothed temperature with a linear
        day term (so aging drift does not alias onto season), over the first
        ``EARLY_BETA_WINDOW_DAYS`` pair-valid days, frozen thereafter. Before
        60 pair-valid days -- or when the temperature barely moved -- the fleet
        value ``TEMPERATURE_BETA`` is the honest estimate.
        """
        if index < 0:
            return TEMPERATURE_BETA
        j = min(min(index, self.size - 1), self._early_beta_freeze)
        sums = self._early_beta_sums
        n = float(sums["n"][j + 1])
        if n < EARLY_BETA_MIN_DAYS:
            return TEMPERATURE_BETA
        st, sv, sd = sums["t"][j + 1], sums["v"][j + 1], sums["d"][j + 1]
        stt, stv, std = sums["tt"][j + 1], sums["tv"][j + 1], sums["td"][j + 1]
        sdd, sdv = sums["dd"][j + 1], sums["dv"][j + 1]
        temp_var = stt / n - (st / n) ** 2
        if not np.isfinite(temp_var) or temp_var < EARLY_BETA_MIN_TEMP_VAR:
            return TEMPERATURE_BETA
        gram = np.array([[n, st, sd], [st, stt, std], [sd, std, sdd]])
        rhs = np.array([sv, stv, sdv])
        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return TEMPERATURE_BETA
        beta = float(solution[1])
        if not np.isfinite(beta):
            return TEMPERATURE_BETA
        return float(np.clip(beta, *EARLY_BETA_CLIP))

    def temp_stats(self, lo: int, hi: int) -> tuple[float, float]:
        lo = max(0, lo)
        hi = min(self.size - 1, hi)
        if hi < lo:
            return float("nan"), float("nan")
        n = int(self.temp_count[hi + 1] - self.temp_count[lo])
        if n == 0:
            return float("nan"), float("nan")
        total = self.temp_sum[hi + 1] - self.temp_sum[lo]
        square = self.temp_sqsum[hi + 1] - self.temp_sqsum[lo]
        mean = total / n
        variance = max(square / n - mean * mean, 0.0)
        return float(mean), float(np.sqrt(variance)) if n > 5 else float("nan")


def feature_row(
    view: DeviceView,
    index: int,
    day_ordinal: int,
    context: FeatureContext,
    shape=None,
    variant: str | None = None,
    raw=None,
    raw_any=None,
) -> list[float] | None:
    """Features at one cutoff, or None when there is nothing usable yet.

    ``variant`` selects the emitted feature set (``FEATURE_VARIANTS``); None
    defers to the module registry, whose default ("base") reproduces the
    historical output exactly. Row eligibility is identical across variants,
    so the training/scoring population never depends on the variant.

    ``raw`` and ``raw_any`` feed the raw-daily channels on variant rows: each a
    per-device callable ``day_ordinal -> sequence of floats or None`` (lengths
    RAW_FEATURE_NAMES / RAW_ANY_FEATURE_NAMES), e.g.
    ``functools.partial(raw_cache.features_at, device_id)``. None (or a None
    return) leaves those columns NaN, which the trees treat as missing; the
    base variant never reads them.
    """
    if index < 0 or index >= view.size:
        return None
    current, staleness = view.value_at_or_before(index)
    if not np.isfinite(current):
        return None
    observations = int(view.count[index])
    if observations < MIN_OBSERVATIONS:
        return None

    temp_now = view.temperature_at_or_before(index)
    compensated = view.compensated_at_or_before(index)

    row = [current, compensated, float(staleness), float(observations)]

    slopes = []
    for window in SLOPE_WINDOWS:
        past, _ = view.value_at_or_before(index - window)
        slopes.append((current - past) / window if np.isfinite(past) else np.nan)
    row += slopes

    comp_slopes = []
    for window in SLOPE_WINDOWS:
        past = view.compensated_at_or_before(index - window)
        comp_slopes.append((compensated - past) / window if np.isfinite(past) else np.nan)
    row += comp_slopes

    def slope(window: int) -> float:
        return slopes[SLOPE_WINDOWS.index(window)]

    row.append(slope(7) - slope(60))
    row.append(slope(30) - slope(120))
    fast, slow = slope(14), slope(90)
    row.append(
        fast / slow
        if np.isfinite(fast) and np.isfinite(slow) and abs(slow) > 1e-6
        else np.nan
    )

    highest = float(view.running_max[index])
    lowest = float(view.running_min[index])
    row.append(highest)
    row.append(highest - current)
    row.append(lowest)

    window_lo = max(0, index - RECENT_WINDOW + 1)
    recent = view.voltage[window_lo : index + 1]
    recent_valid = recent[~np.isnan(recent)]
    row.append(
        float(recent_valid.max() - recent_valid.min()) if recent_valid.size else np.nan
    )

    for threshold in LEVEL_THRESHOLDS:
        first = view.first_below[threshold]
        row.append(float(index - first) if 0 <= first <= index else -1.0)

    for window in CROSSING_WINDOWS:
        s = slope(window)
        row.append(
            min((current - EOL_VOLTAGE) / -s, FAR_CROSSING_DAYS)
            if np.isfinite(s) and s < -1e-5
            else FAR_CROSSING_DAYS
        )
    comp_slope_30 = comp_slopes[SLOPE_WINDOWS.index(30)]
    row.append(
        min((compensated - EOL_VOLTAGE) / -comp_slope_30, FAR_CROSSING_DAYS)
        if np.isfinite(comp_slope_30) and comp_slope_30 < -1e-5
        else FAR_CROSSING_DAYS
    )

    recent_mean, _ = view.temp_stats(index - 29, index)
    lifetime_mean, lifetime_std = view.temp_stats(0, index)
    row.append(recent_mean)
    row.append(lifetime_mean)
    row.append(lifetime_std)
    row.append(temp_now if np.isfinite(temp_now) else np.nan)

    outlooks = [context.outlook(day_ordinal, ahead) for ahead in OUTLOOK_DAYS]
    row += outlooks
    row += [TEMPERATURE_BETA * value for value in outlooks]

    day_of_year = (_EPOCH + pd.Timedelta(days=int(day_ordinal))).dayofyear
    row.append(float(np.sin(2 * np.pi * day_of_year / 365.25)))
    row.append(float(np.cos(2 * np.pi * day_of_year / 365.25)))

    row.append(float(np.isnan(recent).mean()) if recent.size else 1.0)
    row.append(float(index - view.first_index) if view.first_index >= 0 else np.nan)

    row += _knee_features(view, index, slope(30), current)
    row += _shape_features(shape, index)

    resolved = _ACTIVE_VARIANT if variant is None else variant
    if resolved == "base":
        return row
    if resolved not in FEATURE_VARIANTS:
        raise ValueError(f"unknown feature variant {resolved!r}")
    row += _invariant_extras(
        view, index, shape, current, temp_now, recent_mean, lifetime_mean
    )
    for adapter, names in ((raw, RAW_FEATURE_NAMES), (raw_any, RAW_ANY_FEATURE_NAMES)):
        values = adapter(day_ordinal) if adapter is not None else None
        if values is None:
            row += [np.nan] * len(names)
        else:
            row += [float(value) for value in values]
    if resolved == "extended":
        return row
    return [row[i] for i in _VARIANT_INDICES[resolved]]


def _shape_features(shape, index: int) -> list[float]:
    """Within-day statistics, and their rise against the device's own baseline."""
    if shape is None:
        return [np.nan] * (3 * len(SHAPE_WINDOWS) + 4)

    levels: list[float] = []
    for name in ("beta", "v_std", "v_range"):
        levels += [shape.trailing_mean(name, index, w) for w in SHAPE_WINDOWS]
    out = list(levels)
    out.append(shape.trailing_mean("t_range", index, 30))

    for name in ("beta", "v_std", "v_range"):
        recent = shape.trailing_mean(name, index, 30)
        baseline = shape.trailing_mean(name, index, SHAPE_BASELINE_WINDOW)
        if np.isfinite(recent) and np.isfinite(baseline) and abs(baseline) > 1e-9:
            out.append(recent / baseline)
        else:
            out.append(np.nan)
    return out


def _shape_ratio(shape, index: int, name: str, window: int, baseline: int) -> float:
    recent = shape.trailing_mean(name, index, window)
    base = shape.trailing_mean(name, index, baseline)
    if np.isfinite(recent) and np.isfinite(base) and abs(base) > 1e-9:
        return recent / base
    return np.nan


def _invariant_extras(
    view: DeviceView,
    index: int,
    shape,
    current: float,
    temp_now: float,
    temp_recent: float,
    temp_lifetime: float,
) -> list[float]:
    """Within-device contrasts replacing the building-bound absolute scales."""
    out: list[float] = []

    # Temperature enters only as a deviation from the device's own lifetime
    # mean: the level is HVAC/building identity, the deviation is season.
    for value in (temp_now, temp_recent):
        out.append(
            value - temp_lifetime
            if np.isfinite(value) and np.isfinite(temp_lifetime)
            else np.nan
        )

    if shape is None:
        out += [np.nan] * 7
    else:
        # Short and medium rise ratios against the device's own baselines.
        for name in ("beta", "v_std", "v_range"):
            out.append(_shape_ratio(shape, index, name, 7, SHAPE_BASELINE_WINDOW))
        for name in ("beta", "v_std", "v_range"):
            out.append(_shape_ratio(shape, index, name, 30, MID_BASELINE_WINDOW))
        # Within-day dV/dT in units of the device's own history: robust z of
        # the trailing 30-day mean against the causal prefix median/IQR.
        recent = shape.trailing_mean("beta", index, 30)
        median, iqr = shape.prefix_median_iqr("beta", index)
        if np.isfinite(recent) and np.isfinite(median) and np.isfinite(iqr):
            out.append((recent - median) / max(iqr, BETA_Z_MIN_IQR))
        else:
            out.append(np.nan)

    # Seasonal-recovery residual: the voltage move over the window minus what
    # the device's own early-life dV/dT says temperature should have moved it.
    # Near zero for a cell that still responds like its healthy self; strongly
    # negative for a cell that no longer recovers with warming.
    beta_dev = view.early_beta(index)
    for window in RECOVERY_WINDOWS:
        v_past, _ = view.value_at_or_before(index - window)
        t_past = view.temperature_at_or_before(index - window)
        if (
            np.isfinite(current)
            and np.isfinite(v_past)
            and np.isfinite(temp_now)
            and np.isfinite(t_past)
        ):
            out.append((current - v_past) - beta_dev * (temp_now - t_past))
        else:
            out.append(np.nan)
    return out


def _knee_features(
    view: DeviceView, index: int, slope_30: float, current: float
) -> list[float]:
    """How far the recent decline departs from this device's own baseline."""
    first = view.first_index
    if first < 0 or index - first < 60:
        return [np.nan, np.nan, np.nan, np.nan]

    first_value = float(view.voltage[first])
    span = index - first
    baseline = (current - first_value) / span if span > 0 else np.nan
    ratio = (
        slope_30 / baseline
        if np.isfinite(slope_30) and np.isfinite(baseline) and abs(baseline) > 1e-7
        else np.nan
    )

    trend_lo = max(0, index - TREND_WINDOW + 1)
    window = view.voltage[trend_lo : index + 1]
    mask = ~np.isnan(window)
    residual = np.nan
    if mask.sum() >= 10:
        x = np.flatnonzero(mask).astype(float)
        y = window[mask]
        if x.size > 3:
            coefficients = np.polyfit(x[:-1], y[:-1], 1)
            residual = float(y[-1] - np.polyval(coefficients, x[-1]))

    recent_lo = max(0, index - RECENT_WINDOW + 1)
    recent = view.voltage[recent_lo : index + 1]
    positions = np.flatnonzero(~np.isnan(recent))
    worst = np.nan
    if positions.size >= 3:
        values = recent[positions]
        targets = np.searchsorted(positions, positions - 14, side="right") - 1
        usable = targets >= 0
        if usable.any():
            worst = float(np.min(values[usable] - values[targets[usable]]))

    recent_value, _ = view.value_at_or_before(index - RECENT_WINDOW)
    recent_rate = (
        (current - recent_value) / RECENT_WINDOW if np.isfinite(recent_value) else np.nan
    )
    departure = (
        recent_rate - baseline
        if np.isfinite(recent_rate) and np.isfinite(baseline)
        else np.nan
    )
    return [ratio, residual, worst, departure]
