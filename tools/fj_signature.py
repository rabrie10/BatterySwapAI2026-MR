"""A device signature: what kind of battery is this, independent of where it sits.

Every auxiliary representation this project has tried -- volatility ratio,
trajectory templates, survival surprise -- improves ordering when margin is held
fixed and damages it across margins. The reason is always the same: the signal
describes the *state*, and V8's margin already describes the state better.

This module builds a different object. Not "how close is this battery to the
barrier" but "what kind of battery and environment is this", read over the
device's whole history up to the cutoff. If two cells at 0.05 V of margin belong
to regimes whose cells behave differently over the next six weeks, that is
information about the *mapping* from margin to risk, and it is the only kind of
information that can legally move a cross-margin pair.

The rules the features obey, all of them enforced by `tests/test_segment.py`:

* **Causal.** A signature at cutoff `c` reads `series[:c]` and nothing else.
* **No identity.** No building, no room, no device id, no EOL label, no lifetime.
  Test buildings are unseen, so a segment key that memorises buildings is worth
  nothing.
* **Long history, not current state.** Every feature is an average, a slope, a
  coupling or a dispersion over the device's whole observed life, with the one
  deliberate exception (`v_drop_life`) marked as margin-coupled so the analysis
  can drop it.

`v_now` is not a signature feature. It is the thing the signature is supposed to
re-interpret.
"""

from __future__ import annotations

import numpy as np

# (name, margin-coupled) -- a coupled feature carries the current voltage and so
# partly restates the margin; the experiment reports both with and without them.
SIGNATURE = (
    ("t_mean_life", False),
    ("t_std_life", False),
    ("t_p10_life", False),
    ("t_p90_life", False),
    ("t_amp_life", False),
    ("t_cold_frac", False),
    ("t_warm_frac", False),
    ("v_plateau", False),
    ("v_slope_life", False),
    ("v_curv_life", False),
    ("beta_life", False),
    ("beta_r2_life", False),
    ("v_std_detr_life", False),
    ("v_std_ratio_30_180", False),
    ("age_days", False),
    ("obs_frac", False),
    ("v_drop_life", True),
)
NAMES = tuple(name for name, _ in SIGNATURE)
COUPLED = tuple(name for name, flag in SIGNATURE if flag)
PLAIN = tuple(name for name, flag in SIGNATURE if not flag)

MIN_HISTORY = 120          # days of history before a signature is trusted
PLATEAU_DAYS = 30          # the device's own early-life level
COLD_C, WARM_C = 15.0, 25.0


def _slope(y: np.ndarray) -> float:
    """OLS slope per day, in mV/day, over a finite series."""
    good = np.isfinite(y)
    if good.sum() < 10:
        return np.nan
    x = np.flatnonzero(good).astype(float)
    v = y[good]
    x = x - x.mean()
    denominator = float(x @ x)
    return float(1000.0 * (x @ (v - v.mean())) / denominator) if denominator else np.nan


def signature_at(voltage: np.ndarray, temperature: np.ndarray, cutoff: int) -> np.ndarray:
    """The signature of one device read at one cutoff, from its past alone."""
    out = np.full(len(NAMES), np.nan)
    if cutoff < MIN_HISTORY:
        return out
    v = np.asarray(voltage[:cutoff], dtype=float)
    t = np.asarray(temperature[:cutoff], dtype=float)
    both = np.isfinite(v) & np.isfinite(t)
    if both.sum() < MIN_HISTORY // 2:
        return out
    index = {name: position for position, name in enumerate(NAMES)}

    warm = t[np.isfinite(t)]
    out[index["t_mean_life"]] = float(warm.mean())
    out[index["t_std_life"]] = float(warm.std())
    low, high = np.percentile(warm, [10, 90])
    out[index["t_p10_life"]] = float(low)
    out[index["t_p90_life"]] = float(high)
    out[index["t_amp_life"]] = float(high - low)
    out[index["t_cold_frac"]] = float((warm < COLD_C).mean())
    out[index["t_warm_frac"]] = float((warm > WARM_C).mean())

    finite_v = v[np.isfinite(v)]
    plateau = float(np.median(v[:PLATEAU_DAYS][np.isfinite(v[:PLATEAU_DAYS])])) \
        if np.isfinite(v[:PLATEAU_DAYS]).sum() >= 5 else float(np.nanmax(finite_v))
    out[index["v_plateau"]] = plateau
    out[index["v_drop_life"]] = plateau - float(finite_v[-1])

    out[index["v_slope_life"]] = _slope(v)
    half = cutoff // 2
    first, second = _slope(v[:half]), _slope(v[half:])
    out[index["v_curv_life"]] = second - first if np.isfinite(first) and np.isfinite(second) else np.nan

    # Voltage against temperature over the whole life: the device's own thermal
    # coupling, and how much of its variance the coupling explains.
    vv, tt = v[both], t[both]
    days = np.flatnonzero(both).astype(float)
    design = np.column_stack([np.ones(vv.size), days - days.mean(), tt - tt.mean()])
    solution, *_ = np.linalg.lstsq(design, vv, rcond=None)
    residual = vv - design @ solution
    out[index["beta_life"]] = float(solution[2])
    spread = float(vv.var())
    out[index["beta_r2_life"]] = float(1.0 - residual.var() / spread) if spread > 0 else np.nan
    out[index["v_std_detr_life"]] = float(residual.std())

    recent = v[max(cutoff - 30, 0):]
    longer = v[max(cutoff - 180, 0):]
    recent, longer = recent[np.isfinite(recent)], longer[np.isfinite(longer)]
    if recent.size >= 10 and longer.size >= 60 and longer.std() > 0:
        out[index["v_std_ratio_30_180"]] = float(recent.std() / longer.std())

    out[index["age_days"]] = float(cutoff)
    out[index["obs_frac"]] = float(np.isfinite(v).mean())
    return out
