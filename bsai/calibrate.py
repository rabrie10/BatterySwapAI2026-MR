"""Calibration along the remaining-observation axis.

Measured out-of-fold at scenario cutoffs, the V7 model's predicted count of due
batteries against the realised count:

    scenarios 0-15   predicted  7.13   actual 13.25   ratio 0.54   remaining 282d
    scenarios 16-31  predicted  8.62   actual  8.56   ratio 1.01   remaining 170d
    scenarios 32-47  predicted 10.75   actual  6.56   ratio 1.64   remaining  58d
    all              predicted  8.83   actual  9.46   ratio 0.93

The pooled 0.93 hides two large errors of opposite sign. The model predicts the
*most* failures where there are the *fewest*: it has inverted the trend. V6 had
the same disease at a different offset (0.87 / 1.46 / 2.21), and this is finally
the explanation for why every global knob in the V6 sweeps traded the opening
scenarios against the closing ones and landed inside the noise floor -- **a
single scalar cannot correct a bias that changes sign.**

The axis is the remaining observation window, ``end_time - prediction_origin``.
That is known at plan time from ``locations.end_time``, so using it leaks
nothing, and it is a property of the scenario rather than of the calendar, so it
transfers to a split with its own export date.

**Known risk: the axis is confounded on train.** The 48 scenarios run
chronologically from September to July, so the remaining observation window and
the calendar month move together by construction. End-of-life incidence is about
1.55x higher in November-March than in May-September on this data, so part of
what this correction absorbs may be season rather than censoring.

Two things argue for keeping the remaining-observation axis anyway. It correlates
far more strongly with the realised count (Spearman 0.839 against 0.557 for
distance from midwinter), and the low end is mechanically explained: a scenario
with 12 days of observation left can only record failures in 12 of its 42 window
days, and 6.56 x 12/42 = 1.9 against a realised 2.33. That part is real
censoring and transfers to any split.

The residual risk is that the public and private splits cover a different part of
the year. They are almost certainly generated with the same temporal structure --
48 scenarios, the same window, chronological -- in which case the two axes stay
confounded in the same way and the correction transfers. If a future submission
shows this correction hurting, the month of the window is the alternative axis to
try, and it is the more physical one.

The correction is deliberately coarse -- a handful of buckets, fitted on
out-of-fold predictions, smoothed and clamped. It has to survive a change of
buildings, and a flexible fit would not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Wide buckets. The bias is smooth in remaining observation days and there are
# only 48 train scenarios, so anything finer would fit noise.
DEFAULT_EDGES = (0.0, 45.0, 90.0, 150.0, 220.0, 300.0, 1e9)

# A correction outside this range is far more likely to be an artefact of a thin
# bucket than a real effect of that size.
MIN_FACTOR = 0.35
MAX_FACTOR = 2.75


@dataclass
class RemainingCalibration:
    """Multiplicative correction on the predicted CDF, keyed by remaining days."""

    edges: tuple[float, ...] = DEFAULT_EDGES
    factors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.factors:
            self.factors = tuple(1.0 for _ in range(len(self.edges) - 1))

    @property
    def centres(self) -> np.ndarray:
        edges = np.asarray(self.edges, dtype=float)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # The open-ended top bucket has no meaningful midpoint; anchor it just
        # past the last finite edge so interpolation stays flat above it.
        centres[-1] = edges[-2] * 1.25
        return centres

    def factor_for(self, remaining: np.ndarray) -> np.ndarray:
        """Interpolate between bucket centres so the correction is continuous.

        A step function would put a discontinuity inside the planning window --
        two batteries a day apart in observation window would get materially
        different risk, which is not a real effect.
        """
        remaining = np.clip(np.asarray(remaining, dtype=float), 0.0, None)
        return np.interp(remaining, self.centres, np.asarray(self.factors, dtype=float))

    def apply(self, grid: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        factor = self.factor_for(remaining)[:, None]
        return np.clip(grid * factor, 0.0, 1.0)

    @classmethod
    def fit(
        cls,
        remaining: np.ndarray,
        predicted: np.ndarray,
        actual: np.ndarray,
        *,
        edges: tuple[float, ...] = DEFAULT_EDGES,
        min_events: int = 25,
    ) -> "RemainingCalibration":
        """One factor per bucket: realised events over predicted mass.

        Buckets too thin to estimate are left uncorrected rather than fitted to
        noise, and every factor is clamped.
        """
        remaining = np.asarray(remaining, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        factors: list[float] = []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (remaining >= low) & (remaining < high)
            mass = float(predicted[inside].sum())
            events = float(actual[inside].sum())
            if inside.sum() == 0 or mass <= 1e-9 or events < min_events:
                factors.append(1.0)
                continue
            factors.append(float(np.clip(events / mass, MIN_FACTOR, MAX_FACTOR)))
        return cls(edges=tuple(edges), factors=tuple(factors))

    def describe(self) -> str:
        lines = []
        for (low, high), factor in zip(
            zip(self.edges[:-1], self.edges[1:]), self.factors
        ):
            span = f"{low:>6.0f}-{high:>6.0f}" if high < 1e8 else f"{low:>6.0f}+     "
            lines.append(f"  remaining {span} d   x{factor:.3f}")
        return "\n".join(lines)


@dataclass
class DwellAdjust:
    """Survival-evidence correction for cells sitting near the threshold.

    A cell freshly below 2.45 V mostly crosses within the window (realised
    0.80 at predicted 0.81), but the longer it has dipped *without* crossing
    the less likely the next 42 days are to record an EOL: realised 0.29 at
    15-42 days of dwell and 0.18 at 43-90, while the model's confidence RISES
    with dwell (0.84, 0.90) because the volatility it observes feeds the
    passage arithmetic. Four batteries with per-device voltage floors just
    above 2.40 generate a wasted swap in nearly every scenario this way.

    The pattern is fleet-wide (19 batteries, 11 buildings), physical
    (survival evidence), and independent of building identity, so it is
    corrected here with one multiplicative factor per dwell band, fitted
    out-of-fold, applied only to rows within ``margin_cap`` of the threshold.
    """

    margin_cap: float = 0.05
    edges: tuple[float, ...] = (0.0, 14.0, 42.0, 90.0, 1e9)
    factors: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    min_factor: float = 0.1
    max_factor: float = 2.5

    def factor_for(self, margin: np.ndarray, dwell: np.ndarray) -> np.ndarray:
        margin = np.asarray(margin, dtype=float)
        dwell = np.asarray(dwell, dtype=float)
        inside = (margin < self.margin_cap) & (dwell >= 0.0)
        band = np.clip(
            np.searchsorted(np.asarray(self.edges, dtype=float), dwell, side="right") - 1,
            0,
            len(self.factors) - 1,
        )
        factors = np.asarray(self.factors, dtype=float)[band]
        return np.where(inside, factors, 1.0)

    def apply(self, grid: np.ndarray, margin: np.ndarray, dwell: np.ndarray) -> np.ndarray:
        if grid.shape[0] == 0:
            return grid
        factor = self.factor_for(margin, dwell)[:, None]
        return np.clip(grid * factor, 0.0, 1.0)

    @classmethod
    def fit(
        cls,
        margin: np.ndarray,
        dwell: np.ndarray,
        predicted: np.ndarray,
        actual: np.ndarray,
        *,
        margin_cap: float = 0.05,
        edges: tuple[float, ...] = (0.0, 14.0, 42.0, 90.0, 1e9),
        min_events: int = 8,
    ) -> "DwellAdjust":
        margin = np.asarray(margin, dtype=float)
        dwell = np.asarray(dwell, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        inside = (margin < margin_cap) & (dwell >= 0.0)
        factors: list[float] = []
        template = cls(margin_cap=margin_cap, edges=edges)
        for low, high in zip(edges[:-1], edges[1:]):
            band = inside & (dwell >= low) & (dwell < high)
            mass = float(predicted[band].sum())
            events = float(actual[band].sum())
            if band.sum() < 20 or events < min_events or mass <= 1e-9:
                factors.append(1.0)
                continue
            factors.append(
                float(np.clip(events / mass, template.min_factor, template.max_factor))
            )
        return cls(margin_cap=margin_cap, edges=tuple(edges), factors=tuple(factors))

    def describe(self) -> str:
        parts = []
        for (low, high), factor in zip(zip(self.edges[:-1], self.edges[1:]), self.factors):
            span = f"{low:.0f}-{high:.0f}" if high < 1e8 else f"{low:.0f}+"
            parts.append(f"dwell {span}d x{factor:.2f}")
        return f"margin<{self.margin_cap:.3f}: " + "  ".join(parts)


# Reliability bands. Three because the bias changes sign along this axis and
# 454 due rows cannot support more; the isotonic within each band does the
# rest of the pooling.
RELIABILITY_EDGES = (0.0, 90.0, 220.0, 1e9)


@dataclass
class ReliabilityCalibration:
    """Isotonic reliability repair within coarse remaining-observation bands.

    The multiplicative correction balanced the aggregate due count per band by
    scaling every probability with one factor, which pushed mid-probability
    batteries into the top bucket: measured at scenario cutoffs, rows predicted
    above 0.7 realise 0.36 while rows predicted 0.5-0.7 realise 0.41. The
    planner prices swap-versus-defer off these numbers, so the top of the
    distribution being anti-monotone turns directly into wasted swaps in the
    scenarios where a wasted swap is most expensive.

    This correction maps predicted probability to realised frequency with an
    isotonic fit inside each remaining-observation band, so the aggregate count
    and the shape are repaired together. It is applied as a per-row rescale of
    the whole horizon grid, which preserves monotonicity across horizons.
    """

    edges: tuple[float, ...] = RELIABILITY_EDGES
    # Per band: (grid of predicted probabilities, isotonic values at that grid).
    curves: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] = ()
    horizon_column: int = 11  # index of the 42-day horizon in HORIZON_GRID
    min_factor: float = 0.2
    max_factor: float = 3.0

    def _band_of(self, remaining: np.ndarray) -> np.ndarray:
        edges = np.asarray(self.edges, dtype=float)
        remaining = np.clip(np.asarray(remaining, dtype=float), 0.0, None)
        return np.clip(np.searchsorted(edges, remaining, side="right") - 1, 0, len(self.curves) - 1)

    def factor_for(self, probability: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        bands = self._band_of(remaining)
        mapped = np.empty_like(probability)
        for band in np.unique(bands):
            xs, ys = self.curves[int(band)]
            mask = bands == band
            mapped[mask] = np.interp(probability[mask], xs, ys)
        factor = mapped / np.maximum(probability, 1e-4)
        return np.clip(factor, self.min_factor, self.max_factor)

    def apply(self, grid: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        if grid.shape[0] == 0 or not self.curves:
            return grid
        column = min(self.horizon_column, grid.shape[1] - 1)
        factor = self.factor_for(grid[:, column], remaining)[:, None]
        return np.clip(grid * factor, 0.0, 1.0)

    @classmethod
    def fit(
        cls,
        remaining: np.ndarray,
        predicted: np.ndarray,
        actual: np.ndarray,
        *,
        edges: tuple[float, ...] = RELIABILITY_EDGES,
        min_events: int = 25,
        horizon_column: int = 11,
    ) -> "ReliabilityCalibration":
        from sklearn.isotonic import IsotonicRegression

        remaining = np.clip(np.asarray(remaining, dtype=float), 0.0, None)
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        curves: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (remaining >= low) & (remaining < high)
            events = float(actual[inside].sum())
            if inside.sum() < 200 or events < min_events:
                # Too thin to reshape: identity curve.
                curves.append(((0.0, 1.0), (0.0, 1.0)))
                continue
            iso = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            iso.fit(predicted[inside], actual[inside])
            xs = np.unique(
                np.concatenate(
                    [[0.0, 1.0], np.quantile(predicted[inside], np.linspace(0, 1, 41))]
                )
            )
            ys = iso.predict(xs)
            curves.append((tuple(map(float, xs)), tuple(map(float, ys))))
        return cls(edges=tuple(edges), curves=tuple(curves), horizon_column=horizon_column)

    def describe(self) -> str:
        lines = []
        for (low, high), (xs, ys) in zip(zip(self.edges[:-1], self.edges[1:]), self.curves):
            span = f"{low:>5.0f}-{high:>5.0f}" if high < 1e8 else f"{low:>5.0f}+"
            probes = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
            mapped = np.interp(probes, xs, ys)
            pairs = "  ".join(f"{p:.2f}->{m:.2f}" for p, m in zip(probes, mapped))
            lines.append(f"  remaining {span} d: {pairs}")
        return "\n".join(lines)
