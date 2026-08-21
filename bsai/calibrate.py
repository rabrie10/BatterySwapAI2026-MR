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
