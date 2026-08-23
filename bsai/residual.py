"""A 42-day decision-focused residual ranking score on top of V8.

The score is exactly

    f(x) = logit(p_V8) + g(x),    g linear in a fixed set of within-scenario ranks

and it is used *only* to reorder. ``bsai.rerank.RankRemapModel`` then hands V8's
own CDF curves out in the new order, so the per-scenario probability multiset,
the total risk mass and the expected-due budget are identical to V8's. Nothing
here produces a probability the planner ever sees.

Three things make this a *decision* model rather than another RUL model.

**The label is the decision.** A positive is an EOL record inside the 42-day
window. A reliable negative is a battery *observed* to survive past 42 days --
either it has a record later than that, or its observation window still has 42
days left. A battery whose window closes first is neither, and is dropped:
calling it safe is the systematic label noise that would teach the model to
like exactly the closing scenarios where precision is already worst.

**The weight is the evaluator's own arithmetic.** For each landmark the service
value is what the official cost model says servicing is worth against deferring
it, using only training EOL data and the published rates:

    effective   = the day the evaluator prices EOL at -- the record if there is
                  one, else the substitute end of life
    served      = 0.5 * max(effective - swap_day, 0)      early
                + 10  * max(swap_day - effective, 0)      late, if it died first
    deferred    = 10  * max(emergency_day - effective, 0) the emergency queue
    value       = deferred - served

`value` is large and positive for a battery that really dies inside the window
and large and negative for one whose substitute end of life is months away. It
is the quantity a mistake actually costs, so it is what the losses below weight
by -- rather than every landmark counting the same, which is what makes a
plain log-loss optimise the wrong thing on a 2 % positive rate.

**The capacity is deliberately tiny.** ``g`` is linear in eight signals, each
reduced to its within-scenario percentile rank. Ranks remove anything constant
inside a scenario -- season, the calendar, the remaining-observation window --
so the model cannot learn the axes that move *volume*, only the ones that move
*order*. Eight coefficients, fitted out of fold on five building-disjoint folds,
against 454 positives from about 82 devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES
from .hazard import HORIZON_GRID
from .rerank import DECISION_HORIZON, centred_rank, decision_level

# The evaluator's published rates, and the placement the planner actually uses
# (median swap on day 1 of the window, first emergency slot on day 48 -- both
# measured in tools/swap_ledger.py, neither fitted here).
EARLY_RATE = 0.5
LATE_RATE = 10.0
SWAP_DAY = 1.0
EMERGENCY_DAY = 48.0

# Fixed for every objective, so the comparison isolates the loss.
SIGNALS = (
    "voltage_compensated",
    "voltage_min",
    "crossing_30",
    "slope_comp_30",
    "beta_rise",
    "p07_over_p42",
    "rel_margin_room",
    "dwell_45_log",
)


def service_value(effective: np.ndarray) -> np.ndarray:
    """What the official cost model says one swap is worth, per landmark."""
    effective = np.asarray(effective, dtype=float)
    served = EARLY_RATE * np.maximum(effective - SWAP_DAY, 0.0) + LATE_RATE * np.maximum(
        SWAP_DAY - effective, 0.0
    )
    deferred = LATE_RATE * np.maximum(EMERGENCY_DAY - effective, 0.0)
    return deferred - served


def landmark_mask(days_to_eol: np.ndarray, remaining: np.ndarray, due: np.ndarray) -> np.ndarray:
    """Rows whose 42-day fate is actually observed.

    Positive: an EOL record inside the window. Reliable negative: a record after
    the window, or 42 days of observation still to come. Everything else is
    censored before the horizon and is excluded rather than called safe.
    """
    observed = np.isfinite(days_to_eol) | (np.asarray(remaining, dtype=float) >= DECISION_HORIZON)
    return np.asarray(due, dtype=bool) | observed


def signal_columns(
    features: np.ndarray,
    grid: np.ndarray,
    remaining: np.ndarray,
    devices: np.ndarray | None,
    room_of: dict | None,
) -> dict[str, np.ndarray]:
    """The eight raw signals, oriented so that larger always means riskier."""
    def column(name: str) -> np.ndarray:
        return features[:, FEATURE_NAMES.index(name)].astype(float)

    index42 = list(HORIZON_GRID).index(DECISION_HORIZON)
    p42 = np.clip(grid[:, index42], 1e-9, 1.0)
    p07 = grid[:, list(HORIZON_GRID).index(7)]
    dwell = column("days_below_2.45")
    margin = column("voltage") - 2.4

    out = {
        # Distance to a temperature-corrected barrier: the reading moves
        # +0.00463 V per degree, the barrier is a chemical state.
        "voltage_compensated": -column("voltage_compensated"),
        "voltage_min": -column("voltage_min"),
        "crossing_30": -column("crossing_30"),
        "slope_comp_30": -column("slope_comp_30"),
        "beta_rise": column("beta_rise"),
        # How much of the model's certainty is barrier saturation rather than
        # a decline it has actually seen.
        "p07_over_p42": -(p07 / p42),
        "dwell_45_log": np.log1p(np.maximum(dwell, 0.0)),
    }
    out["rel_margin_room"] = -_room_contrast(margin, remaining, devices, room_of)
    return out


def _room_contrast(
    value: np.ndarray,
    remaining: np.ndarray,
    devices: np.ndarray | None,
    room_of: dict | None,
) -> np.ndarray:
    """value minus the median of the same room's other rows in this scenario.

    A voltage that is low because the room is cold moves with its roommates; a
    battery that is genuinely dying does not. Within-scenario and within-room,
    so no building identity is learned. NaN where the room has no peers, which
    the rank transform sends to the bottom.
    """
    out = np.full(value.shape[0], np.nan)
    if devices is None or room_of is None:
        return out
    rooms = np.asarray([str(room_of.get(str(d), "?")) for d in devices])
    for room in np.unique(rooms):
        rows = np.flatnonzero(rooms == room)
        if rows.size < 3:
            continue
        values = value[rows]
        finite = np.isfinite(values)
        if finite.sum() < 3:
            continue
        for position, row in enumerate(rows):
            if not finite[position]:
                continue
            others = np.delete(values[finite], np.flatnonzero(rows[finite] == row)[0])
            out[row] = values[position] - float(np.median(others))
    return out


def design(
    features: np.ndarray,
    grid: np.ndarray,
    remaining: np.ndarray,
    devices: np.ndarray | None,
    room_of: dict | None,
    names: tuple[str, ...] = SIGNALS,
) -> np.ndarray:
    """One row per landmark, each signal as a centred within-group rank."""
    columns = signal_columns(features, grid, remaining, devices, room_of)
    return np.column_stack([centred_rank(columns[name]) for name in names])


@dataclass
class ResidualScorer:
    """``logit(p_V8) + w . rank(x)``, evaluated on one scenario at a time."""

    weights: np.ndarray
    room_of: dict | None = None
    names: tuple[str, ...] = SIGNALS
    horizons: tuple[int, ...] = HORIZON_GRID
    objective: str = ""

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        level = np.clip(decision_level(grid, remaining, self.horizons), 1e-9, 1 - 1e-9)
        anchor = np.log(level / (1.0 - level))
        matrix = design(features, grid, remaining, devices, self.room_of, self.names)
        return anchor + matrix @ np.asarray(self.weights, dtype=float)


# --------------------------------------------------------------------------
# the three objectives
# --------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fit_pointwise(
    matrix: np.ndarray,
    anchor: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    *,
    l2: float,
    focal_gamma: float = 0.0,
) -> np.ndarray:
    """Weighted log-loss on ``sigmoid(anchor + X w)``; focal when gamma > 0.

    The anchor enters with its coefficient pinned at one, which is what makes
    this a *residual* fit: ``w`` can only correct V8, never replace it.
    """
    from scipy.optimize import minimize

    weight = weight / max(weight.sum(), 1e-12) * weight.size

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = anchor + matrix @ w
        p = _sigmoid(z)
        pt = np.where(label > 0.5, p, 1.0 - p)
        pt = np.clip(pt, 1e-12, 1.0)
        modulator = (1.0 - pt) ** focal_gamma if focal_gamma else 1.0
        loss = float((weight * modulator * -np.log(pt)).mean()) + l2 * float(w @ w)
        # d/dz of -log(pt) is (p - y); the focal modulator adds a second term.
        base = p - label
        if focal_gamma:
            grad_z = modulator * base + focal_gamma * (1.0 - pt) ** (focal_gamma - 1.0) * (
                -np.log(pt)
            ) * np.where(label > 0.5, -1.0, 1.0) * p * (1.0 - p)
        else:
            grad_z = base
        gradient = matrix.T @ (weight * grad_z) / weight.size + 2.0 * l2 * w
        return loss, gradient

    result = minimize(
        objective, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 800},
    )
    return result.x


def fit_pairwise(
    matrix: np.ndarray,
    anchor: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    weight: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    """P(pos above neg) = sigmoid((a_pos - a_neg) + w . (x_pos - x_neg))."""
    from scipy.optimize import minimize

    if positives.size == 0:
        return np.zeros(matrix.shape[1])
    difference = matrix[positives] - matrix[negatives]
    offset = anchor[positives] - anchor[negatives]
    weight = weight / max(weight.sum(), 1e-12) * weight.size

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        margin = offset + difference @ w
        loss = float((weight * np.logaddexp(0.0, -margin)).mean()) + l2 * float(w @ w)
        gradient = -(difference.T @ (weight * _sigmoid(-margin))) / weight.size
        return loss, gradient + 2.0 * l2 * w

    result = minimize(
        objective, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 800},
    )
    return result.x


def build_pairs(
    scenario: np.ndarray,
    battery: np.ndarray,
    due: np.ndarray,
    anchor: np.ndarray,
    value: np.ndarray,
    usable: np.ndarray,
    *,
    delta: float,
    rng: np.random.Generator,
    max_per_scenario: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ambiguous within-scenario (due, survivor) pairs, weighted by what the
    wrong order costs.

    Only pairs V8 already scores within ``delta`` logits of each other are kept:
    the rest are decisions the incumbent is already making correctly or is
    hopeless on, and spending capacity there is how a model ends up re-learning
    that a 2.95 V cell is healthy. The weight is the service-value gap, so a
    pair whose misordering costs 300 points counts three hundred times a pair
    whose misordering costs one. Each due *device* then carries the same total
    weight, because the 48 scenarios overlap by about 85 % and one device can
    contribute up to 48 near-copies.
    """
    pos_out: list[np.ndarray] = []
    neg_out: list[np.ndarray] = []
    weight_out: list[np.ndarray] = []
    for index in np.unique(scenario):
        rows = np.flatnonzero((scenario == index) & usable)
        if rows.size < 2:
            continue
        positives = rows[due[rows]]
        negatives = rows[~due[rows]]
        if positives.size == 0 or negatives.size == 0:
            continue
        gap = np.abs(anchor[positives][:, None] - anchor[negatives][None, :])
        pi, ni = np.nonzero(gap <= delta)
        if pi.size == 0:
            continue
        if pi.size > max_per_scenario:
            keep = rng.choice(pi.size, max_per_scenario, replace=False)
            pi, ni = pi[keep], ni[keep]
        pos_out.append(positives[pi])
        neg_out.append(negatives[ni])
        weight_out.append(np.maximum(value[positives[pi]] - value[negatives[ni]], 0.0))
    if not pos_out:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    pos = np.concatenate(pos_out)
    neg = np.concatenate(neg_out)
    weight = np.concatenate(weight_out)
    devices, inverse = np.unique(battery[pos], return_inverse=True)
    per_device = np.bincount(inverse, weights=weight, minlength=devices.size)
    weight = weight / np.maximum(per_device[inverse], 1e-9)
    return pos, neg, weight


@dataclass
class OofResidualScorer:
    """Dispatch each row to the weights fitted without its own building.

    The same discipline ``bsai/validation.py`` applies to the passage model: a
    device is never ranked by coefficients that saw its building. Rows from a
    building with no fitted fold fall back to the incumbent order, which is the
    identity, rather than to somebody else's weights.
    """

    by_building: dict
    building_of: dict
    room_of: dict | None = None
    names: tuple[str, ...] = SIGNALS
    horizons: tuple[int, ...] = HORIZON_GRID
    objective: str = ""

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        if devices is None:
            raise ValueError("out-of-fold residual scoring needs the device of each row")
        level = np.clip(decision_level(grid, remaining, self.horizons), 1e-9, 1 - 1e-9)
        out = np.log(level / (1.0 - level))
        matrix = design(features, grid, remaining, devices, self.room_of, self.names)
        buildings = np.asarray(
            [str(self.building_of.get(str(d), "")) for d in devices], dtype=object
        )
        for building in np.unique(buildings):
            weights = self.by_building.get(str(building))
            if weights is None:
                continue
            mask = buildings == building
            out[mask] = out[mask] + matrix[mask] @ np.asarray(weights, dtype=float)
        return out
