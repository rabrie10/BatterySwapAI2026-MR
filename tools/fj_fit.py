"""Fit a residual precision reranker with a pairwise objective, building-disjoint.

The deployed transformation is order-only (``bsai/rerank.py``), so the thing to
learn is an *order*, not a probability. The objective is therefore the pairwise
one: inside one scenario, take a battery that really was due within 42 days and
one that is known to survive past 42 days, and learn to put the first above the
second. Restricting to pairs the incumbent already scores similarly spends the
capacity on the decision boundary instead of re-learning that a 2.95 V cell is
healthy.

Three constructions keep the capacity where the evidence is:

* **Within-scenario standardisation.** Every signal is centred on its own
  scenario's median and scaled by that scenario's IQR. Anything constant within
  a scenario -- season, the calendar, the scenario's remaining-observation
  window -- becomes exactly zero and cannot be learned. That is a feature: those
  axes move *volume*, which ``bsai/calibrate.py`` already handles and which V9
  and V19 both proved is not where the remaining points are.
* **Censoring-correct labels.** A positive is an EOL record inside 42 days. A
  negative is a battery *observed* to survive 42 days past the cutoff. A row
  whose observation window closes first is neither, and is dropped rather than
  called safe.
* **Device-level weights.** The 48 scenarios overlap by about 85 %, so one
  device contributes up to 48 near-copies. Each positive device carries the same
  total weight regardless of how many rows it produced.

    python tools/fj_fit.py --signals rel_temp_room,dwell_45_log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.optimize import minimize  # noqa: E402

HORIZON = 42.0


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def standardise_within(
    scenario: np.ndarray, value: np.ndarray, subset: np.ndarray | None = None
) -> np.ndarray:
    """Centred percentile rank inside each scenario, in [-0.5, +0.5].

    A z-score is the wrong normaliser here. The decision population is the ~15
    riskiest of about 414 alive batteries, and against the whole fleet's median
    and IQR every one of them is an outlier -- ``observations`` or ``slope_30``
    for the candidates all land past the clip and the ordering *inside* the
    candidates, which is the only thing an order-only remap can use, is
    flattened away. A rank is monotone-invariant, needs no clip, and removes any
    scenario-level shift or scale exactly, so anything constant within a
    scenario (the calendar, the season, the remaining window) is exactly zero
    and cannot be learned.

    ``subset`` restricts the ranking to the candidate rows, so the resolution is
    spent where the swaps are.
    """
    out = np.zeros(value.shape[0])
    for index in np.unique(scenario):
        rows = np.flatnonzero(scenario == index)
        if subset is not None:
            rows = rows[subset[rows]]
        if rows.size < 4:
            continue
        column = value[rows]
        good = np.isfinite(column)
        if good.sum() < 4:
            continue
        order = np.argsort(np.argsort(column[good], kind="stable"), kind="stable")
        out[rows[good]] = order / max(good.sum() - 1, 1) - 0.5
    return out


def label_and_mask(frame) -> tuple[np.ndarray, np.ndarray]:
    """(positive, usable). A row is usable only if its 42-day fate is observed."""
    positive = frame.due
    observed = np.isfinite(frame.days_to_eol) | (frame.remaining >= HORIZON)
    return positive, positive | observed


def build_pairs(
    frame,
    anchor: np.ndarray,
    usable: np.ndarray,
    *,
    top_k: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Comparable within-scenario (due, survivor) pairs and their weights."""
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero((frame.scenario == index) & usable)
        if rows.size < 2:
            continue
        order = rows[np.argsort(-anchor[rows], kind="stable")][:top_k]
        pos = order[frame.due[order]]
        neg = order[~frame.due[order]]
        if pos.size == 0 or neg.size == 0:
            continue
        gap = np.abs(anchor[pos][:, None] - anchor[neg][None, :])
        pi, ni = np.nonzero(gap <= delta)
        if pi.size == 0:
            continue
        positives.append(pos[pi])
        negatives.append(neg[ni])
    if not positives:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    pos = np.concatenate(positives)
    neg = np.concatenate(negatives)
    # One unit of weight per distinct due device, spread over its pairs.
    devices = frame.battery[pos]
    unique, inverse, counts = np.unique(devices, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    return pos, neg, weight * (pos.size / weight.sum())


def fit_weights(
    design: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    weight: np.ndarray,
    *,
    l2: float,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise logistic ranking on top of a fixed anchor.

    ``P(pos above neg) = sigmoid((a_pos - a_neg) + w . (s_pos - s_neg))``. The
    anchor enters with its coefficient pinned at one, so the weights are a
    *correction* to the incumbent order rather than a fresh ranker, and a
    weight's size is directly readable as how many places of V8's own ordering
    one unit of the signal is allowed to move a battery.
    """
    if pos.size == 0:
        return np.zeros(design.shape[1])
    difference = design[pos] - design[neg]
    offset = (
        np.zeros(pos.size) if anchor is None else anchor[pos] - anchor[neg]
    )

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        margin = offset + difference @ w
        # log(1 + exp(-margin)), evaluated without overflowing.
        loss = np.logaddexp(0.0, -margin)
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(margin, -60, 60)))
        value = float((weight * loss).sum() / weight.sum() + l2 * w @ w)
        gradient = -(difference * (weight * sigmoid)[:, None]).sum(axis=0) / weight.sum()
        return value, gradient + 2.0 * l2 * w

    result = minimize(
        objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 500},
    )
    return result.x


def design_matrix(
    frame,
    signals: dict[str, np.ndarray],
    names: list[str],
    subset: np.ndarray | None = None,
) -> np.ndarray:
    return np.column_stack(
        [standardise_within(frame.scenario, signals[name], subset) for name in names]
    )


def oof_scores(
    frame,
    anchor: np.ndarray,
    design: np.ndarray,
    usable: np.ndarray,
    partitions: list[np.ndarray],
    *,
    top_k: int,
    delta: float,
    l2: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Score every row through weights fitted without its own building group."""
    out = anchor.copy()
    fitted: list[np.ndarray] = []
    for held in partitions:
        inside = np.isin(frame.building, held)
        pos, neg, weight = build_pairs(
            _view(frame, ~inside), anchor[~inside], usable[~inside],
            top_k=top_k, delta=delta,
        )
        w = fit_weights(design[~inside], pos, neg, weight, l2=l2)
        fitted.append(w)
        out[inside] = anchor[inside] + design[inside] @ w
    return out, fitted


class _Sub:
    """A frame restricted to a row mask, for the pair builder."""

    def __init__(self, frame, mask: np.ndarray) -> None:
        self.scenario = frame.scenario[mask]
        self.battery = frame.battery[mask]
        self.due = frame.due[mask]
        self.building = frame.building[mask]


def _view(frame, mask: np.ndarray) -> "_Sub":
    return _Sub(frame, mask)


def anchor_rank(
    scenario: np.ndarray, level: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """The incumbent order as a centred rank, on the residual signals' scale.

    ``logit(p)`` spans about twenty-five units across a scenario while a centred
    rank spans one, so a residual weighted in logit units can essentially never
    overturn the incumbent -- the fit converges to weights that change nothing.
    Replacing the anchor with its own within-scenario rank is a monotone
    transform, so the incumbent ordering is untouched, and it puts the anchor
    and the residuals on one scale. Non-candidates are pushed below every
    candidate and keep their own relative order.
    """
    out = np.zeros(level.shape[0])
    for index in np.unique(scenario):
        rows = np.flatnonzero(scenario == index)
        inside = rows[candidates[rows]]
        outside = rows[~candidates[rows]]
        if inside.size:
            order = np.argsort(np.argsort(level[inside], kind="stable"), kind="stable")
            out[inside] = order / max(inside.size - 1, 1) - 0.5
        if outside.size:
            order = np.argsort(np.argsort(level[outside], kind="stable"), kind="stable")
            out[outside] = order / max(outside.size - 1, 1) - 2.0
    return out
