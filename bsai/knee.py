"""Knee-onset probability floor over the (margin, beta_30) plane.

The late pool -- the ~3.6-3.9 dues per scenario the planner misses, worth about
a thousand local points -- is not noise. Measured on the raw out-of-fold frame
(outputs/frame_oof_raw_beta.parquet), the missed dues sit at a median margin of
0.12 V at the cutoff and fail a median 25 days later, and 85% of them carry a
trailing within-day dV/dT (beta_30, the internal-resistance channel) above twice
the fleet median. The IR channel fires; the Gaussian increment model cannot say
"probably flat, but a 20-30% chance the plunge starts inside the window",
because a Wiener path has one drift, not a mixture over knee onset.

So the correction is not a rescale -- the model's own p in this population is
0.003-0.14 against realized rates of 0.06-0.28, and multiplying near-zero by
anything keeps it near zero. It is a *floor*: within coarse margin x beta_30
cells, the predicted 42-day probability is raised to the out-of-fold empirical
due rate of the cell, shrunk toward the pooled rate and clamped, exactly the
DwellAdjust discipline (bsai/calibrate.py) on a different axis pair.

Censoring is respected on both sides. Floors are fitted only on rows with at
least ``remaining_gate`` days of observation left (the frame's label is already
capped at min(42, remaining)), and at apply time the floor is scaled by
``min(1, remaining / 42)`` so a battery that can only be observed for 30 more
days is never floored at the full 42-day hazard.

Ordering caveat: ``WienerModel.predict_grid`` applies ``knee_boost`` *before*
the remaining-observation calibration, which then multiplies the grid by a
factor of 0.4-2.4. A floor set to the empirical rate would land at rate x
factor after that stage. ``compensation`` holds the fold's RemainingCalibration
so the floor is pre-divided by that factor and lands at the empirical rate on
the planner's side. With an isotonic ReliabilityCalibration in the slot the
divide is wrong -- fit the floors against isotonic-calibrated predictions
instead and leave ``compensation`` unset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bands chosen from the measured due-rate surface, not tuned: below 0.05 V the
# dwell adjustment owns the row, above 0.20 V the realized 42-day rate is under
# 2% everywhere; beta_30 cuts at roughly 1.2x and 2.3x the fleet median.
MARGIN_EDGES = (0.05, 0.10, 0.15, 0.20)
BETA_EDGES = (0.008, 0.012, 0.016, 1e9)

# Index of the 42-day horizon in bsai.hazard.HORIZON_GRID.
HORIZON_COLUMN_42 = 11
WINDOW_DAYS = 42.0

# A floor above this is far more likely a thin-cell artefact than a real rate:
# the densest cell in the plane realizes 0.28.
MAX_FLOOR = 0.35


def _band(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Band index per row plus an in-domain mask (NaN-safe)."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    safe = np.where(finite, values, edges[0] - 1.0)
    inside = finite & (safe >= edges[0]) & (safe < edges[-1])
    index = np.clip(
        np.searchsorted(edges, safe, side="right") - 1, 0, len(edges) - 2
    )
    return index, inside


@dataclass
class KneeBoost:
    """Floor on the 42-day probability, keyed by (margin, beta_30) cell.

    ``apply`` matches the hook in ``WienerModel.predict_grid``: the whole
    horizon grid is rescaled by ``max(p42, floor) / p42`` so the curve keeps
    its shape and stays monotone across horizons.
    """

    margin_edges: tuple[float, ...] = MARGIN_EDGES
    beta_edges: tuple[float, ...] = BETA_EDGES
    # floors[margin_band][beta_band]; 0.0 means "no floor" for that cell.
    floors: tuple[tuple[float, ...], ...] = ()
    remaining_gate: float = 30.0
    window_days: float = WINDOW_DAYS
    censor_scale: bool = True
    horizon_column: int = HORIZON_COLUMN_42
    # Fold's RemainingCalibration (multiplied in *after* this boost); the floor
    # is pre-divided by its factor so it lands at the empirical rate.
    compensation: object | None = None
    # Fit diagnostics, kept so describe() can show the evidence.
    cell_rows: tuple[tuple[int, ...], ...] = field(default=(), repr=False)
    cell_events: tuple[tuple[int, ...], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.floors:
            rows = len(self.margin_edges) - 1
            cols = len(self.beta_edges) - 1
            self.floors = tuple(tuple(0.0 for _ in range(cols)) for _ in range(rows))

    # ------------------------------------------------------------------ apply
    def floor_for(
        self,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray | None = None,
    ) -> np.ndarray:
        """Effective empirical-rate floor per row (0 outside the domain)."""
        margin = np.asarray(margin, dtype=float)
        m_band, m_in = _band(np.asarray(self.margin_edges, dtype=float), margin)
        b_band, b_in = _band(np.asarray(self.beta_edges, dtype=float), beta30)
        table = np.asarray(self.floors, dtype=float)
        floor = np.where(m_in & b_in, table[m_band, b_band], 0.0)
        if remaining is not None:
            remaining = np.asarray(remaining, dtype=float)
            floor = np.where(remaining >= float(self.remaining_gate), floor, 0.0)
            if self.censor_scale:
                scale = np.clip(remaining / float(self.window_days), 0.0, 1.0)
                floor = floor * scale
        return floor

    def factor_for(
        self,
        p: np.ndarray,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray | None = None,
    ) -> np.ndarray:
        """Multiplier turning p into max(p, floor), on the pre-hook scale."""
        p = np.asarray(p, dtype=float)
        floor = self.floor_for(margin, beta30, remaining)
        if self.compensation is not None and remaining is not None:
            downstream = np.clip(
                np.asarray(self.compensation.factor_for(np.asarray(remaining, dtype=float))),
                0.2,
                None,
            )
            floor = floor / downstream
        return np.maximum(p, floor) / np.maximum(p, 1e-9)

    def apply(
        self,
        grid: np.ndarray,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray | None = None,
    ) -> np.ndarray:
        if grid.shape[0] == 0:
            return grid
        column = min(self.horizon_column, grid.shape[1] - 1)
        factor = self.factor_for(grid[:, column], margin, beta30, remaining)[:, None]
        return np.clip(grid * factor, 0.0, 1.0)

    # -------------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        margin: np.ndarray,
        beta30: np.ndarray,
        actual: np.ndarray,
        remaining: np.ndarray,
        *,
        margin_edges: tuple[float, ...] = MARGIN_EDGES,
        beta_edges: tuple[float, ...] = BETA_EDGES,
        remaining_gate: float = 30.0,
        min_rows: int = 50,
        min_events: int = 10,
        shrink: float = 25.0,
        max_floor: float = MAX_FLOOR,
        censor_scale: bool = True,
    ) -> "KneeBoost":
        """Per-cell empirical due rate, shrunk toward the pooled rate.

        ``actual`` must be the censor-capped label (due within
        ``min(42, remaining)``), which is what the diagnostic frame carries.
        Cells with fewer than ``min_rows`` rows or ``min_events`` events get no
        floor at all -- an absolute rate from a thin cell is worse than the
        model's own number. Estimable cells are shrunk with ``shrink``
        pseudo-rows at the pooled in-domain rate and clamped at ``max_floor``.
        """
        margin = np.asarray(margin, dtype=float)
        beta30 = np.asarray(beta30, dtype=float)
        actual = np.asarray(actual, dtype=float)
        remaining = np.asarray(remaining, dtype=float)

        m_edges = np.asarray(margin_edges, dtype=float)
        b_edges = np.asarray(beta_edges, dtype=float)
        m_band, m_in = _band(m_edges, margin)
        b_band, b_in = _band(b_edges, beta30)
        eligible = m_in & b_in & (remaining >= float(remaining_gate))

        pooled = float(actual[eligible].mean()) if eligible.any() else 0.0
        rows_out: list[tuple[float, ...]] = []
        n_out: list[tuple[int, ...]] = []
        e_out: list[tuple[int, ...]] = []
        for mi in range(len(m_edges) - 1):
            row: list[float] = []
            ns: list[int] = []
            es: list[int] = []
            for bi in range(len(b_edges) - 1):
                cell = eligible & (m_band == mi) & (b_band == bi)
                n = int(cell.sum())
                events = int(actual[cell].sum())
                ns.append(n)
                es.append(events)
                if n < min_rows or events < min_events:
                    row.append(0.0)
                    continue
                rate = (events + shrink * pooled) / (n + shrink)
                row.append(float(np.clip(rate, 0.0, max_floor)))
            rows_out.append(tuple(row))
            n_out.append(tuple(ns))
            e_out.append(tuple(es))
        return cls(
            margin_edges=tuple(margin_edges),
            beta_edges=tuple(beta_edges),
            floors=tuple(rows_out),
            remaining_gate=float(remaining_gate),
            censor_scale=censor_scale,
            cell_rows=tuple(n_out),
            cell_events=tuple(e_out),
        )

    def describe(self) -> str:
        lines = [
            f"{'margin band':>13} | "
            + "  ".join(
                f"beta [{lo:.3f},{hi:.3f})" if hi < 1e8 else f"beta {lo:.3f}+      "
                for lo, hi in zip(self.beta_edges[:-1], self.beta_edges[1:])
            )
        ]
        for mi, (lo, hi) in enumerate(zip(self.margin_edges[:-1], self.margin_edges[1:])):
            cells = []
            for bi, floor in enumerate(self.floors[mi]):
                n = self.cell_rows[mi][bi] if self.cell_rows else -1
                e = self.cell_events[mi][bi] if self.cell_events else -1
                cells.append(f"{floor:.3f} (n={n:>4},e={e:>3})")
            lines.append(f"[{lo:.2f},{hi:.2f})    | " + "  ".join(cells))
        return "\n".join(lines)


# Remaining-observation bands for the banded floor. Measured on the raw frame,
# the knee-cell due rate is savagely non-stationary along this axis: the hot
# cells realize 0.55 at remaining >= 220 (the opening scenarios harvesting the
# ripe stock the dataset opens with) against 0.03-0.13 below 220. A flat floor
# averages those into a number that is wrong everywhere. Same confound warning
# as bsai/calibrate.py: on train this axis moves with the calendar.
REMAINING_EDGES = (30.0, 150.0, 220.0, 1e9)

# The banded table gets two beta bands, not three: the added axis triples the
# cell count, and the hot region's 62 events cannot honestly fund a 0.012/0.016
# split (20 events in the top sliver disappears under any per-fold 25-event
# gate). Elevated-vs-very-elevated matters less than elevated-vs-not.
BANDED_BETA_EDGES = (0.008, 0.012, 1e9)


@dataclass
class KneeBoostBanded:
    """One KneeBoost table per coarse remaining-observation band.

    Same ``apply`` contract as KneeBoost, so it drops into the same
    ``model.knee_boost`` hook. Cells need more events than the flat table
    (default 25) because the extra axis multiplies the ways to fit noise.
    """

    remaining_edges: tuple[float, ...] = REMAINING_EDGES
    boosts: tuple[KneeBoost, ...] = ()
    compensation: object | None = None

    def _band_of(self, remaining: np.ndarray) -> np.ndarray:
        edges = np.asarray(self.remaining_edges, dtype=float)
        remaining = np.asarray(remaining, dtype=float)
        return np.where(
            remaining >= edges[0],
            np.clip(np.searchsorted(edges, remaining, side="right") - 1, 0, len(self.boosts) - 1),
            -1,
        )

    def floor_for(
        self,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray,
    ) -> np.ndarray:
        margin = np.asarray(margin, dtype=float)
        remaining = np.asarray(remaining, dtype=float)
        bands = self._band_of(remaining)
        out = np.zeros(margin.shape[0] if margin.ndim else 1, dtype=float)
        for band in np.unique(bands):
            if band < 0:
                continue
            mask = bands == band
            out[mask] = self.boosts[int(band)].floor_for(
                margin[mask], np.asarray(beta30, dtype=float)[mask], remaining[mask]
            )
        return out

    def factor_for(
        self,
        p: np.ndarray,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray,
    ) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        floor = self.floor_for(margin, beta30, remaining)
        if self.compensation is not None:
            downstream = np.clip(
                np.asarray(self.compensation.factor_for(np.asarray(remaining, dtype=float))),
                0.2,
                None,
            )
            floor = floor / downstream
        return np.maximum(p, floor) / np.maximum(p, 1e-9)

    def apply(
        self,
        grid: np.ndarray,
        margin: np.ndarray,
        beta30: np.ndarray,
        remaining: np.ndarray | None = None,
    ) -> np.ndarray:
        if grid.shape[0] == 0 or remaining is None or not self.boosts:
            return grid
        column = min(self.boosts[0].horizon_column, grid.shape[1] - 1)
        factor = self.factor_for(grid[:, column], margin, beta30, remaining)[:, None]
        return np.clip(grid * factor, 0.0, 1.0)

    @classmethod
    def fit(
        cls,
        margin: np.ndarray,
        beta30: np.ndarray,
        actual: np.ndarray,
        remaining: np.ndarray,
        *,
        remaining_edges: tuple[float, ...] = REMAINING_EDGES,
        beta_edges: tuple[float, ...] = BANDED_BETA_EDGES,
        # 40 rows, not 50: a leave-fold-out complement of the hot cell holds 42
        # rows with 29 events at rate 0.69, and refusing to learn from 29
        # events because they arrived in few rows is the wrong side of caution.
        # The event gate (25) is the one that guards against noise.
        min_rows: int = 40,
        min_events: int = 25,
        shrink: float = 25.0,
        max_floor: float = 0.65,
        **kwargs,
    ) -> "KneeBoostBanded":
        """Fit one table per band on that band's rows only.

        The shrink target is the band's own pooled rate, so a hot opening band
        does not inherit the quiet mid-year rate and vice versa. ``max_floor``
        is wider than the flat table's because the measured opening-band rate
        genuinely reaches 0.55.
        """
        remaining = np.asarray(remaining, dtype=float)
        boosts: list[KneeBoost] = []
        for lo, hi in zip(remaining_edges[:-1], remaining_edges[1:]):
            inside = (remaining >= lo) & (remaining < hi)
            boost = KneeBoost.fit(
                np.asarray(margin, dtype=float)[inside],
                np.asarray(beta30, dtype=float)[inside],
                np.asarray(actual, dtype=float)[inside],
                remaining[inside],
                beta_edges=beta_edges,
                remaining_gate=float(lo),
                min_rows=min_rows,
                min_events=min_events,
                shrink=shrink,
                max_floor=max_floor,
                **kwargs,
            )
            boosts.append(boost)
        return cls(remaining_edges=tuple(remaining_edges), boosts=tuple(boosts))

    def describe(self) -> str:
        lines = []
        for (lo, hi), boost in zip(
            zip(self.remaining_edges[:-1], self.remaining_edges[1:]), self.boosts
        ):
            span = f"[{lo:.0f},{hi:.0f})" if hi < 1e8 else f"{lo:.0f}+"
            lines.append(f"remaining {span}:")
            lines.append(boost.describe())
        return "\n".join(lines)
