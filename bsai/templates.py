"""EOL-aligned trajectory templates: how far from death does this shape look?

Every previous attempt on this branch reduced a trajectory to scalars -- a slope,
a dwell, a volatility ratio -- and then asked a model to combine them. This does
not. It keeps the trajectory as a trajectory and asks a different question:

    the 82 devices that crossed 2.4 V each leave a *curve* running into their own
    end of life. Take the segment of one of those curves ending L days before the
    crossing. Does the last W days of this battery's history look like it?

If the nearest matches are segments that ended 20 days before a death, the
battery is shaped like a cell 20 days from death. If they are segments that
ended 300 days before, it is not. **The output is a predicted lead time**, on a
continuous axis, estimated by k-nearest-neighbour regression over template
segments -- not a class label, and not a summary statistic.

Two properties make this worth trying after the volatility ratio failed.

* **It uses the deaths as curves rather than as events.** 82 crossings give 82
  labels but several thousand distinguishable pre-EOL segments, so the sample
  size that limits a classifier does not limit this in the same way.
* **It can order across margins.** The matched-volatility signal only knew how
  to compare batteries at the same margin, which is a third of the decisions.
  Template distance is defined between any two trajectories, and the lead-time
  axis is exactly a cross-margin ordering.

Normalisation decides what "shape" means, so both are built and measured:

``anchored``  subtract the window's last value. The comparison is then over the
             *recent change*, and the absolute level is discarded -- the axis
             that is genuinely new, and the one margin cannot already express.
``level``     no subtraction. Distance then includes where the battery sits, so
             it partly re-encodes margin; kept as the control that says how much
             of any gain is shape rather than level.

Fold discipline is the whole experiment. A template may only come from a device
whose building is in the *training* half of the fold being scored, so no
held-out building's death ever contributes a template to a query from that
building. ``build_templates`` takes the allowed device set explicitly and there
is no default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EOL_THRESHOLD = 2.4
TEMPERATURE_BETA = 0.00463
REFERENCE_TEMPERATURE = 20.0

# Window lengths, in days of the smoothed daily grid.
WINDOWS = (30, 60, 90, 180)
# Lead times sampled off each crossing: the segment ending L days before it.
LEAD_GRID = tuple(range(0, 366, 7))
# A segment or query is unusable if more than this fraction of it is missing.
MAX_MISSING = 0.25


def adjusted(voltage: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Voltage with the fleet temperature coefficient removed."""
    filled = np.where(np.isnan(temperature), REFERENCE_TEMPERATURE, temperature)
    return voltage - TEMPERATURE_BETA * (filled - REFERENCE_TEMPERATURE)


def _segment(series: np.ndarray, end: int, width: int) -> np.ndarray | None:
    """``width`` days ending at ``end`` inclusive, gaps filled, or None."""
    start = end - width + 1
    if start < 0 or end >= series.size:
        return None
    window = series[start : end + 1].astype(float)
    missing = np.isnan(window)
    if missing.mean() > MAX_MISSING or missing[-1]:
        return None
    if missing.any():
        index = np.arange(width)
        window = np.interp(index, index[~missing], window[~missing])
    return window


def normalise(window: np.ndarray, mode: str) -> np.ndarray:
    if mode == "level":
        return window
    if mode == "anchored":
        return window - window[-1]
    raise ValueError(f"unknown normalisation {mode!r}")


@dataclass
class TemplateBank:
    """Pre-EOL segments and the lead time each one ended at."""

    vectors: np.ndarray          # (n_templates, width)
    lead: np.ndarray             # days between the segment end and the crossing
    device: np.ndarray

    def __len__(self) -> int:
        return int(self.vectors.shape[0])


def build_templates(
    series: dict,
    crossing: dict,
    allowed: set,
    *,
    width: int,
    channel: str,
    mode: str,
    leads: tuple[int, ...] = LEAD_GRID,
) -> TemplateBank:
    """Segments from the crossing devices in ``allowed`` and nobody else.

    ``allowed`` is the set of devices whose building sits in the training half
    of the fold being scored. It is required, not defaulted, because getting it
    wrong is the one mistake that would invalidate the entire experiment.
    """
    vectors: list[np.ndarray] = []
    lead_out: list[int] = []
    device_out: list[str] = []
    for device, (voltage, temperature, _origin) in series.items():
        if device not in allowed:
            continue
        cross = crossing.get(device)
        if cross is None or cross < 0:
            continue
        track = voltage if channel == "voltage" else adjusted(voltage, temperature)
        for lead in leads:
            window = _segment(track, cross - lead, width)
            if window is None:
                continue
            vectors.append(normalise(window, mode))
            lead_out.append(lead)
            device_out.append(device)
    if not vectors:
        return TemplateBank(
            np.zeros((0, width)), np.zeros(0, dtype=int), np.zeros(0, dtype=object)
        )
    return TemplateBank(
        np.vstack(vectors).astype(np.float32),
        np.asarray(lead_out, dtype=int),
        np.asarray(device_out, dtype=object),
    )


def build_queries(
    series: dict,
    devices: np.ndarray,
    cutoff: np.ndarray,
    *,
    width: int,
    channel: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One vector per row, plus the mask of rows that produced one."""
    out = np.zeros((devices.size, width), dtype=np.float32)
    usable = np.zeros(devices.size, dtype=bool)
    for row in range(devices.size):
        entry = series.get(str(devices[row]))
        if entry is None:
            continue
        voltage, temperature, _origin = entry
        track = voltage if channel == "voltage" else adjusted(voltage, temperature)
        window = _segment(track, int(cutoff[row]), width)
        if window is None:
            continue
        out[row] = normalise(window, mode)
        usable[row] = True
    return out, usable


def nearest_lead(
    queries: np.ndarray,
    bank: TemplateBank,
    *,
    k: int = 25,
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """k-NN regression of lead time, and the distance to the nearest template.

    Distances are squared Euclidean over the window, computed by the usual
    expansion so the whole thing is one matrix product. The lead estimate is a
    softmin-weighted mean rather than a plain average, so a query that matches
    one segment closely is not dragged toward the bulk of the bank.
    """
    if len(bank) == 0:
        return np.full(queries.shape[0], np.nan), np.full(queries.shape[0], np.nan)
    template_norm = (bank.vectors.astype(np.float64) ** 2).sum(axis=1)
    lead = bank.lead.astype(float)
    predicted = np.empty(queries.shape[0])
    closest = np.empty(queries.shape[0])
    neighbours = min(k, len(bank))
    for start in range(0, queries.shape[0], chunk):
        block = queries[start : start + chunk].astype(np.float64)
        distance = (
            (block**2).sum(axis=1)[:, None]
            + template_norm[None, :]
            - 2.0 * block @ bank.vectors.astype(np.float64).T
        )
        np.maximum(distance, 0.0, out=distance)
        index = np.argpartition(distance, neighbours - 1, axis=1)[:, :neighbours]
        taken = np.take_along_axis(distance, index, axis=1)
        closest[start : start + chunk] = np.sqrt(taken.min(axis=1))
        scale = np.median(taken, axis=1, keepdims=True)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        weight = np.exp(-taken / scale)
        weight /= weight.sum(axis=1, keepdims=True)
        predicted[start : start + chunk] = (weight * lead[index]).sum(axis=1)
    return predicted, closest


@dataclass
class TemplateScorer:
    """Reorder within margin bins by predicted lead time, out of fold by building.

    The deployment shape `docs/FINAL_TERMINALITY.md` settled on: a signal
    validated at matched margin is spent at matched margin, so rows are
    permuted only against others in the same 0.01 V bin and everything outside
    the candidate band keeps the incumbent order exactly.

    ``banks`` maps a fold id to the template bank built without that fold's
    buildings; ``fold_of`` maps a building to its fold. A device whose building
    has no bank keeps the incumbent order rather than borrowing another fold's.
    """

    series: dict
    end_ordinal: dict
    banks: dict
    fold_of: dict
    building_of: dict
    width: int = 60
    channel: str = "voltage"
    mode: str = "anchored"
    weight: float = 0.25
    band: tuple[float, float] = (0.0, 0.10)
    bin_width: float = 0.01
    k: int = 25
    horizons: tuple[int, ...] = ()

    def score(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None,
        grid: np.ndarray,
    ) -> np.ndarray:
        from .features import FEATURE_NAMES
        from .hazard import HORIZON_GRID
        from .rerank import centred_rank, decision_level

        horizons = self.horizons or HORIZON_GRID
        base = centred_rank(decision_level(grid, remaining, horizons))
        if devices is None:
            return base
        margin = features[:, FEATURE_NAMES.index("voltage")].astype(float) - EOL_THRESHOLD
        band = np.flatnonzero((margin >= self.band[0]) & (margin <= self.band[1]))
        if band.size < 3:
            return base

        lead = np.full(band.size, np.nan)
        cutoffs = np.full(band.size, -1, dtype=int)
        for position, row in enumerate(band):
            entry = self.series.get(str(devices[row]))
            end = self.end_ordinal.get(str(devices[row]))
            if entry is None or end is None:
                continue
            index = int(round(end - float(remaining[row]) - entry[2]))
            if 0 <= index:
                cutoffs[position] = min(index, entry[0].size - 1)
        folds = np.asarray([
            self.fold_of.get(self.building_of.get(str(devices[row]), ""), -1)
            for row in band
        ])
        for fold in np.unique(folds):
            bank = self.banks.get(int(fold))
            if bank is None or len(bank) == 0:
                continue
            chosen = np.flatnonzero((folds == fold) & (cutoffs >= 0))
            if chosen.size == 0:
                continue
            queries, ok = build_queries(
                self.series, devices[band[chosen]], cutoffs[chosen],
                width=self.width, channel=self.channel, mode=self.mode,
            )
            if ok.sum() == 0:
                continue
            predicted, _ = nearest_lead(queries[ok], bank, k=self.k)
            lead[chosen[ok]] = predicted

        out = base.copy()
        bins = np.floor(margin[band] / self.bin_width).astype(int)
        for value in np.unique(bins):
            inside = bins == value
            rows = band[inside]
            if rows.size < 2 or np.isfinite(lead[inside]).sum() < 2:
                continue
            neighbour = np.where(np.isfinite(lead[inside]), lead[inside], np.inf)
            blended = centred_rank(base[rows]) + self.weight * centred_rank(-neighbour)
            source = rows[np.argsort(-base[rows], kind="stable")]
            target = rows[np.lexsort((-base[rows], -blended))]
            out[target] = base[source]
        return out
