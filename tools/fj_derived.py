"""Derived per-row signals for the residual-ranking experiments.

Three families the 64 shipped features do not contain:

* **CDF shape.** The planner reads a whole curve but the decision is one
  number, and two batteries with the same 42-day probability can carry very
  different front-loading. ``p07_over_p42`` and friends make that visible.
* **Peer contrast.** ``margin`` minus the median margin of the same room's
  other alive devices *in this scenario*. A voltage that is low because the
  room is cold moves with its peers; a battery that is genuinely dying does
  not. Causal: it reads only the same cutoff the row itself is read at.
  Room support is thin (79 rooms over 461 devices), so the building fallback is
  computed alongside and the peer count is carried so it can be weighted or
  gated.
* **Dwell.** ``days_below_X`` is already a shipped feature, but the
  first-passage law saturates at small margin -- once ``m -> 0`` the crossing
  probability goes to one whatever the drift model says -- so dwell cannot
  express itself through the model's own output. Here it is a candidate for
  reordering that saturated top.

Nothing here reads a future value, an EOL label, or a building identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.hazard import HORIZON_GRID  # noqa: E402

EOL_THRESHOLD = 2.4
MIN_PEERS = 2


def room_of(dataset: Path = Path("dataset/train")) -> dict[str, str]:
    devices = pd.read_csv(dataset / "devices.csv")
    return dict(zip(devices["device_id"].astype(str), devices["room_id"].astype(str)))


def _group_median_contrast(
    scenario: np.ndarray, group: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """value - median(value of the *other* rows sharing scenario and group)."""
    out = np.full(value.shape[0], np.nan)
    count = np.zeros(value.shape[0])
    keys = np.char.add(np.char.add(scenario.astype(str), "|"), group.astype(str))
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    edges = np.flatnonzero(np.r_[True, keys_sorted[1:] != keys_sorted[:-1], True])
    for start, stop in zip(edges[:-1], edges[1:]):
        rows = order[start:stop]
        values = value[rows]
        finite = np.isfinite(values)
        if finite.sum() < MIN_PEERS + 1:
            count[rows] = max(int(finite.sum()) - 1, 0)
            continue
        total = values[finite]
        for position, row in enumerate(rows):
            if not finite[position]:
                continue
            others = np.delete(total, np.flatnonzero(rows[finite] == row)[0])
            out[row] = values[position] - float(np.median(others))
            count[row] = others.size
    return out, count


def derived(frame, grid: np.ndarray, rooms: dict[str, str]) -> dict[str, np.ndarray]:
    """Every extra signal, keyed by name."""
    def column(name: str) -> np.ndarray:
        return frame.features[:, FEATURE_NAMES.index(name)].astype(float)

    margin = column("voltage") - EOL_THRESHOLD
    room = np.asarray([rooms.get(b, "?") for b in frame.battery])
    out: dict[str, np.ndarray] = {"margin": margin}

    # --- CDF shape -------------------------------------------------------
    index42 = HORIZON_GRID.index(42)
    p42 = np.clip(grid[:, index42], 1e-9, 1.0)
    for horizon in (7, 14, 21, 28, 35):
        p = grid[:, HORIZON_GRID.index(horizon)]
        out[f"p{horizon:02d}_over_p42"] = p / p42
    out["cdf_front_mass"] = grid[:, : index42 + 1].sum(axis=1) / (p42 * (index42 + 1))
    out["p42"] = p42
    out["p_late_tail"] = np.clip(grid[:, HORIZON_GRID.index(126)] - p42, 0.0, 1.0)

    # --- dwell -----------------------------------------------------------
    for threshold in ("2.45", "2.50", "2.55", "2.60"):
        name = f"days_below_{threshold}"
        dwell = column(name)
        out[f"dwell_{threshold}"] = dwell
        # -1 means "never below", which is not a small dwell but its opposite.
        out[f"dwell_{threshold}_present"] = np.where(dwell < 0, 0.0, 1.0)
    out["dwell_45_log"] = np.log1p(np.maximum(out["dwell_2.45"], 0.0))

    # --- peer contrast ---------------------------------------------------
    for label, group in (("room", room), ("bldg", frame.building)):
        for name, values in (
            ("margin", margin),
            ("slope30", column("slope_30")),
            ("temp", column("temp_now")),
        ):
            contrast, count = _group_median_contrast(frame.scenario, group, values)
            out[f"rel_{name}_{label}"] = contrast
            if name == "margin":
                out[f"peers_{label}"] = count

    # --- staleness / coverage -------------------------------------------
    out["staleness"] = column("staleness")
    out["gap_fraction_90"] = column("gap_fraction_90")
    out["stale_x_margin"] = column("staleness") * margin

    return out
