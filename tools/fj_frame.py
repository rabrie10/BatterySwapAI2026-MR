"""Shared loading for the final-J2W experiments: the cached scenario frame plus
whatever CDF grid a candidate model puts on it.

``tools/build_scenario_frame.py`` already caches the exact population the
forecaster is asked about -- 19,890 (scenario, battery) rows with their 64
causal features, the remaining observation window, the out-of-fold V8
probability and the realised 42-day label. Everything in this phase is a
re-scoring of those rows, so nothing here touches the 8.5 M-row timeseries and
every experiment costs seconds instead of thirteen minutes.

The one thing the cache does not hold is the *shape* of the CDF: it stores the
42-day column only. ``grid_for`` rebuilds the whole 24-point grid by dispatching
each row to the fold model that never saw its building, which is the same
routing ``OofHazardModel`` does inside the forecaster.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.hazard import HORIZON_GRID  # noqa: E402

DECISION_COLUMN = HORIZON_GRID.index(42)
DEFAULT_FRAME = REPO_ROOT / "outputs" / "v9_frame.npz"


@dataclass
class Frame:
    features: np.ndarray
    scenario: np.ndarray
    battery: np.ndarray
    building: np.ndarray
    probability: np.ndarray
    remaining: np.ndarray
    due: np.ndarray
    days_to_eol: np.ndarray
    substitute_eol: np.ndarray

    @property
    def n_scenarios(self) -> int:
        return int(np.unique(self.scenario).size)

    def column(self, name: str) -> np.ndarray:
        return self.features[:, FEATURE_NAMES.index(name)].astype(float)


def load_frame(path: Path = DEFAULT_FRAME) -> Frame:
    data = np.load(path, allow_pickle=True)
    return Frame(
        features=data["features"],
        scenario=data["scenario_index"],
        battery=np.asarray([str(b) for b in data["battery"]]),
        building=np.asarray([str(b) for b in data["building"]]),
        probability=data["probability"],
        remaining=data["remaining"],
        due=data["due"].astype(bool),
        days_to_eol=data["days_to_eol"],
        substitute_eol=data["substitute_eol"],
    )


def grid_for(frame: Frame, folds_path: Path, *, volatility_scale: float = 1.0) -> np.ndarray:
    """The out-of-fold CDF grid, one row per frame row, 24 horizons wide."""
    bundle = joblib.load(folds_path)
    by_building = bundle["by_building"]
    out = np.zeros((frame.features.shape[0], len(HORIZON_GRID)))
    for building in np.unique(frame.building):
        model = by_building.get(building)
        if model is None:
            raise KeyError(f"no fold model for building {building!r}")
        model.volatility_scale = volatility_scale
        mask = frame.building == building
        out[mask] = model.predict_grid(frame.features[mask], frame.remaining[mask])
    return out


def per_scenario_rank(scenario: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Dense descending rank of ``score`` inside each scenario (0 = riskiest)."""
    rank = np.empty(score.shape[0], dtype=np.int64)
    for index in np.unique(scenario):
        mask = scenario == index
        order = np.argsort(-score[mask], kind="stable")
        local = np.empty(order.size, dtype=np.int64)
        local[order] = np.arange(order.size)
        rank[mask] = local
    return rank


def decision_probability(grid: np.ndarray, remaining: np.ndarray) -> np.ndarray:
    """The 42-day number the planner actually sees.

    ``HazardForecaster.predict`` caps the whole curve at its own value at the
    censoring horizon, because no EOL record can be filed after observation
    ends. For a row with fewer than 42 days of observation left that cap, not
    the 42-day column, is the decision probability.
    """
    xs = np.concatenate([[0.0], np.asarray(HORIZON_GRID, dtype=float)])
    out = np.empty(grid.shape[0])
    for row in range(grid.shape[0]):
        ys = np.concatenate([[0.0], grid[row]])
        at_horizon = float(np.interp(42.0, xs, ys))
        at_censor = float(np.interp(max(float(remaining[row]), 0.0), xs, ys))
        out[row] = 0.0 if remaining[row] < 0.0 else min(at_horizon, at_censor)
    return np.clip(out, 0.0, 1.0)
