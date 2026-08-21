"""Out-of-fold model dispatch, so validation never scores a device with a model
that has seen its building.

The public and private splits contain buildings we have never seen, and the
observed EOL rate per training building spans 0.043 to 0.833. Scoring a device
with a model trained on its own building would hide exactly the failure that put
41 swaps per scenario on the public leaderboard against 11 locally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureContext
from .hazard import HazardModel


@dataclass
class OofHazardModel:
    """Presents one model's interface while routing each device to its fold."""

    by_building: dict[str, HazardModel]
    building_of: dict[str, str]
    climatology: np.ndarray
    model_version: str = "bsai-hazard-oof/v1"

    @property
    def horizons(self) -> tuple[int, ...]:
        return next(iter(self.by_building.values())).horizons

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def probability_scale_for_origin(self, origin) -> float:
        model = next(iter(self.by_building.values()))
        method = getattr(model, "probability_scale_for_origin", None)
        return 1.0 if method is None else float(method(origin))

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        if devices is None:
            raise ValueError("Out-of-fold prediction needs the device of each row")
        out = np.zeros((features.shape[0], len(self.horizons)))
        buildings = np.array(
            [self.building_of.get(str(d), "") for d in devices], dtype=object
        )
        for building in np.unique(buildings):
            model = self.by_building.get(str(building))
            if model is None:
                # A building with no fold model would silently score zero, which
                # reads as "never due" -- refuse instead of quietly deferring.
                raise KeyError(f"No out-of-fold model for building {building!r}")
            mask = buildings == building
            out[mask] = model.predict_grid(features[mask], remaining[mask])
        return out

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        return next(iter(self.by_building.values())).cdf_at(grid_values, days)
