"""Out-of-fold forecaster wrapper.

Lives in an importable module (not in a script's ``__main__``) so that the
pickled artifact can be loaded by any consumer -- ``tools/benchmark_task2.py``,
``script.py``, or a notebook -- rather than only by the script that created it.
See ``tools/fit_oof_forecasters.py`` for why the harness exists.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from batteryswap_solution.forecast import RiskForecast


class OutOfFoldForecaster:
    """Routes each battery to the fold model that did not train on its building.

    Implements the same ``predict()`` contract as ``Task1Forecaster``. This
    makes a train-scenario benchmark an honest estimate of unseen-building
    behaviour, which is the condition the public and private splits actually
    present. Batteries from a building that was never held out fall back to
    ``default_forecaster`` (fit on everything) -- the same behaviour a real
    submission has when it meets a genuinely new building.
    """

    model_version = "task1-oof-harness/v1"

    def __init__(self, fold_forecasters, building_to_fold, default_forecaster) -> None:
        self.fold_forecasters = fold_forecasters
        self.building_to_fold = building_to_fold
        self.default_forecaster = default_forecaster

    def predict(self, battery_data, locations, *, prediction_origin,
                horizon_days, evaluation_observation_end):
        id_column = "battery_id" if "battery_id" in locations else "battery"
        building_column = "building_id" if "building_id" in locations else "building"
        loc = locations.copy()
        loc[id_column] = loc[id_column].astype(str)

        folds = loc[building_column].astype(str).map(self.building_to_fold)
        curves, tails, summaries = [], [], []
        metadata = None

        for fold_value, group in loc.groupby(folds.fillna(-1), sort=True):
            model = self.fold_forecasters.get(int(fold_value), self.default_forecaster)
            part = model.predict(
                battery_data, group,
                prediction_origin=prediction_origin,
                horizon_days=horizon_days,
                evaluation_observation_end=evaluation_observation_end,
            )
            metadata = metadata or part.metadata
            curves.append(part.curves)
            tails.append(part.tail)
            if part.summaries is not None and not part.summaries.empty:
                summaries.append(part.summaries)

        return RiskForecast(
            replace(metadata, model_version=self.model_version),
            pd.concat(curves, ignore_index=True),
            pd.concat(tails, ignore_index=True),
            pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
        )
