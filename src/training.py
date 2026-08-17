
import pandas
import numpy
import pickle
from typing import Optional, Sequence
from pathlib import Path
import pathlib
import os


from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy
import numpy as np
import pandas
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from batteryswap_public.interfaces import Planner, RULModel
from batteryswap_public.utils import load_dataset, iterate_scenarios
from batteryswap_public.evaluate import evaluate_plan, check_plan_valid

log = structlog.get_logger()


class OrderedPlanner(Planner):
    def __init__(self, rul_estimator):
        self.rul_estimator = rul_estimator

    def plan(self, battery_data, locations, travel_costs, settings):

        # Remaining Useful Life estimation
        # FIXME: consider balance of over/under-estimation
        percentile = 'p50'
        rul = self.rul_estimator.predict(battery_data)
        rul_days = rul[percentile]
        # convert to a date
        start_time = battery_data.reset_index()['end_time'].max().normalize()
        predict_eol = start_time + pandas.to_timedelta(rul_days, unit='D')

        loc = locations.copy().set_index('battery')
        loc['eol_time'] = predict_eol
        order = loc.sort_values('eol_time', ascending=True)

        # Planner
        # Stupid heuristic: Do one swap per day
        # FIXME: take travel distances into account
        # FIXME: take co-location into account
        # FIXME: take daily and weekly limits into account
        days = start_time + pandas.to_timedelta(numpy.arange(len(order)), unit='D')
        plan = pandas.DataFrame({
            'day': days,
            'battery': order.index,
        })

        check_plan_valid(plan, locations, start_time=start_time)

        return plan


class DummyRULModel(RULModel):
    # RUL model that predicts (no information rate)
    # FIXME: make a model that actually uses the data to improve predictions

    def __init__(
        self,
        time_col: str = 'end_time',
        group_col: str = 'device_id',
        value_cols: Sequence[str] = ('voltage', 'temperature'),
        quantiles: Sequence[float] = (0.5, ),
    ):
        self.time_col = time_col
        self.group_col = group_col
        self.value_cols = list(value_cols)
        self.quantiles = sorted(quantiles)
        self.quantile_cols = [f"p{round(q * 100):02d}" for q in self.quantiles]

        self.model = None
        self.use_total_elapsed_days = True

    def _compute_features(self, unit_df: pd.DataFrame) -> np.ndarray:
        unit_df = unit_df.sort_values(self.time_col)

        all_stats = {}
        # FIXME: actually compute some features 

        feature_names = sorted(all_stats.keys())
        self._feature_names_ = feature_names  # stored for inspection/debugging
        return np.array([all_stats[k] for k in feature_names], dtype=float)

    def _build_feature_matrix(self, timeseries: pd.DataFrame) -> tuple[np.ndarray, list]:
        rows, ids = [], []
        for unit_id, unit_df in timeseries.groupby(self.group_col):
            if len(unit_df) < 2:
                continue
            rows.append(self._compute_features(unit_df))
            ids.append(unit_id)
        return np.vstack(rows), ids

    def fit(self, timeseries: pd.DataFrame, rul: pd.Series):
        X, ids = self._build_feature_matrix(timeseries)
        y = np.array([rul[unit_id] for unit_id in ids])

        # FIXME: actually use an estimator that learns
        self.model = DummyRegressor(strategy='median')
        self.model.fit(X, y)
        return self

    def predict(self, timeseries: pd.DataFrame) -> pd.DataFrame:
        rows, ids = [], []
        for unit_id, unit_df in timeseries.groupby(self.group_col):
            rows.append(self._compute_features(unit_df))
            ids.append(unit_id)

        X = np.vstack(rows)

        # every quantile column just gets the single point prediction.
        point_pred = self.model.predict(X)
        preds = {col: point_pred for col in self.quantile_cols}

        out = pd.DataFrame(preds, index=pd.Index(ids, name=self.group_col))
        return out[self.quantile_cols]


def train_rul_model(locations, timeseries, eol_times, scenarios, limit_scenarios=None):

    # Collect training data
    # FIXME: train/validate/test split to estimate generalized predictive performance
    gen = iterate_scenarios(locations, timeseries, eol_times, scenarios)
    cut_dfs = []
    cut_ruls = []
    scenarios_loaded = 0
    for scenario, locs, cut, eol in gen:
        print('load-scenario')
        cut = cut.reset_index()
        cut['device_id'] = cut['device_id'].astype(str) + scenario['name']
        cut = cut.set_index(['device_id', 'end_time'])
        cut_dfs.append(cut)
        plan_start = pandas.Timestamp(scenario['start_time'])
        unobserved_rul = 120

        # Convert EOL datetimes to RUL in days relative to planning time
        rul_days = (eol - plan_start) / pandas.Timedelta(days=1)
        rul_days.index = pandas.Series(rul_days.index) + scenario['name']
        rul_days = rul_days.fillna(unobserved_rul)
        cut_ruls.append(rul_days)

        assert set(rul_days.index) == set(cut.index.get_level_values('device_id'))

        if limit_scenarios is not None:
            if scenarios_loaded > limit_scenarios:
                break
        scenarios_loaded += 1

        
    # Train a RUL prediction model
    rul_model = DummyRULModel()
    X = pandas.concat(cut_dfs)
    Y = pandas.concat(cut_ruls)

    print(X.head())
    print(Y.head())

    rul_model.fit(X, Y)

    # FIXME: do model selection

    return rul_model


class Config(BaseSettings):
    """
    Automatically provides command-line argument support for specified fields
    """
    model_config = SettingsConfigDict(
        env_prefix="",
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
    )
 
    dataset_path: Optional[Path] = None
    split : str = 'train'

def main():
    cfg = Config()

    if cfg.dataset_path is None:
        dataset_path = os.environ.get('BATTERYSWAP_DATASET_PATH', None)
        assert dataset_path
        dataset_path = Path(dataset_path)
    else:
        dataset_path = cfg.dataset_path

    split_path = dataset_path / cfg.split
    locations, timeseries, eol_times, scenarios  = load_dataset(split_path)

    log.info('evaluate-load-data', path=dataset_path)


    # Prediction model training
    #rul_model = DummyRULModel()
    rul_model = train_rul_model(locations, timeseries, eol_times, scenarios, limit_scenarios=1)
    log.info('train-done')

    log.info('evaluate')
    # Evaluate on the planning scenarios
    gen = iterate_scenarios(locations, timeseries, eol_times, scenarios)
    for scenario, locs, cut, eol in gen:
        scenario_name = scenario['name']
        travel_costs = scenario['travel_costs']
        settings = scenario['settings']

        planner = OrderedPlanner(rul_model)
        plan = planner.plan(cut, locs, travel_costs, settings)

        start_time = pandas.Timestamp(scenario['start_time'])

        transitions, daily, overall = evaluate_plan(plan, locs, travel_costs, settings, eol_times=eol, start_time=start_time)

        print('scores', scenario_name, overall)


    # Save best planner
    planner = OrderedPlanner(rul_model)

    planner_path = 'batteryswap_example/planners/best.pickle' 
    with open(planner_path, "wb") as f:
        pickle.dump(planner, f)
        print('planner-save', planner_path)


if __name__ == '__main__':
    main()
