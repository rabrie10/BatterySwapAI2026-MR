"""Train V9's scenario-incidence model from grouped-OOF V7 forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.forecaster import HazardForecaster
from bsai.portfolio import (
    CrossFittedIncidenceModel,
    ScenarioIncidenceModel,
    horizon_probabilities,
    scenario_feature_names,
    scenario_features,
)
from bsai.validation import OofHazardModel


def candidates():
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        yield f"ridge_{alpha:g}", make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)
        )
    for alpha in (0.1, 1.0, 10.0):
        yield f"poisson_{alpha:g}", make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=2000)
        )


def cross_fitted_predictions(estimator, x, y, groups) -> np.ndarray:
    prediction = np.empty(len(y), dtype=float)
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    for train, validation in splitter.split(x, y, groups):
        fitted = estimator.fit(x[train], y[train])
        prediction[validation] = fitted.predict(x[validation])
    return np.clip(prediction, 0.0, None)


def proxy_policy_score(examples, predicted_due, multiplier, buffer, maximum_budget):
    rows = []
    for example, count_prediction in zip(examples, predicted_due):
        budget = int(
            np.clip(
                np.ceil(multiplier * max(float(count_prediction), 0.0) + buffer),
                2,
                maximum_budget,
            )
        )
        selected = example["ranked_ids"][:budget]
        due = example["due"].reindex(selected).fillna(False).to_numpy(dtype=bool)
        days_to_eol = example["days_to_eol"].reindex(selected).to_numpy(dtype=float)
        early = 0.5 * np.clip(days_to_eol[~due] - 42.0, 0.0, None).sum()
        early += 0.5 * 5.0 * int(due.sum())
        missed_ids = example["due"].index[example["due"] & ~example["due"].index.isin(selected)]
        missed_days = example["days_to_eol"].reindex(missed_ids).to_numpy(dtype=float)
        late = 10.0 * np.clip(48.0 - missed_days, 0.0, None).sum()
        rows.append(
            {
                "budget": budget,
                "due": int(example["due"].sum()),
                "hits": int(due.sum()),
                "early": float(early),
                "late": float(late),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "multiplier": float(multiplier),
        "buffer": float(buffer),
        "mean_budget": round(float(frame["budget"].mean()), 3),
        "recall": round(float(frame["hits"].sum() / frame["due"].sum()), 4),
        "precision": round(float(frame["hits"].sum() / frame["budget"].sum()), 4),
        "early_proxy": round(float(frame["early"].mean()), 3),
        "late_proxy": round(float(frame["late"].mean()), 3),
        "timing_proxy": round(float((frame["early"] + frame["late"]).mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--folds", type=Path, default=Path("outputs/v7_wiener_folds.joblib")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/v9_incidence.joblib")
    )
    parser.add_argument(
        "--oof-model", type=Path, default=Path("outputs/v9_incidence_oof.joblib")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/v9_incidence_training_report.json")
    )
    parser.add_argument("--maximum-budget", type=int, default=24)
    parser.add_argument("--service-multiplier", type=float, default=1.25)
    parser.add_argument("--service-buffer", type=float, default=6.0)
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)

    feature_rows = []
    targets = []
    examples = []
    scenario_names = []
    scenario_origins = []
    for scenario, locs, cut, active_eol in iterate_scenarios(
        locations, timeseries, eol_times, scenarios
    ):
        start = pd.Timestamp(scenario["start_time"])
        horizon = int(scenario["settings"].planning_window_days)
        end = start + pd.Timedelta(days=float(horizon))
        forecast = forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probability = horizon_probabilities(forecast)
        ranked_ids = probability.sort_values(ascending=False).index
        recorded = active_eol.reindex(probability.index)
        due = recorded.notna() & (recorded <= end)
        end_time = pd.to_datetime(locs.set_index(locs["battery"].astype(str))["end_time"])
        substitute = end_time.dt.normalize() + pd.Timedelta(
            days=float(scenario["settings"].unobserved_eol_days)
        )
        effective = recorded.fillna(substitute)
        days_to_eol = (effective - start.normalize()) / pd.Timedelta("1D")

        feature_rows.append(scenario_features(forecast))
        targets.append(float(due.sum()))
        scenario_names.append(str(scenario["name"]))
        scenario_origins.append(start.normalize().isoformat())
        examples.append(
            {
                "ranked_ids": ranked_ids,
                "due": due,
                "days_to_eol": days_to_eol.astype(float),
            }
        )
        print(
            f"{scenario['name']:>5} due={int(due.sum()):2d} "
            f"raw_expected={float(probability.sum()):5.2f}",
            flush=True,
        )

    x = np.vstack(feature_rows)
    y = np.asarray(targets, dtype=float)
    groups = np.arange(len(y)) // 8
    model_reports = []
    fitted_candidates = []
    for name, estimator in candidates():
        prediction = cross_fitted_predictions(estimator, x, y, groups)
        report = {
            "name": name,
            "mae": round(float(np.mean(np.abs(prediction - y))), 4),
            "rmse": round(float(np.sqrt(np.mean((prediction - y) ** 2))), 4),
            "bias": round(float(np.mean(prediction - y)), 4),
            "predicted_mean": round(float(prediction.mean()), 4),
        }
        model_reports.append(report)
        fitted_candidates.append((report["mae"], name, estimator, prediction))

    raw_prediction = x[:, scenario_feature_names().index("expected_due_f5")]
    model_reports.append(
        {
            "name": "raw_v7_expected_due",
            "mae": round(float(np.mean(np.abs(raw_prediction - y))), 4),
            "rmse": round(float(np.sqrt(np.mean((raw_prediction - y) ** 2))), 4),
            "bias": round(float(np.mean(raw_prediction - y)), 4),
            "predicted_mean": round(float(raw_prediction.mean()), 4),
        }
    )
    _, selected_name, selected_estimator, oof_prediction = min(
        fitted_candidates, key=lambda item: (item[0], item[1])
    )

    policies = []
    for multiplier in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        for buffer in (0.0, 2.0, 4.0, 6.0):
            policies.append(
                proxy_policy_score(
                    examples,
                    oof_prediction,
                    multiplier,
                    buffer,
                    args.maximum_budget,
                )
            )
    selected_policy = min(
        policies, key=lambda row: (row["timing_proxy"], row["mean_budget"])
    )
    selected_estimator.fit(x, y)
    model = ScenarioIncidenceModel(
        estimator=selected_estimator,
        feature_names=scenario_feature_names(),
        service_multiplier=args.service_multiplier,
        service_buffer=args.service_buffer,
        minimum_budget=2,
        maximum_budget=args.maximum_budget,
    )
    oof_model = CrossFittedIncidenceModel(
        predicted_due_by_origin={
            origin: float(predicted)
            for origin, predicted in zip(scenario_origins, oof_prediction)
        },
        service_multiplier=args.service_multiplier,
        service_buffer=args.service_buffer,
        minimum_budget=2,
        maximum_budget=args.maximum_budget,
    )

    report = {
        "n_scenarios": len(y),
        "actual_due_mean": round(float(y.mean()), 4),
        "blocked_groups": groups.tolist(),
        "selected_model": selected_name,
        "models": sorted(model_reports, key=lambda row: row["mae"]),
        "selected_policy": selected_policy,
        "deployment_policy": {
            "multiplier": args.service_multiplier,
            "buffer": args.service_buffer,
            "selection_basis": "representative end-to-end planner screen",
        },
        "policies": sorted(policies, key=lambda row: row["timing_proxy"]),
        "scenario_predictions": [
            {
                "scenario": name,
                "actual_due": int(actual),
                "predicted_due": round(float(predicted), 4),
            }
            for name, actual, predicted in zip(scenario_names, y, oof_prediction)
        ],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.oof_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(oof_model, args.oof_model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved V9 incidence model -> {args.model}")


if __name__ == "__main__":
    main()
