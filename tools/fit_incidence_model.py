"""Attach a grouped-CV cure-incidence model to an existing Task 1 artifact."""

from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
import json
import pickle
from pathlib import Path
import sys
import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_dataset

from src.risk.cutoffs import build_cutoff_grid, build_example_table
from src.risk.features import build_feature_series
from src.risk.model import (
    INCIDENCE_WEIGHTING_MODES,
    ROBUST_CURE_MODEL_VERSION,
    fit_horizon_classifier,
    fit_incidence_classifier,
    fit_horizon_rate_calibrator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset/train"))
    parser.add_argument(
        "--base-forecaster", type=Path, default=Path("models/risk_forecaster.pkl")
    )
    parser.add_argument(
        "--out-path", type=Path, default=Path("models/risk_forecaster.pkl")
    )
    parser.add_argument(
        "--report-path", type=Path, default=Path("docs/incidence_training_report.json")
    )
    parser.add_argument("--synthetic-step-days", type=int, default=21)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument(
        "--incidence-weighting",
        choices=INCIDENCE_WEIGHTING_MODES,
        default="cutoff",
    )
    parser.add_argument("--physical-uncertainty-days", type=float, default=1.0)
    parser.add_argument("--physical-risk-weight", type=float, default=0.6)
    parser.add_argument(
        "--physical-shape-min-remaining-days", type=float, default=210.0
    )
    parser.add_argument("--horizon-rate-smoothing-window", type=int, default=9)
    parser.add_argument("--horizon-rate-cap-multiplier", type=float, default=1.0)
    parser.add_argument("--horizon-rate-activation-ratio", type=float, default=1.9)
    parser.add_argument("--direct-horizon-weight", type=float, default=0.0)
    args = parser.parse_args()

    if not 0.0 <= args.physical_risk_weight <= 1.0:
        parser.error("--physical-risk-weight must be between 0 and 1")
    if (
        args.horizon_rate_smoothing_window < 1
        or args.horizon_rate_smoothing_window % 2 == 0
    ):
        parser.error("--horizon-rate-smoothing-window must be a positive odd integer")
    if args.horizon_rate_cap_multiplier <= 0.0:
        parser.error("--horizon-rate-cap-multiplier must be positive")
    if args.horizon_rate_activation_ratio < 1.0:
        parser.error("--horizon-rate-activation-ratio must be at least 1.0")
    if not 0.0 <= args.direct_horizon_weight <= 1.0:
        parser.error("--direct-horizon-weight must be between 0 and 1")

    started = time.perf_counter()
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    scenario_starts = [pd.Timestamp(scenario["start_time"]) for scenario in scenarios]
    start_times = pd.to_datetime(locations["start_time"])
    if start_times.dt.tz is not None:
        start_times = start_times.dt.tz_localize(None)
    end_times = pd.to_datetime(locations["end_time"])
    if end_times.dt.tz is not None:
        end_times = end_times.dt.tz_localize(None)
    observation_end = end_times.max().normalize()
    cutoff_dates = build_cutoff_grid(
        scenario_starts,
        min(start_times.min(), min(scenario_starts)),
        max(scenario_starts),
        step_days=args.synthetic_step_days,
    )

    feature_series = build_feature_series(timeseries)
    table = build_example_table(locations, eol_times, feature_series, cutoff_dates)
    incidence_model, incidence_transform, report = fit_incidence_classifier(
        table,
        observation_end,
        n_folds=args.n_folds,
        seed=args.seed,
        regularization_grid=args.regularization,
        weighting=args.incidence_weighting,
    )
    horizon_model, horizon_transform, horizon_report = fit_horizon_classifier(
        table,
        observation_end,
        horizon_days=42,
        n_folds=args.n_folds,
        seed=args.seed,
        regularization_grid=args.regularization,
    )
    horizon_rate_calibrator, horizon_rate_table = fit_horizon_rate_calibrator(
        locations,
        eol_times,
        scenarios,
        observation_end,
        smoothing_window=args.horizon_rate_smoothing_window,
        cap_multiplier=args.horizon_rate_cap_multiplier,
        activation_ratio=args.horizon_rate_activation_ratio,
    )

    with args.base_forecaster.open("rb") as handle:
        forecaster = pickle.load(handle)
    if not is_dataclass(forecaster):
        raise TypeError("Base forecaster must be a dataclass artifact")
    forecaster = replace(
        forecaster,
        model_version=ROBUST_CURE_MODEL_VERSION,
        physical_uncertainty_days=float(args.physical_uncertainty_days),
        physical_risk_weight=float(args.physical_risk_weight),
        physical_shape_min_remaining_days=float(
            args.physical_shape_min_remaining_days
        ),
        incidence_model=incidence_model,
        incidence_transform=incidence_transform,
        incidence_weighting=args.incidence_weighting,
        horizon_rate_calibrator=horizon_rate_calibrator,
        horizon_model=horizon_model,
        horizon_transform=horizon_transform,
        direct_horizon_days=42,
        direct_horizon_weight=float(args.direct_horizon_weight),
    )

    report["config"] = {
        "dataset_path": str(args.dataset_path),
        "base_forecaster": str(args.base_forecaster),
        "synthetic_step_days": args.synthetic_step_days,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "regularization_grid": args.regularization,
        "incidence_weighting": args.incidence_weighting,
        "observation_end": observation_end,
        "physical_uncertainty_days": args.physical_uncertainty_days,
        "physical_risk_weight": args.physical_risk_weight,
        "physical_shape_min_remaining_days": args.physical_shape_min_remaining_days,
        "horizon_rate_smoothing_window": args.horizon_rate_smoothing_window,
        "horizon_rate_cap_multiplier": args.horizon_rate_cap_multiplier,
        "horizon_rate_activation_ratio": args.horizon_rate_activation_ratio,
        "direct_horizon_weight": args.direct_horizon_weight,
        "model_version": ROBUST_CURE_MODEL_VERSION,
    }
    report["horizon_rate_calibration"] = horizon_rate_table.to_dict(
        orient="records"
    )
    report["direct_horizon_model"] = horizon_report
    report["n_examples"] = int(len(table))
    report["elapsed_seconds"] = float(time.perf_counter() - started)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as handle:
        pickle.dump(forecaster, handle)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved cure forecaster -> {args.out_path}")


if __name__ == "__main__":
    main()
