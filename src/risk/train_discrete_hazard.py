"""Train the separate H1 boosted daily-hazard challenger artifact.

Example:

    python -m src.risk.train_discrete_hazard \
      --dataset-path data/raw/train \
      --out-path models/risk_forecaster_discrete_hazard.pkl \
      --report-path docs/task1_discrete_hazard_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from batteryswap_public.utils import load_dataset

from .cutoffs import build_cutoff_grid
from .discrete_hazard import fit_discrete_hazard_forecaster


DEFAULT_SEED = 20260818
DEFAULT_GRID = (
    {
        "learning_rate": 0.06,
        "max_iter": 140,
        "max_leaf_nodes": 15,
        "max_depth": 5,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
    },
    {
        "learning_rate": 0.04,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "max_depth": 6,
        "min_samples_leaf": 200,
        "l2_regularization": 2.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/raw/train"))
    parser.add_argument(
        "--out-path", type=Path, default=Path("models/risk_forecaster_discrete_hazard.pkl")
    )
    parser.add_argument(
        "--report-path", type=Path, default=Path("docs/task1_discrete_hazard_report.json")
    )
    parser.add_argument("--synthetic-step-days", type=int, default=21)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-horizon", type=int, default=365)
    parser.add_argument("--weighting", choices=("interval", "normalized"), default="normalized")
    parser.add_argument(
        "--physical-uncertainty-days",
        type=float,
        default=0.0,
        help="0 evaluates the required no-blend H1 baseline; 20 applies the existing physical blend.",
    )
    parser.add_argument(
        "--grid",
        choices=("small", "single"),
        default="small",
        help="'single' runs only the first conservative configuration for smoke/reproduction.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    starts = [pd.Timestamp(scenario["start_time"]) for scenario in scenarios]
    deployment = pd.to_datetime(locations["start_time"])
    if deployment.dt.tz is not None:
        deployment = deployment.dt.tz_localize(None)
    cutoff_dates = build_cutoff_grid(
        starts,
        min(deployment.min(), min(starts)),
        max(starts),
        step_days=args.synthetic_step_days,
    )
    grid = DEFAULT_GRID[:1] if args.grid == "single" else DEFAULT_GRID
    forecaster, report = fit_discrete_hazard_forecaster(
        locations,
        eol_times,
        timeseries,
        cutoff_dates,
        parameter_grid=grid,
        n_folds=args.n_folds,
        seed=args.seed,
        max_horizon=args.max_horizon,
        weighting=args.weighting,
        physical_uncertainty_days=args.physical_uncertainty_days,
    )
    report["config"] = {
        "dataset_path": str(args.dataset_path),
        "synthetic_step_days": args.synthetic_step_days,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "max_horizon": args.max_horizon,
        "weighting": args.weighting,
        "physical_uncertainty_days": args.physical_uncertainty_days,
        "grid": list(grid),
        "n_official_scenarios": len(starts),
        "n_cutoff_dates": int(len(cutoff_dates)),
        "model_version": forecaster.model_version,
    }
    report["elapsed_seconds"] = time.perf_counter() - started

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as handle:
        pickle.dump(forecaster, handle)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"Selected params: {report['selected_params']}")
    print(f"Weighting: {args.weighting}; physical blend scale: {args.physical_uncertainty_days}")
    print(f"OOF Brier: {report['oof_metrics']['brier']}")
    print(
        f"Landmarks={report['n_landmarks']}, hazard rows={report['n_hazard_rows']}, "
        f"event rows={report['n_event_rows']}, unique EOL={report['n_unique_eol_events']}"
    )
    print(f"Saved challenger -> {args.out_path}")
    print(f"Saved report -> {args.report_path}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
