"""Train the Task 1 risk forecaster and serialize models/risk_forecaster.pkl.

Usage (from repository root):

    python -m src.risk.train --dataset-path data/raw/train --out-path models/risk_forecaster.pkl

Reproducibility: fixed random seed (default 20260818, matching the Task 2
planner's default), deterministic building-grouped fold assignment, and a
deterministic synthetic-cutoff grid derived only from the dataset's own
timestamps. Re-running with the same dataset and arguments reproduces the
same artifact.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
from batteryswap_public.utils import load_dataset

from .cutoffs import build_cutoff_grid, time_holdout_mask
from .model import HORIZONS_FOR_METRICS, derive_design, fit_task1_forecaster
from .cutoffs import build_example_table
from .features import build_feature_series

DEFAULT_SEED = 20260818


def _time_holdout_report(table: pd.DataFrame, forecaster, report: dict) -> dict:
    """Secondary diagnostic: performance on the temporally latest cutoffs only.

    This is never used for fitting or calibration (see cutoffs.time_holdout_mask
    docstring); it is a stress check that the causal building-grouped OOF score
    is not silently masking a time-drift problem.
    """

    from .model import evaluate_predictions  # local import to avoid a cycle at module load

    holdout_mask = time_holdout_mask(table).to_numpy()
    if holdout_mask.sum() == 0:
        return {}
    design = derive_design(table)
    scaled = forecaster.transform.transform(design.loc[holdout_mask])
    times = [float(h) for h in HORIZONS_FOR_METRICS]
    survival = forecaster.aft_model.predict_survival_function(scaled, times=times)
    raw_cdf = 1.0 - survival.to_numpy()  # shape (n_horizons, n_holdout_rows)
    calibrated = forecaster.calibrator.apply(raw_cdf)
    holdout_table = table.loc[holdout_mask].reset_index(drop=True)
    predicted_by_horizon = {
        horizon: calibrated[index, :] for index, horizon in enumerate(HORIZONS_FOR_METRICS)
    }
    metrics = evaluate_predictions(holdout_table, predicted_by_horizon)
    return {"brier": metrics["brier"], "log_loss": metrics["log_loss"], "n_examples": int(holdout_mask.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--out-path", type=Path, default=Path("models/risk_forecaster.pkl"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/task1_training_report.json"))
    parser.add_argument("--synthetic-step-days", type=int, default=21)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    started = time.perf_counter()
    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    scenario_starts = [pd.Timestamp(scenario["start_time"]) for scenario in scenarios]

    start_times = pd.to_datetime(locations["start_time"])
    if start_times.dt.tz is not None:
        start_times = start_times.dt.tz_localize(None)
    timeline_start = min(start_times.min(), min(scenario_starts))
    timeline_end = max(scenario_starts)
    cutoff_dates = build_cutoff_grid(
        scenario_starts, timeline_start, timeline_end, step_days=args.synthetic_step_days
    )

    forecaster, report = fit_task1_forecaster(
        locations, eol_times, timeseries, cutoff_dates, n_folds=args.n_folds, seed=args.seed
    )

    feature_series = build_feature_series(timeseries)
    table = build_example_table(locations, eol_times, feature_series, cutoff_dates)
    report["time_holdout"] = _time_holdout_report(table, forecaster, report)

    report["config"] = {
        "dataset_path": str(args.dataset_path),
        "synthetic_step_days": args.synthetic_step_days,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_official_scenarios": len(scenario_starts),
        "n_cutoff_dates": int(len(cutoff_dates)),
        "curated_features": list(forecaster.transform.columns),
        "model_version": forecaster.model_version,
    }
    report["elapsed_seconds"] = time.perf_counter() - started

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as handle:
        pickle.dump(forecaster, handle)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"Selected model: {forecaster.model_family} (penalizer={forecaster.penalizer})")
    print(f"CV concordance: {report['cv_concordance']:.4f}")
    print(f"CV brier by horizon: {report['cv_brier_by_horizon']}")
    print(f"Examples: {report['n_examples']} rows, {report['n_devices']} devices, {report['n_events']} events")
    print(f"Saved forecaster -> {args.out_path}")
    print(f"Saved report -> {args.report_path}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
