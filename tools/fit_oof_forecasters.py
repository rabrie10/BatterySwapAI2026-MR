"""Build an out-of-fold forecaster that never predicts a battery from a model
that has seen that battery's building.

Why this exists
---------------
Both the in-repo benchmarks (`tools/benchmark_task2.py --mode real`) and the
tactical experiment suite fit Task 1 on *all* train devices and then score it
on train scenarios. Model selection inside those fits is building-grouped and
therefore honest, but the end-to-end `total_cost` they report is in-sample:
every battery being forecast was in the training set.

The competition is not in-sample. Public and private contain different
buildings entirely, and the 2026-08-19 submission showed the gap is material:
the mixture-cure model scored 2648.61 locally but 4252.33 on the public split,
with the swap count rising from ~11 to ~41 per scenario. That is the signature
of an incidence model that is over-confident on buildings it has never seen.

`docs/SOLUTION_DESIGN_SPEC.md` Sec 8.1 requires exactly this harness
("Evaluate only with out-of-fold forecasts when tuning Task 2"). It produces a
forecaster that, for every battery, delegates to the fold model trained
*without* that battery's building -- so a train-scenario benchmark becomes an
honest estimate of unseen-building behaviour, and can be used to select
conservatism without spending official submissions.

Output: `models/risk_forecaster_oof.pkl`, a drop-in `RiskForecaster`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_dataset  # noqa: E402

from src.risk.cutoffs import assign_building_folds, build_cutoff_grid, build_example_table  # noqa: E402
from src.risk.features import build_feature_series  # noqa: E402
from src.risk.oof import OutOfFoldForecaster  # noqa: E402
from src.risk.model import (  # noqa: E402
    CURATED_FEATURES,
    Task1Forecaster,
    _build_lifelines_frame,
    derive_design,
    fit_aft,
    fit_feature_transform,
    fit_incidence_classifier,
    fit_platt_calibrator,
)


def fit_fold_forecaster(table, design, train_mask, observation_end, args, template):
    """Fit AFT + incidence on one fold's training rows only."""

    transform = fit_feature_transform(design.loc[train_mask], CURATED_FEATURES)
    scaled = transform.transform(design.loc[train_mask])
    frame = _build_lifelines_frame(scaled, table.loc[train_mask])
    aft = fit_aft(template.model_family, frame, template.penalizer)

    incidence_model, incidence_transform, _ = fit_incidence_classifier(
        table.loc[train_mask].reset_index(drop=True),
        observation_end,
        n_folds=args.inner_folds,
        seed=args.seed,
        regularization_grid=args.regularization,
        weighting=template.incidence_weighting,
        use_remaining_window=bool(
            getattr(template, "incidence_uses_remaining_window", True)
        ),
    )
    return replace(
        template,
        aft_model=aft,
        transform=transform,
        incidence_model=incidence_model,
        incidence_transform=incidence_transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--template", type=Path, default=Path("models/risk_forecaster.pkl"),
                        help="Artifact whose hyperparameters/config are reused per fold")
    parser.add_argument("--out-path", type=Path, default=Path("models/risk_forecaster_oof.pkl"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/oof_harness_report.json"))
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--synthetic-step-days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--regularization", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()

    started = time.perf_counter()
    with args.template.open("rb") as handle:
        template = pickle.load(handle)
    print(f"template: {getattr(template, 'model_version', '?')}", flush=True)

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset_path)
    scenario_starts = [pd.Timestamp(s["start_time"]) for s in scenarios]
    start_times = pd.to_datetime(locations["start_time"])
    if start_times.dt.tz is not None:
        start_times = start_times.dt.tz_localize(None)
    end_times = pd.to_datetime(locations["end_time"])
    if end_times.dt.tz is not None:
        end_times = end_times.dt.tz_localize(None)
    observation_end = end_times.max().normalize()

    cutoff_dates = build_cutoff_grid(
        scenario_starts, min(start_times.min(), min(scenario_starts)),
        max(scenario_starts), step_days=args.synthetic_step_days,
    )
    table = build_example_table(
        locations, eol_times, build_feature_series(timeseries), cutoff_dates
    )
    design = derive_design(table)
    folds = assign_building_folds(table, n_folds=args.n_folds, seed=args.seed).to_numpy()

    building_to_fold = (
        table[["building_id"]].assign(fold=folds)
        .drop_duplicates("building_id").set_index("building_id")["fold"].astype(int).to_dict()
    )

    fold_forecasters: dict[int, Task1Forecaster] = {}
    for fold in range(args.n_folds):
        test_mask = folds == fold
        train_mask = ~test_mask
        n_buildings = len({b for b, f in building_to_fold.items() if f == fold})
        if test_mask.sum() == 0 or table.loc[train_mask, "event"].sum() == 0:
            print(f"fold {fold}: skipped (insufficient data)", flush=True)
            continue
        t0 = time.perf_counter()
        fold_forecasters[fold] = fit_fold_forecaster(
            table, design, train_mask, observation_end, args, template
        )
        print(
            f"fold {fold}: held out {n_buildings} buildings / {int(test_mask.sum())} rows "
            f"({time.perf_counter()-t0:.0f}s)",
            flush=True,
        )

    forecaster = OutOfFoldForecaster(fold_forecasters, building_to_fold, template)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as handle:
        pickle.dump(forecaster, handle)

    report = {
        "template_version": getattr(template, "model_version", "?"),
        "n_folds": args.n_folds,
        "folds_fitted": sorted(fold_forecasters),
        "n_buildings": len(building_to_fold),
        "buildings_per_fold": {
            str(f): int(sum(1 for v in building_to_fold.values() if v == f))
            for f in sorted(set(building_to_fold.values()))
        },
        "n_examples": int(len(table)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved OOF forecaster -> {args.out_path}")


if __name__ == "__main__":
    main()
