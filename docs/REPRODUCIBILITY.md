# Reproducibility guide

## Environment

Use Python **3.12**, the version checked locally and configured in CI. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.portfolio.txt
```

`requirements.portfolio.txt` pins the direct dependencies for the current model, planner, demo, and core tests. It is not a fully locked transitive environment. `requirements.txt` describes the organizer-provided competition environment and is retained separately; its broad package list includes tools the shipped solution does not use. `requirements.dev.txt` contains additional research dependencies for older experiments.

The dataset-independent demo works without Git LFS or competition data:

```bash
python tools/demo.py
```

## Shipped model

The repository uses Git LFS for model binaries. A normal clone without LFS may contain a small text pointer instead of an actual model:

```bash
git lfs install
git lfs pull --include="models/v7_wiener.joblib"
python tools/demo.py --model
```

See the [model card](../models/README.md) for artifact identity and limitations. Load only trusted joblib/pickle artifacts.

## Competition training data

Obtain the official dataset through the competition's authorized distribution; it is not bundled in this repository. The expected training layout is:

```text
dataset/train/
  battery_metrics.parquet
  devices.csv
  eol_times.csv
  scenarios.json
```

The [challenge reference](OFFICIAL_CHALLENGE_REFERENCE.md) contains organizer links and dataset context. Keep competition data out of Git.

## Current shipped settings

The source of truth is [`bsai/config.py`](../bsai/config.py). Submission defaults, training artifact scale, and validation defaults share these values:

| Setting | Value |
|---|---:|
| Model artifact | `models/v7_wiener.joblib` |
| Wiener volatility scale | 1.0 |
| Probability multiplier | 1.5 |
| CP-SAT seconds | 1.0 |
| Local / uncertain search evaluations | 240 / 240 |
| Robust emergency samples | 0 |
| Candidate margin hours | 12.0 |

The near-threshold transmission veto is defined in `bsai/forecaster.py`: more than 14 days stale **and** voltage margin below 0.1 V.

The portfolio update also corrects a probability-accounting edge case: when observation ends inside the planning window, mass removed by the transmission veto is assigned to unobserved failure rather than to an impossible recorded event after the window. This passes the existing censoring invariant but has not been rescored on the leaderboard. Historical scores remain attributed to their original submissions.

## Retrain without overwriting the evidence

```bash
python tools/train_wiener.py --stride 4 --out outputs/retrained_wiener.joblib --folds-out outputs/retrained_folds.joblib --report outputs/retrained_training_report.json
```

This performs building-grouped cross-validation, reports ranking/calibration metrics, and fits the production artifact on all training buildings. It saves a volatility scale of **1.0** by default, matching the shipped configuration. The best calibration-gap scale is still reported separately as `calibration_selected_scale`; research can explicitly choose another saved scale using `--volatility-scale`.

The historical training report selected 1.4, while the shipped artifact used 1.0. Earlier source saved the selected scale automatically. That discrepancy is corrected in the current tool; the committed historical report remains unchanged as evidence. Retraining was not rerun for this documentation update, and byte-identical artifacts across library versions are not promised.

## Evaluate the complete system

Out-of-fold evaluation after the training command above:

```bash
python tools/validate_v6.py --folds outputs/retrained_folds.joblib --model outputs/retrained_wiener.joblib --report outputs/validation_oof.json
```

Despite its historical filename, `validate_v6.py` now defaults to the shipped Wiener configuration. Each device is forecast using a model that held out its building. Defaults include probability scale 1.5, volatility scale 1.0, robust samples 0, candidate margin 12, and search budgets 240/240.

Inspect the committed production artifact on training scenarios:

```bash
python tools/validate_v6.py --production --limit 2 --report outputs/validation_smoke.json
```

This is **in-fold diagnostic evaluation**, because the production model saw these buildings during fitting. It validates integration and configuration; it is not a generalization estimate. Both modes honor `--volatility-scale`, and reports save mode, full configuration, per-scenario costs, and timing. Runs fail if the planner logs an error fallback.

The historical results in [RESULTS.md](RESULTS.md) were generated with older configurations. Do not expect the current defaults to reproduce every historical table. To compare a change, hold the model, mode, settings, and scenario subset fixed and save both reports. Six-week windows starting weekly overlap heavily; the tool reports block summaries as well as scenario means.

## Generate a competition-format submission

macOS / Linux:

```bash
BATTERYSWAP_DATASET_PATH=dataset/train BATTERYSWAP_SPLITS=train python script.py
```

Windows PowerShell:

```powershell
$env:BATTERYSWAP_DATASET_PATH = "dataset/train"
$env:BATTERYSWAP_SPLITS = "train"
python script.py
```

This writes `submission.csv`. The official entry point defaults to `/tmp/data` and `public,private`; local training-data smoke checks should explicitly set the overrides above. Keep one planner instance across scenarios to reuse incremental feature caches.

`Dockerfile` retains the organizer image and submission entry point. Docker builds require the actual LFS artifact. The upstream image uses `latest`, so it is not a frozen environment and a Docker build was not verified as part of the portfolio update.

## Tests

The same dataset-independent suite used in GitHub Actions:

```bash
python -m unittest tests.test_task2 tests.test_runtime tests.test_task1_forecast tests.test_candidates tests.test_blend tests.test_configuration -v
python tools/demo.py
python tools/check_portfolio.py
```

With data and weights available, run the full suite:

```bash
python -m unittest discover -s tests -v
```

Some full-suite tests skip when their required data is absent. Coverage includes smoothing parity with the official function, forecast contracts, observation censoring, routing, plan completeness, operational replay parity, runtime handling, and production-validation scale overrides. CI checks avoid loading LFS pointer files by selecting dataset-independent modules explicitly.

For stable local runtimes, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, as CI does. A fixed seed does not guarantee byte-identical CP-SAT plans under a wall-clock search limit or across platforms.

## Figures

The static README chart is generated from `docs/results.json`:

```bash
python -m pip install matplotlib
python tools/render_results.py
```

Changing source scores should update the ledger, results narrative, and chart together. The check script verifies headline calculations and current documentation links; historical notes retain their original context.
