# H1 Discrete-Time Hazard Challenger

Status: promoted to the submission default after local evaluation on 2026-08-20.
The existing AFT implementation and artifact remain unchanged as an automatic
packaging fallback.

## Formulation

For a causal `(battery, cutoff)` landmark, the model holds the six existing H1
features fixed and expands the known future into daily intervals. It fits a
`HistGradientBoostingClassifier` to

```text
h_k = P(T in (k - 1, k] | T > k - 1, x_cutoff)
```

and converts hazards into a CDF with

```text
S(k) = product(1 - h_j, j=1..k)
F(k) = 1 - S(k)
```

Observed EOL contributes exactly one positive interval. Censored landmarks
contribute negatives only for fully observed intervals through
`floor(duration_days)`; the unknown partial interval after censoring is not
labelled. The fitted horizon is capped at 365 days.

H1 uses `latest_voltage`, `voltage_slope_28d`, `frac_low_voltage_28d`,
`age_days`, `not_yet_deployed`, `cold_start`, and `horizon_day`. Buildings are
held intact across four folds. Raw out-of-fold hazards are converted to CDFs,
then the existing horizon-conditional isotonic approach is fitted at
7/14/21/28/35/42/60/90/120/180/240/300/365 days. Inference interpolates these
maps and applies cumulative maximum plus `[0, 1]` clipping.

## Weighting

Both `interval` and `normalized` modes are implemented. H1 evaluates
`normalized` weighting:

```text
landmark device weight / number of known intervals for that landmark
```

The interval-likelihood alternative repeats the existing per-device landmark
weight on every daily row. It is statistically conventional, but in this
landmark-expanded dataset it lets batteries with long known follow-up dominate
twice: once through many cutoffs and again through many intervals. Normalized
weighting was therefore chosen for H1; this is an experimental estimand and is
not claimed to be universally correct.

## Results

Training data: 461 batteries, 82 unique EOL events, 48,059 landmarks,
12,238,183 hazard rows, 3,125 positive interval rows, 82.21% device censoring.
Expanded positives are correlated landmark views of 82 events, not independent
physical failures.

| Horizon | OOF Brier | Trivial Brier | Ratio | Log loss | Cal. ratio | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.002906 | 0.003199 | 0.908 | 0.01152 | 0.998 | 0.982 |
| 14 | 0.005375 | 0.006352 | 0.846 | 0.01973 | 0.999 | 0.980 |
| 21 | 0.007945 | 0.009621 | 0.826 | 0.02857 | 0.997 | 0.976 |
| 28 | 0.010353 | 0.012874 | 0.804 | 0.03658 | 1.007 | 0.972 |
| 35 | 0.012682 | 0.015808 | 0.802 | 0.04383 | 0.997 | 0.969 |
| 42 | 0.014830 | 0.018821 | 0.788 | 0.05071 | 1.000 | 0.966 |

The latest 20% time subset uses building-grouped OOF predictions but is not a
strict train-past/test-future split. Its 42-day Brier is 0.02381 and calibration
ratio is 1.626, indicating temporal overprediction drift.

| Model | Total | Early | Late | Swaps | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing log-normal AFT v4 | 3128.884 | 1589.719 | 1072.500 | 27.958 | 556.2s |
| H1 hazard, normalized, no blend (submission sklearn 1.7.2) | **3092.082** | 1861.427 | **691.875** | 35.771 | **1051.9s** |
| All defer | 3324.679 | — | — | 0 | — |

The full benchmark is in-sample because the final artifact was fitted on all
train batteries and then evaluated on train scenarios. It is directional, not
an honest generalization estimate. A fold-routed OOF planner benchmark remains
the main missing decision-grade experiment.

## Reproduction

The checked-in submission artifact was trained with scikit-learn 1.7.2 inside
the submission image. Keep training and inference on the same minor version;
the development environment may resolve a newer scikit-learn release that
cannot safely deserialize this artifact.

```powershell
docker build -t batteryswap-hazard-submission:local .

docker run --rm -v "${PWD}/data/raw:/tmp/data:ro" -v "${PWD}/models:/outmodels" -v "${PWD}/docs:/outdocs" --entrypoint python3 batteryswap-hazard-submission:local -m src.risk.train_discrete_hazard --dataset-path /tmp/data/train --out-path /outmodels/risk_forecaster_discrete_hazard.pkl --report-path /outdocs/task1_discrete_hazard_report.json --synthetic-step-days 21 --n-folds 4 --seed 20260818 --max-horizon 365 --weighting normalized --physical-uncertainty-days 0 --grid single

docker build -t batteryswap-hazard-submission:local .

.\venv\Scripts\python.exe -m unittest tests.test_discrete_hazard
.\venv\Scripts\python.exe -m unittest tests.test_task1_forecast tests.test_task2

docker run --rm -v "${PWD}/data/raw/train:/tmp/train:ro" -v "${PWD}/tools:/app/tools:ro" -v "${PWD}/docs:/tmp/docs" --entrypoint python3 batteryswap-hazard-submission:local -m tools.benchmark_task2 --dataset-path /tmp/train --limit 0 --solver-seconds 2 --local-search 160 --robust-samples 4 --late-risk-multiplier 1 --mode real --forecaster-path models/risk_forecaster_discrete_hazard.pkl --quiet --record /tmp/docs/local_benchmark_hazard_log.csv --label "H1 normalized no physical blend"
```

## Recommendation

Use H1 as the submission default while retaining AFT as the packaged fallback.
It passes the contract, improves grouped OOF statistics, beats the in-sample
AFT planner benchmark by 36.802 cost units (1.18%), and completes the 48-scenario
run within the 30-minute limit. The margin is modest and temporal calibration
drift remains visible, so a fold-routed OOF end-to-end planner benchmark is the
next experiment rather than a prerequisite for this submission. H2 and the
physical blend were not tested; H1 no-blend already reduced late cost, so the
requested priority rule did not justify adding a blend.
