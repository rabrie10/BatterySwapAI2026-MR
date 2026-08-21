# V9 incidence-ranked portfolio experiment

## Purpose

V9 tests a different decomposition of the decision problem:

1. Current main's calibrated V7 model forecasts and ranks battery risks.
2. An independent scenario model estimates how many batteries will reach EOL.
3. A broad candidate pool (`ceil(2 * predicted_due + 6)`, capped at 40) is
   passed to the planner.
4. The optimizer may schedule at most `ceil(1.25 * predicted_due + 6)`, capped
   at 24. It may schedule fewer when expected service cost is not beneficial.

The incidence model never uses scenario identity or future data as a feature.
It is a Poisson regression trained on aggregate statistics of the calibrated
V7 forecast. Model selection and validation predictions use six chronological
groups of eight scenarios. The underlying remaining-observation calibration is
also out-of-fold by building.

## Honest validation result

Current main already fixes most aggregate calibration error, so the independent
count model provides only a modest additional improvement:

| Count estimator | OOF MAE | OOF bias |
| --- | ---: | ---: |
| V9 Poisson | 2.36 | -0.17 |
| Sum of calibrated V7 probabilities | 2.85 | -0.05 |

A fast end-to-end screen used scenarios 0, 3, 6, 20, 23, 26, 40, 43, and 46
and computed every expensive forecast once. It included ordinary current main
as a same-run control:

| Policy | Mean total | Early | Late | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| V9 `1.25 * K + 6` | 2279.66 | 623.06 | 1243.33 | 0.535 | 0.365 |
| Current main control | 2296.39 | 736.78 | 1122.22 | 0.581 | 0.299 |
| V9 `1.25 * K` | 2388.06 | 505.39 | 1514.44 | 0.442 | 0.384 |

The selected policy improves this screen by only 16.73 points, far inside the
repository's roughly 100-point promotion/noise threshold. A full 48-scenario
planner run was therefore not used to claim promotion. This is an experimental
submission branch, not a proven replacement for current main.

The result is still informative: scenario-level incidence improves precision,
but aggressive caps lose too much recall because the battery-level ordering is
still wrong. The next high-value route is a directly trained pairwise/listwise
or cost-weighted battery ranker, not further tuning of the count multiplier.

## Reproduction

Train the incidence artifact after producing calibrated V7 folds:

```powershell
.\venv\Scripts\python.exe tools\train_portfolio.py
```

Run the cached-forecast screen with its ordinary-main control:

```powershell
.\venv\Scripts\python.exe tools\screen_portfolio.py
```

`script.py` loads `models/v7_wiener.joblib` and
`models/v9_incidence.joblib`. If the V9 artifact is absent, it safely falls
back to ordinary V7 decisions.
