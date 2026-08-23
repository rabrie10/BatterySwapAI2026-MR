---
license: mit
---

# BatterySwapAI 2026

Competition solution for RUL forecasting and cost-aware battery-swap planning.
The submitted entry point is `script.py`.

- **Task 1** is `bsai/`: a Wiener first-passage model for the 2.4 V crossing,
  driven by within-day features the official smoothing discards.
- **Task 2** is `batteryswap_solution/`: the scheduling and routing optimizer.

They meet at the versioned forecast contract in
`batteryswap_solution/forecast.py`, so the model can be replaced without
touching any scheduling code.

This README describes what currently ships, including the two changes made after
the V7/V8 model work: the stale-transmission veto and the probability scale. The
model itself, its measurements and the approaches that failed are in
[`docs/V7_IMPLEMENTATION.md`](docs/V7_IMPLEMENTATION.md) and
[`docs/PLAN_V7_MARGIN.md`](docs/PLAN_V7_MARGIN.md); the previous generation is in
[`docs/V6_IMPLEMENTATION.md`](docs/V6_IMPLEMENTATION.md). Where those documents
and this one disagree about what is shipped, this one is current — they predate
the veto and the scale.

## Where it stands

What `script.py` actually ships, and what each change scored on the public
leaderboard:

| commit | change | public |
|---|---|---:|
| `7792b78` | V8 Wiener model + V10 planner budgets | 1985.43 |
| `89573f5` | + stale-transmission veto | **1555.80** |
| `37ae686` | + probability scale 1.5 | not yet scored |

The shipped configuration, all of it explicit in source rather than defaulted:

| setting | value | where |
|---|---|---|
| Task 1 artifact | `models/v7_wiener.joblib` | `script.py: DEFAULT_MODEL_PATH` |
| probability scale | 1.5 | `script.py: PROBABILITY_SCALE` |
| volatility scale | 1.0 | `script.py: load_forecaster()` |
| stale-transmission veto | `v_stale_days > 14` and `margin < 0.1` | `bsai/forecaster.py` |
| solver seconds | 1.0 | `script.py` |
| local / uncertain search | 240 / 240 | `script.py` |
| robust emergency samples | 0 | `script.py` |

For reference, out-of-fold by building over all 48 train scenarios:

| configuration | mean total cost |
|---|---:|
| all-defer (service nothing) | 3324.7 |
| shipped v3 | 2644.9 |
| V6 hazard classifier | 2526.0 |
| V7 Wiener | 2293.2 |
| perfect knowledge, this planner (scenarios 0-11) | 77.8 |

**Local validation does not track public on the early/late axis.** In production
mode the totals agree closely (local 1937.57 against public 1985.43 at
`7792b78`), but the composition is inverted: locally the planner is late-heavy
(early 671 / late 1020), on public it is early-heavy (early 962 / late 658).
Tuning that axis against local validation points the wrong way, which is why the
veto was justified from a due-rate table rather than from a local cost delta.

Runtime projects to roughly 15-16 minutes for the 96 public and private
scenarios against the 30-minute limit.

The change that mattered was not the model class. A two-line control -- rank by
`margin / -slope`, no model at all -- matched V6 exactly (precision 0.309
against 0.300 at twelve swaps), so fifty-one features of gradient boosting were
worth nothing over a straight line. `smooth_series` collapses 8.5 million hourly
readings into 360,847 daily numbers, and every V6 feature was a function of that
one collapsed series. The within-day voltage response to the daily temperature
cycle -- a proxy for internal resistance, the documented knee precursor --
separates due from not-due with AUC 0.871 on exactly the population the smoothed
series cannot rank at all.

The local mean is a weak predictor of the leaderboard: v3 scored 2644.9 locally
and 4252.3 on public. What that public score actually reflected was
over-servicing in scenarios where the observation window is closing — 41 swaps
per scenario against roughly 9.5 due. That defect is measured and fixed here
(predicted-to-actual due ratio 2.21 -> ~1.0 in the closing scenarios), which is a
difference local validation cannot score.

## What the problem actually is

Lower score is better. Per scenario the evaluator charges work time, travel,
overtime, flat 100-point daily and weekly capacity penalties, 0.5 per day for
swapping early, and 10 per day for swapping late. A battery that reaches EOL
inside the 42-day window and is *not* serviced gets a dedicated emergency visit
after the window, costing between 60 and 480 hours.

On the train split roughly **9.5 batteries per scenario** genuinely reach EOL
inside the window, out of about 420 alive. Servicing nothing at all scores
3324.7. Perfect knowledge with this planner scores 77.8 on scenarios 0-11. The
whole competition is the question of *which* batteries to touch.

## Task 1: `bsai/`

| module | role |
|---|---|
| `smoothing.py` | Incremental, exact reimplementation of `smooth_series`. Pinned to the official function to 1e-12, including the partial-day boundary. |
| `features.py` | 64 causal features: the smoothed grid, plus the within-day statistics that grid discards. |
| `shape.py` | Incremental within-day statistics from the raw hourly readings: voltage spread and the dV/dT response to the daily temperature cycle. |
| `wiener.py` | Wiener first passage: learned drift and volatility, closed-form crossing probability. |
| `hazard.py` | Cutoff sampling and the previous multi-horizon classifier, kept for comparison. |
| `margin.py` | Quantile regression on the running minimum; measured, not shipped. |
| `forecaster.py` | Adapter to the Task 2 forecast contract: the censoring branch, the uniform probability scale, and the stale-transmission veto. |
| `validation.py` | Out-of-fold dispatch, so no device is ever scored by a model that saw its building. |
| `runtime.py` | Wall-clock governor for the 30-minute evaluation budget. |

Three design decisions that came from measurement rather than habit:

**The target is a distribution, not a number.** A mean RUL regressor measured on
train has MAE 25 days but is optimistic by +17 days exactly where it matters
(true RUL under 14 days) and pessimistic by 21 days at 70-120 days. That is
regression to the mean, in the direction that makes you late, and late costs
twenty times what early costs.

**Censoring is known, not estimated.** `locations.end_time` is handed to
`plan()`, so for each battery we know the last day an EOL could be recorded, and
the evaluator's substitute EOL for the unrecorded case is exactly
`normalize(end_time + unobserved_eol_days)`. The predicted CDF is capped at its
own value at that horizon. This is what makes the closing scenarios stop
demanding service on their own, with no hand-tuned survivor gate.

**Temperature is a first-order driver.** Within-device, residual voltage tracks
residual temperature at +0.00463 V/degC, positive in 100% of 454 train devices.
The 4.87 degC indoor annual swing is 0.023 V, which near the knee is about two
weeks of remaining life, and EOL incidence is 1.76x higher in Nov-Mar than in
May-Sep. Features include temperature-compensated levels and slopes plus the
expected seasonal temperature change across the planning window.

**A battery that has stopped transmitting cannot become due.** EOL is defined
from the timeseries, so a device with no recent readings cannot record a 2.4 V
crossing, and every swap spent on one is waste. Within the near-threshold band
the due rate falls from 30.8% while transmitting to about 5% past 14 days of
silence, against a break-even service probability near 15% (mean early cost per
wasted swap about 45, against about 261 per miss). `bsai/forecaster.py`
therefore vetoes service when `v_stale_days > 14` **and** `margin < 0.1`, where
`margin` is the last smoothed voltage minus 2.4 and `v_stale_days` is the gap
between the last day the smoothed series has a value and the cutoff. It fires on
about 9.6 batteries per scenario.

The effect exists only inside the near-threshold band — across the whole
population staleness barely moves the due rate (2.4% transmitting against
1.5-2.7% stale) — so this is a veto justified by cost asymmetry, not a ranking
feature. Both thresholds are fixed constants; neither is swept or configurable.

Note that `features.py` also exposes a feature named `staleness`, which is *not*
the same quantity: `DeviceView.value_at_or_before` clamps its index to the end
of the series, so it measures gaps *inside* the series and reads about zero for
exactly the stopped devices this rule targets. The veto computes silence against
the unclamped cutoff ordinal instead.

**Service volume is set by a uniform probability scale.** After the veto removed
the sub-break-even population, every serviced margin band clears break-even (the
marginal band at 0.273 precision) while service ran at 13.6 per scenario against
12.3 due, with late costing more than twice early. `PROBABILITY_SCALE = 1.5` in
`script.py` multiplies the whole CDF, raising service to roughly 19 per scenario.
The veto is applied *after* the scaling, so vetoed batteries are never
resurrected by it, and the veto condition depends only on `margin` and
`v_stale_days`, both independent of the scale.

## Task 2: `batteryswap_solution/`

- `forecast.py`: the versioned Task 1 -> Task 2 contract and its validation.
- `costs.py`: expected early, late, deferred-emergency and evaluator-unobserved
  timing costs, in the unconditional-pmf convention.
- `optimizer.py`: joint CP-SAT service/defer and day assignment under room,
  building, overtime, daily and weekly constraints.
- `routing.py`: exact Held-Karp routes for small days, insertion + 2-opt above.
- `replay.py`: evaluator-exact fast replay, checked against
  `batteryswap_public.evaluate_plan()` before every returned plan.
- `planner.py`: the official `Planner.plan()` adapter, local search, and an
  all-defer safety fallback.

## Reproducing the submission

Both commands are deterministic given the seeds in the source.

```bash
python tools/train_wiener.py --stride 4
```

Writes `models/v7_wiener.joblib` (the shipped artifact, fitted on every
building), `outputs/v7_folds.joblib` (the five fold models, used for validation
and not needed at submission time), and `docs/v7_training_report.json`.

**Known discrepancy — this command does not reproduce the shipped artifact
byte-for-byte.** `tools/train_wiener.py` selects `best_scale` by calibration gap
and assigns it to the model before dumping; that selection is **1.4**, while the
committed `models/v7_wiener.joblib` carries **1.0**. A model regenerated from
this source therefore arrives with a different volatility scale. Everything else
reproduces exactly: a regenerated run matches `docs/v7_training_report.json` in
every substantive field (n=88013, positives 862, AUC 0.9823, PR-AUC 0.4303,
predicted/actual 0.637, every precision@k), differing only in wall-clock seconds.

The submission is insulated from this: `script.py` pins `volatility_scale = 1.0`
explicitly after loading, so a regenerated artifact arriving at 1.4 cannot change
what ships. 1.0 is the value that produced the scores in the table above; 1.4
measured worse (+10.1 against baseline on 42 train scenarios) despite being
better calibrated in isolation (predicted/actual 1.025 against 0.637).

```bash
python tools/validate_v6.py --folds outputs/v7_folds.joblib     --model models/v7_wiener.joblib --volatility-scale 1.0
```

Scores the production planner over all 48 train scenarios using predictions from
models that never saw the device's own building, and prints the anchors so a
result is never read in isolation.

Two cautions when comparing a validation run against the shipped build:

- `tools/validate_v6.py` constructs its own `PlannerConfig` and does not set
  `robust_emergency_samples`, so it inherits the dataclass default of **4**,
  while `script.py` ships **0**. A run left on defaults is not measuring the
  submitted configuration.
- `--volatility-scale` is honoured only in the out-of-fold branch of
  `build_forecaster`; under `--production` the flag is ignored. A production-mode
  sweep over it measures nothing but wall-clock noise.

Out-of-fold numbers are also not comparable to the public leaderboard on
service volume: out-of-fold runs at roughly 29 swaps per scenario against the
shipped 19, which is a different operating regime, not a pessimistic estimate of
the same one. Use `--production` when the question is what the submission does.

Local submission generation:

```bash
python script.py
```

with `BATTERYSWAP_DATASET_PATH` pointing at the dataset and
`BATTERYSWAP_SPLITS=train`. The official run uses `public,private`.

## Validation protocol

Folds are grouped by **building**, because the public and private splits contain
different buildings and the observed EOL rate per training building spans 0.043
to 0.833. A random split leaks that structure and flatters every number.

The 48 train scenarios are not 48 independent samples: they start a week apart
and each covers six weeks, so adjacent windows overlap by roughly 85% and the
effective sample size is nearer eight. `tools/validate_v6.py --blocks` reports
non-overlapping block means for that reason, and differences under about 100 on
the 48-scenario mean should be treated as noise.

## Tests

```bash
python -m unittest discover -s tests -v
```

Coverage includes exact equality of the incremental smoothing cache against the
official `smooth_series` (single-pass, incremental, and truncated-prefix),
forecast-contract validity on an early and a late scenario, the censoring
branches, the runtime governor, route optimization, plan completeness, and
equality between the fast operational replay and
`batteryswap_public.evaluate_plan()`.

## Environment overrides

- `BATTERYSWAP_MODEL_PATH` -- Task 1 artifact, default `models/v7_wiener.joblib`
- `BATTERYSWAP_PLANNER_PATH` -- load a pickled planner instead
- `BATTERYSWAP_SOLVER_SECONDS`, `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS`,
  `BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS`, `BATTERYSWAP_ROBUST_SAMPLES`
- `BATTERYSWAP_LATE_RISK_MULTIPLIER`, `BATTERYSWAP_MINIMUM_EXPECTED_IMPROVEMENT`
- `BATTERYSWAP_SOFT_DEADLINE`, `BATTERYSWAP_HARD_DEADLINE` -- governor thresholds

Defaults in `script.py` are the shipped values, so the submission is correct with
no variables set: solver seconds 1.0, local and uncertain search 240/240,
`BATTERYSWAP_ROBUST_SAMPLES` 0 (note the `PlannerConfig` dataclass default is 4).

The probability scale, the volatility scale and the veto thresholds are
deliberately **not** environment-configurable. They are constants in source
(`PROBABILITY_SCALE` in `script.py`, `STALE_DAYS_LIMIT` and
`STALE_MARGIN_LIMIT` in `bsai/forecaster.py`) so that what ships cannot depend on
a variable being set at evaluation time.

## Third-party components

All runtime dependencies are supplied by the competition image and listed in
`requirements.txt`. The ones this solution actually uses are `numpy`, `pandas`,
`scikit-learn`, `scipy`, `joblib`, `fastparquet`, `ortools` and
`batteryswap_public`, under BSD-3-Clause, BSD-3-Clause, BSD-3-Clause,
BSD-3-Clause, BSD-3-Clause, Apache-2.0, Apache-2.0 and the competition's own
terms respectively. No external datasets and no third-party pretrained model
weights are used. Participant-authored code in this repository is MIT licensed;
see `LICENSE`.
