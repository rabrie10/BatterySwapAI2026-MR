# Task 1 Implementation Guide

Status: implemented and locally verified, last updated 2026-08-18.

This document is the engineering specification for the Task 1 risk model in
this repository: `src/risk/`. It describes the causal feature pipeline, the
survival model, calibration, causal out-of-fold validation, the artifact
handoff to Task 2, and the exact commands needed to reproduce
`models/risk_forecaster.pkl`.

Task 2's contract and planner (`batteryswap_solution/`) are unchanged by this
work. Task 1 only produces an object implementing `RiskForecaster.predict()`
per `batteryswap-risk-forecast/v1`, as documented in
[TASK2_IMPLEMENTATION.md](TASK2_IMPLEMENTATION.md) sections 4 and 12.

## 1. Scope and design choice

Task 1 must emit, for every active battery, a calibrated daily failure CDF
over the planning horizon plus the three-state evaluator-aligned tail split
(observed-after-horizon / unobserved-EOL / final in-horizon mass). See
TASK2_IMPLEMENTATION.md Sec 4 for the exact contract this satisfies.

**Model family: one censoring-aware parametric AFT model** (Weibull,
LogNormal, or LogLogistic — selected by causal out-of-fold Brier score), fit
on a small, curated covariate set, with a 1-D Platt recalibration layer on
top.

This is a deliberate choice given the dataset: **only 82 observed physical
EOL events** across 461 devices (82.2% censored). A discrete-time hazard
classifier with a rich feature set (the kind of model that would normally be
the P0/P1 "primary model" per the design spec) needs enough events per
covariate to be trustworthy; with 82 events, a large feature set is not
identifiable, and a tree-based classifier's daily-hazard curve would not
extrapolate sensibly hundreds of days past the 42-day planning horizon (which
the observed-tail / unobserved-EOL split can require — see Sec 4). A single
parametric AFT model:

- extrapolates smoothly and monotonically to arbitrary horizons by
  construction, with no special-casing needed for the tail;
- is well-identified with an order of magnitude fewer covariates than events,
  which 82 events can actually support with a regularization penalizer;
- gives a closed-form, cheaply-integrable survival function, so the
  mean-excess-RUL tail statistic is an exact numeric integral rather than an
  approximation.

The tradeoff is real and documented, not hidden: AFT's linear-in-log-time
covariate effects cannot capture strong nonlinear interactions (e.g. a
temperature-conditioned voltage residual) the way a boosted tree could. If
more labeled events become available, replacing/ensembling with a
discrete-time `HistGradientBoostingClassifier` hazard model over the same
causal features is the natural P2 extension (see Sec 8).

## 2. Causal feature pipeline (`src/risk/features.py`)

### 2.1 Daily panel

Raw hourly voltage/temperature readings are aggregated to one robust
(median voltage, median temperature, reading count) row per
`(device_id, calendar day)`. The panel is sparse — a day with no reading
simply has no row — which lets every rolling feature use pandas'
offset-based rolling (`rolling('{n}D')`) directly on the sparse,
possibly-gappy series without reindexing or NaN-handling: a `'28D'` window
aggregates over however many rows actually fall in the trailing 28 calendar
days, which is exactly the gap-tolerant behavior we want.

### 2.2 Rolling causal features (computed once per device)

For six calendar windows (7/14/28/56/90/180 days): trailing mean, std, min,
completeness (`n_readings / window_days`), fraction of readings within
0.1V of the 2.4V EOL threshold, and the longest single-day gap ending in that
window. For three windows (14/28/90 days): a closed-form trailing OLS slope
of voltage against time, computed via rolling sums
(`Σt, Σy, Σty, Σt²` → `slope = (nΣty - ΣtΣy) / (nΣt² - (Σt)²)`) so no
per-row Python loop is needed. A voltage/threshold-crossing extrapolation
(`distance_to_threshold / max(-slope_90d, min_decline)`, capped at 3650 days)
completes the state/age family from the design spec.

Each device's full rolling-feature series is computed **once** from its own
raw readings — never per (device, cutoff) — specifically to avoid the
re-scan-per-cutoff cost trap flagged in `docs/SOLUTION_DESIGN_SPEC.md` Sec
2.4/11. A `(device, cutoff)` lookup is then an O(log n) `searchsorted` for
"the latest available row with date <= cutoff" (`features.lookup_asof`),
which is exactly the row a rolling window ending on that date would produce —
by construction, a rolling window anchored at date `d` only ever aggregates
rows with date `<= d`, so this lookup can never see data after `cutoff`. This
no-future-leakage property is regression-tested directly in
`tests/test_task1_forecast.py::FeatureCausalityTests` by asserting that
appending future rows to a device's history does not change the feature
values already looked up at an earlier cutoff.

### 2.3 Building context (leave-one-out)

`features.leave_one_out_building_features` adds peer aggregates (median
latest voltage, slope, low-voltage fraction, crossing-day estimate, history
length) computed by a `groupby((cutoff, building)).transform` over the other
devices' own already-causal features at the *same* cutoff, explicitly
excluding the device itself. Because this only ever mixes other devices'
individually-causal features at an identical point in time, it carries
neither future information nor device-identity leakage, and it generalizes to
unseen buildings at inference time by construction (it only assumes "other
active batteries at this cutoff", never a specific building identity).

## 3. Cutoff / label construction (`src/risk/cutoffs.py`)

### 3.1 Examples

An example is `(device_id, cutoff)` for a cutoff at which the device is
"at risk" — its terminal time (observed EOL if any, else its own data window
end) is strictly after `cutoff`. `duration_days = terminal_time - cutoff`,
`event = 1` iff the terminal time is an observed EOL. Standard right-censored
(duration, event) survival notation already implements the spec's masked
multi-horizon label rule (Sec 5.1) for free: a censored duration correctly
contributes no information about horizons beyond its own censoring point to
the AFT likelihood, while an observed event contributes exact information at
every horizon. The exact boundary behavior (event/censoring exactly at a
horizon) is unit-tested in `MaskedLabelBoundaryTests`.

Cutoffs are the union of the 48 official scenario start dates (highest value,
because they match inference exactly) and a synthetic grid (default: every 21
days) spanning the dataset's full observed timeline, to increase event
coverage. A device not yet deployed at a given cutoff is still included if
some official scenario would present it as "active" before its own
`start_time` (a real occurrence in this dataset — the official
`iterate_scenarios` only filters by EOL, not by deployment date); this is
flagged with a `not_yet_deployed` indicator feature rather than silently
excluded, so the model sees this cold-start regime during training instead of
only encountering it as a surprise at inference.

Repeated cutoffs from the same device are downweighted
(`sample_weight = 1 / n_cutoffs_for_device`, contract `weights_col` in
lifelines) so a long-lived device with many cutoffs does not dominate the
likelihood over a short-lived one, per design spec Sec 5.1.

### 3.2 Causal grouped validation

`cutoffs.assign_building_folds` assigns every cutoff from one building to the
same fold (deterministic hash of building id, not sklearn's order-dependent
`GroupKFold`), giving the "unseen buildings" validation axis. All model
selection, calibration fitting, and reported CV metrics use only
out-of-fold predictions from this grouping — no cutoff from a validation
building ever contributes to feature normalization, imputation, model
fitting, or calibration for that fold.

`cutoffs.time_holdout_mask` additionally marks the temporally latest 20% of
cutoffs as a secondary, **diagnostic-only** time-based holdout (never used
for fitting/calibration) — the "unknown time period" axis the task
instructions ask for. It is reported separately in the training report so a
model that is only causally valid across buildings, but has quietly drifted
across the training period, cannot go unnoticed.

## 4. Model (`src/risk/model.py`)

### 4.1 Curated features

Fourteen covariates (listed in `model.CURATED_FEATURES`): latest voltage,
28-/90-day voltage slope, 28-day voltage volatility, log1p(crossing-day
extrapolation), 28-day mean temperature, 28-day low-voltage fraction, 90-day
completeness, age, days since last reading, two building leave-one-out
aggregates, and `not_yet_deployed`/`cold_start` flags. This list is
deliberately short: with 82 events, the classical "~10 events per covariate"
survival-model rule of thumb would cap a stable fit at well under half the
size of the full engineered feature set (60+ columns across six rolling
windows); `distance_to_threshold` was dropped from the naive superset because
it is an exact linear function of `latest_voltage` (pure collinearity, no
extra information). `cold_start` (fewer than 3 total readings) acts as a
single unified missing-data indicator rather than adding one indicator column
per feature, since the causal columns tend to go missing together.
Continuous covariates are z-scored using **train-fold-only** statistics
inside cross-validation, and full-data statistics for the final artifact.

### 4.2 Model selection

Grid search over `{Weibull, LogNormal, LogLogistic} × penalizer ∈ {0.1, 0.5,
1.0}` AFT fitters (`lifelines`), scored by mean out-of-fold Brier score across
{7, 14, 21, 28, 35, 42}-day horizons under the exact masked-label rule (Sec
3.1), with concordance index reported as a secondary check. See
`docs/task1_training_report.json` (generated by `src/risk/train.py`) for the
selected configuration and the full grid.

### 4.3 Calibration

A single pooled Platt (1-D logistic-on-logit) calibrator is fit on the
out-of-fold predictions from the selected configuration, pooled across all
six horizons and weighted by the same per-device sample weights. Platt
scaling was chosen over isotonic regression per the design spec's explicit
guidance ("Platt/beta calibration first, isotonic only when sample support is
adequate") — with 82 physical events, isotonic bins would have very high
variance. The calibrator's slope is asserted positive at fit time (falling
back to the identity map otherwise), which is what guarantees the calibrated
curve stays monotone non-decreasing in time: raw AFT survival is monotone by
construction, and composing a monotone-increasing calibration map preserves
that.

### 4.4 Contract math

Given one calibrated conditional CDF `G` (day offset from `prediction_origin`)
and `C` = `evaluation_observation_end` as a day offset:

```text
failure_cdf(d)               = G(min(d, C))
prob_observed_after_horizon  = max(G(C) - G(horizon_end), 0)
prob_unobserved_eol          = max(1 - G(C), 0)
```

These three identities sum to exactly one for any monotone `G` regardless of
whether `C` falls before, at, or after the horizon end — `Task1Forecaster.predict()`
evaluates `G` at capped time arguments rather than patching the tail after
the fact, so the contract's probability-mass identity holds by construction,
not by post-hoc renormalization. `mean_excess_rul_days_given_observed_after_horizon`
is the closed-form conditional-mean-excess integral
`(∫ S(x)dx over [horizon_end, C] - (C - horizon_end)·S(C)) / (S(horizon_end) - S(C))`,
evaluated by trapezoidal numerical integration of the *calibrated* survival
function on a grid capped at 220 points. All of this (including the boundary
case where `C <= horizon_end`, which collapses `prob_observed_after_horizon`
and the mean-excess term to exactly zero through the same formula rather than
a separate branch) is regression-tested in `ContractMathTests` against a
hand-computed exponential-survival closed form.

Cold-start batteries (feature series empty or fewer than 3 readings) are
never special-cased with a separate fallback branch: they simply produce
NaN-heavy raw features that get median-imputed, with `cold_start=True` and a
low `data_quality` score surfaced in `summaries` — the model itself is
trained on plenty of real cold-start examples (see Sec 3.1), so it already
knows what a conservative population-level prediction looks like for this
regime.

## 5. Validation results

See `docs/task1_training_report.json` for the exact numbers from the most
recent training run (selected family/penalizer, causal grouped OOF
concordance and Brier/log-loss by horizon, the full grid search, calibrator
parameters, and the diagnostic-only time-holdout metrics).

## 6. Runtime and packaging

- Training (`src/risk/train.py`) is offline and not subject to the
  competition's 30-minute evaluation limit; it uses only
  `pandas`/`numpy`/`lifelines`/`scikit-learn`, all in the official package
  list (`docs/OFFICIAL_CHALLENGE_REFERENCE.md` Sec "Available Packages").
- Inference (`Task1Forecaster.predict()`) does one batched
  `predict_survival_function` call per scenario across all active batteries
  and all needed day offsets (curve horizon + tail-integration grid, capped
  at 220 extra points) — not a per-battery loop — so it stays well inside the
  per-scenario budget the Task 2 planner already operates under.
- The artifact is a single `pickle`-serialized `Task1Forecaster` dataclass
  (`model_family`, `penalizer`, fitted `lifelines` AFT model, feature
  transform statistics, Platt calibrator parameters) with no external file
  dependencies, matching the pattern already used for
  `models/cox_baseline.pkl` in this repository.

## 7. Reproducibility

```powershell
python -m src.risk.train --dataset-path data/raw/train --out-path models/risk_forecaster.pkl --report-path docs/task1_training_report.json --synthetic-step-days 21 --n-folds 5 --seed 20260818
```

- Deterministic building-fold assignment (seeded hash), deterministic
  synthetic-cutoff grid derived only from the dataset's own timestamps, fixed
  `sample_weight` scheme, fixed AFT `penalizer` grid. Re-running with the same
  dataset and arguments reproduces the same artifact.
- `docs/task1_training_report.json` records the exact configuration (dataset
  path, step size, fold count, seed, curated feature list, model version
  string) alongside the metrics, so a report and an artifact can always be
  matched to each other.

## 8. Task 1 handoff checklist (per TASK2_IMPLEMENTATION.md Sec 12)

- Serialized forecaster: `models/risk_forecaster.pkl`; immutable
  `model_version = "task1-aft-survival/v1"` (bump on any retrain with a
  materially different configuration).
- Training data/feature version: this document, Sec 2-4, plus
  `docs/task1_training_report.json`'s `config` block.
- Causal out-of-fold forecasts: reproducible via the training command above
  (Sec 7); OOF predictions are not persisted separately, only their
  aggregated metrics (persisting full per-row OOF predictions was judged
  unnecessary extra artifact surface for a single-owner project).
- Per-horizon calibration/ranking metrics: `docs/task1_training_report.json`
  (`cv_brier_by_horizon`, `cv_log_loss_by_horizon`, `cv_concordance`).
- Every active battery receives a complete daily CDF: enforced by
  `batteryswap_solution.forecast.validate_forecast`, exercised end-to-end in
  `EndToEndSyntheticFitTests`.
- Observed-tail / unobserved-EOL mass: Sec 4.4 above.
- Runtime/memory: Sec 6 above; see also the fallback and real-mode benchmark
  commands below.
- Required packages: `pandas`, `numpy`, `scikit-learn`, `lifelines`, `scipy`
  — all already in `requirements.txt` / the official package list.
- Reproducibility seed and command: Sec 7 above (`seed=20260818`, matching
  the Task 2 planner's own default seed).

## 9. Commands

```powershell
python -m unittest discover -s tests -v
python tools/benchmark_task2.py --mode fallback --limit 3
python tools/benchmark_task2.py --mode real --limit 12
```

`--mode real` loads `models/risk_forecaster.pkl` into `CompetitionPlanner`
and scores it with the official evaluator on train scenarios — the actual
integration check that this artifact both satisfies the v1 contract under
real scenario data and improves total cost over the deterministic fallback,
not just that the submission path survives without it.
