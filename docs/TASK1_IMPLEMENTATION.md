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
on a small, curated covariate set, with a 1-D Platt recalibration layer, and
a sharp deterministic physical-extrapolation prior blended in at prediction
time. That last piece is not a minor detail — Sec 5 explains why it turned
out to be necessary, not optional.

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
temperature-conditioned voltage residual) the way a boosted tree could, and —
as Sec 5 found empirically, not just theoretically — its covariate
coefficients are so heavily shrunk by only 82 events that the AFT curve alone
under-predicts near-term risk for a specific, common physical pattern
(voltage plateaued just above the EOL threshold rather than declining
smoothly). If more labeled events become available, replacing/ensembling with
a discrete-time `HistGradientBoostingClassifier` hazard model over the same
causal features is the natural P2 extension (see Sec 9).

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

Six covariates (listed in `model.CURATED_FEATURES`): latest voltage, 28-day
voltage slope, 28-day low-voltage fraction, age, and `not_yet_deployed`/
`cold_start` flags. `features.py` computes a much larger engineered set
(60+ columns across six rolling windows, plus building leave-one-out
aggregates — Sec 2.2/2.3); the AFT model deliberately uses only a small
subset of it.

This started as a 14-covariate set (adding 90-day slope, volatility,
log1p(crossing-day extrapolation), temperature, completeness, days-since-
last-reading, and two building leave-one-out aggregates) and was cut down
after the retrain #2 investigation in Sec 5 found that **every one of those
14 covariates had p > 0.25 in the fitted model, including `latest_voltage`
alone with no other covariates present.** This is an expected consequence,
not a bug: `lifelines`' `robust=True` sandwich variance estimator correctly
accounts for the fact that the 48,059 training rows are really only ~461
independent devices with 82 independent events (each device contributes many
person-period rows at different cutoffs), so no covariate can reach classical
significance regardless of how many are included. But it does mean including
weak covariates purely dilutes optimizer attention without buying identification, so the set was cut to the smallest group with a
direct physical justification for individual battery risk (voltage level,
its recent trend, and how much of its recent history was already close to
the threshold), plus age and data-availability flags. `distance_to_threshold`
was excluded throughout because it is an exact linear function of
`latest_voltage` (pure collinearity, no extra information); `crossing_days_log`
and the building leave-one-out features were dropped in this cut — see Sec 5
for why `crossing_days_extrapolated` came back afterward, in a different
role. `cold_start` (fewer than 3 total readings) acts as a single unified
missing-data indicator rather than one indicator column per feature, since
the causal columns tend to go missing together. Continuous covariates are
z-scored using **train-fold-only** statistics inside cross-validation, and
full-data statistics for the final artifact.

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

### 4.5 Physical-prior blend (`Task1Forecaster.predict`)

The final curve is `max(calibrated AFT CDF, physical crossing-day CDF)` at
every evaluated time point, not the AFT curve alone. `_conditional_logistic_cdf`
reuses the same location-scale logistic form as `forecast.VoltageTrendForecaster`,
located at each battery's own `crossing_days_extrapolated` (the same causal
feature described in Sec 2.2) with a fixed 20-day uncertainty scale, and the
pointwise maximum is taken before the final `cummax` monotonicity pass — so
whichever of the two estimates thinks failure is more imminent wins at each
point in time, and the blended curve is still guaranteed monotone and
contract-valid. Cold-start batteries fall back to a placeholder crossing-day
distance far outside the horizon, so the physical term contributes ~0 there
and the AFT+calibration curve alone determines their forecast.

This exists because it was *measured*, not assumed, to matter — see Sec 5.
`PhysicalPriorBlendTests` in `tests/test_task1_forecast.py` locks in both
directions: a battery in clear physical decline gets lifted to high risk even
when a stub AFT model predicts exactly zero, and a battery with flat voltage
(and therefore a huge, capped crossing-day estimate) stays near zero — the
blend does not spuriously inflate risk for batteries with no real signal.

## 5. Validation results

### 5.1 Statistical (out-of-fold) metrics

Full numbers are in `docs/task1_training_report.json`; this reflects the
2026-08-18 retrain (`data/raw/train`, `synthetic_step_days=21`, `n_folds=4`,
6-covariate feature set, `families=(weibull, lognormal)`,
`penalizers=(0.1, 0.5)`, `seed=20260818`): **48,059** `(device, cutoff)`
examples from all 461 devices, **6,236** person-period rows with an in-window
observed event (pooled from the dataset's 82 unique physical EOL events
across many cutoffs). Elapsed training time (grid search + final fit): ~2,850s
(~47min) — an offline, one-time cost, not part of the competition's
per-scenario runtime budget. (An earlier 14-covariate, 3-family, 3-penalizer
grid search took ~7,685s and is what first surfaced usable OOF metrics before
the feature-set investigation below; it is superseded by this retrain, not
separately reported.)

**Selected model**: LogNormal AFT, penalizer=0.1 (same family the earlier
14-covariate search also selected). Causal building-grouped out-of-fold
concordance: **0.906**. Brier/log loss by horizon:

| Horizon (days) | Brier | Log loss |
| ---: | ---: | ---: |
| 7 | 0.00320 | 0.02010 |
| 14 | 0.00630 | 0.02893 |
| 21 | 0.00932 | 0.03718 |
| 28 | 0.01208 | 0.04489 |
| 35 | 0.01444 | 0.05126 |
| 42 | 0.01673 | 0.05799 |

(Full grid search table is in `docs/task1_training_report.json`; concordance
and Brier are essentially unchanged from the 14-covariate version — see
Sec 5.2 for why that alone was misleading.) Calibrator: Platt slope=0.711,
intercept=0.029.

### 5.2 Why out-of-fold statistical metrics were not enough

This is the most important methodological finding from building Task 1, so
it is documented in full rather than summarized away.

The 14-covariate model above passed every contract test and had a
respectable OOF concordance (0.900). But loading it into the *actual*
`CompetitionPlanner` and scoring it with the official evaluator
(`tools/benchmark_task2.py --mode real`) on real train scenarios told a
different story: mean `total_cost` on scenarios `s_0..s_2` was **~3,115**,
worse than the existing deterministic `VoltageTrendForecaster` fallback's
**~2,338** on the same three scenarios, and one scenario (`s_4`) scheduled
*zero* in-window swaps at all (a legitimate optimizer decision given that
forecast, not a crash).

Diagnosis (reproduced with a small ad hoc script, not committed — the
methodology is what matters): for the 9 batteries in `s_0` with a real
observed EOL inside the 42-day horizon (in some cases only 15-19 days out),
the fitted AFT model predicted only 3-31% cumulative failure probability by
day 42, and a **90-130 day** mean-excess RUL for the "survives past the
horizon" branch. Inspecting the fitted coefficients directly explained why:
every covariate, including `latest_voltage` fit *alone* with nothing else in
the model, had p > 0.25. The point estimates were directionally sensible and
reasonably stable across covariate subsets (so not literally noise), but with
only 82 independent events, `lifelines`' `robust=True` sandwich variance
estimator correctly reports that no individual covariate's effect is sharply
identified — so the AFT's location parameter regresses heavily toward the
population baseline for every battery. Since the population baseline
survival time is dominated by the 82.2% of devices that essentially never
fail, this shrinkage systematically under-predicts near-term risk exactly for
the subpopulation the planner most needs to distinguish: batteries whose
voltage has plateaued just above 2.4V (a near-zero trailing slope in every
rolling window, since a battery near end-of-life often oscillates near
threshold for a while before crossing) rather than declining smoothly.
Reducing the feature set from 14 to 6 covariates (Sec 4.1) did not fix this —
a retrain with the smaller set actually scored *worse* in the same
integration benchmark (mean `total_cost` ≈ 3,834) — confirming the problem
was never "too many features," it was that **AFT's covariate-driven
shrinkage cannot supply the sharp, individually-differentiated risk signal
this planning problem needs from only 82 events, regardless of which or how
many covariates it is given.**

The fix (Sec 4.5) does not require retraining: blend in a sharp, low-variance
*deterministic* estimate — the same physical crossing-day extrapolation
already computed as a causal feature — as a floor under the AFT+calibration
curve. This directly targets the diagnosed failure mode (the AFT under-reacts
to voltage proximity to the threshold) while keeping the AFT model
responsible for everything it *is* well-identified for from 82 events pooled
together: overall calibration, the observed/unobserved-EOL tail split, and
monotone extrapolation for genuinely cold-start batteries where no physical
extrapolation is available.

### 5.3 End-to-end integration results (after the blend)

Re-running the identical integration benchmark with the blend in place, mean
`total_cost` against the official evaluator:

| Scenarios | All-defer | `VoltageTrendForecaster` fallback | Task 1 (blended) |
| --- | ---: | ---: | ---: |
| `s_0..s_2` (3 scenarios) | 4,341.5 | 2,337.7 | **1,998.9** |
| `s_0..s_11` (12 scenarios) | 4,885.5 | 6,622.9 | **4,451.1** |

The blended model now beats the deterministic fallback on **all three**
directly-compared scenarios individually, not just on average, and beats both
baselines on the broader 12-scenario sample (the fallback is in fact worse
than all-defer over 12 scenarios — dominated by one very early-swap-heavy
scenario — which is itself evidence that a 3-scenario spot check is not a
reliable comparison and the 12-scenario number is the one to trust). This
satisfies the design spec's own primary acceptance bar for Task 1 (Sec 5.5:
"most importantly, improves held-out total planner cost"), on the largest
sample this session's time budget allowed; see Sec 6 for what a
pre-submission check should still add.

**Time holdout (diagnostic only, not used for fitting/calibration)** — the
temporally latest 20% of cutoffs: Brier degrades moderately at longer
horizons relative to the causal building-grouped OOF numbers (see
`docs/task1_training_report.json`'s `time_holdout` block for the exact
figures from whichever retrain most recently populated it). This is an
expected, moderate gap, not a red flag — but per Sec 10 of
`docs/SOLUTION_DESIGN_SPEC.md` (competition tuning protocol), it is the
reason the physical-uncertainty-days constant (20, fixed, not tuned on
leaderboard feedback) and the general tradeoff noted in Sec 1 should be kept
in mind before trusting long-tail extrapolation on a materially different
private split.

## 6. Known limitations and suggested next check

- **12 scenarios is not 48.** Sec 5.3's comparison used 12 of the 48 train
  scenarios (chosen for wall-clock budget within this session, each real-mode
  scenario costs ~25-35s end-to-end through the full planner). Before an
  official submission, re-run
  `python tools/benchmark_task2.py --mode real --limit 0` (all scenarios) and
  confirm the improvement holds; also compare worst-case/tail scenario cost,
  not only the mean, per the design spec's competition tuning protocol.
- **`physical_uncertainty_days=20.0` is a fixed constant**, not fit from
  data. It matches `VoltageTrendForecaster`'s own default scale (18), which
  the existing benchmark evidence already shows is reasonable, but it was not
  independently re-tuned here. Sensitivity to this constant is worth checking
  as a P1 follow-up.
- **The AFT component's practical contribution is now smaller than
  originally intended** — for batteries with a clear physical decline signal,
  the blend is dominated by the deterministic term, not the fitted model.
  The AFT model still does real work (calibration shape, the tail/censoring
  split, and the sole signal for genuinely cold-start batteries), but this is
  a different balance than Sec 1's original design intent, and is worth
  revisiting if a future data release brings more physical events.
- **Full-run wall-clock time is still a live risk for the actual
  public+private submission**, not just a train-split curiosity. A precisely
  timed full run of `script.py` on all 48 train scenarios took **19.91
  minutes** (`Measure-Command`, 2026-08-19), after the Sec 7 feature-path fix
  (down from ~24.5 minutes before it). That is already close to the "target
  at most 20 minutes" safety margin `docs/SOLUTION_DESIGN_SPEC.md` Sec 11 sets
  for the *whole* run under the competition's 30-minute hard limit — and the
  real submission evaluates train-sized public *and* private splits together,
  not train alone. If public+private combined have a scenario count anywhere
  near 2x train's 48, total runtime would land around ~40 minutes and likely
  fail outright. Per-scenario cost is now dominated by Task 2's CP-SAT +
  local search (`batteryswap_solution/planner.py`), not Task 1's forecast
  call (see Sec 7), so closing this gap further means tuning
  `PlannerConfig`/`OptimizationConfig` (`BATTERYSWAP_SOLVER_SECONDS`,
  `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS`, etc.) — outside Task 1's scope and
  not done here pending explicit sign-off, since it trades plan quality for
  speed.

## 7. Runtime and packaging

- Training (`src/risk/train.py`) is offline and not subject to the
  competition's 30-minute evaluation limit; it uses only
  `pandas`/`numpy`/`lifelines`/`scikit-learn`, all in the official package
  list (`docs/OFFICIAL_CHALLENGE_REFERENCE.md` Sec "Available Packages").
- Inference (`Task1Forecaster.predict()`) does one batched
  `predict_survival_function` call per scenario across all active batteries
  and all needed day offsets (curve horizon + tail-integration grid, capped
  at 220 extra points) — not a per-battery loop — so it stays well inside the
  per-scenario budget the Task 2 planner already operates under. Feature
  extraction for that call uses `features.compute_features_asof`, which
  computes exactly one row per device directly (O(largest rolling window)),
  not `features.build_feature_series`'s full per-date rolling history
  (O(entire device history), correct and necessary for training's
  many-cutoffs-per-device reuse, but wasted work at inference where only one
  date is ever needed). This was a measured, real bug, not a hypothetical
  one: before the fix, `predict()` alone took **~40-45s for a single
  scenario** (profiled directly, not estimated) — comparable to or larger
  than the entire rest of that scenario's planning time combined. See Sec 6
  for the resulting full-run timing and the risk that remains.
- The artifact is a single `pickle`-serialized `Task1Forecaster` dataclass
  (`model_family`, `penalizer`, fitted `lifelines` AFT model, feature
  transform statistics, Platt calibrator parameters) with no external file
  dependencies, matching the pattern already used for
  `models/cox_baseline.pkl` in this repository.

## 8. Reproducibility

```powershell
python -m src.risk.train --dataset-path data/raw/train --out-path models/risk_forecaster.pkl --report-path docs/task1_training_report.json --synthetic-step-days 21 --n-folds 4 --seed 20260818 --families weibull,lognormal --penalizers 0.1,0.5 --physical-uncertainty-days 20.0
```

These are `train.py`'s defaults, so a bare invocation with just
`--dataset-path` reproduces the currently-committed artifact.

- Deterministic building-fold assignment (seeded hash), deterministic
  synthetic-cutoff grid derived only from the dataset's own timestamps, fixed
  `sample_weight` scheme, fixed AFT family/penalizer grid, fixed physical
  blend scale. Re-running with the same dataset and arguments reproduces the
  same artifact.
- `docs/task1_training_report.json` records the exact configuration (dataset
  path, step size, fold count, seed, family/penalizer grid, physical
  uncertainty scale, curated feature list, model version string) alongside
  the metrics, so a report and an artifact can always be matched to each
  other. Note: the currently-committed report's `config` block predates the
  `--families`/`--penalizers`/`--physical-uncertainty-days` CLI flags (it was
  produced by an equivalent ad hoc script during the Sec 5.2 investigation,
  before those flags were added to `train.py` for reproducibility); its
  metrics are still the exact metrics of the committed artifact.

## 9. Task 1 handoff checklist (per TASK2_IMPLEMENTATION.md Sec 12)

- Serialized forecaster: `models/risk_forecaster.pkl`; immutable
  `model_version = "task1-aft-survival-blended/v1"` (bump on any retrain with
  a materially different configuration — this version already reflects the
  6-covariate feature set plus the physical-prior blend from Sec 4.5/5.2, not
  the original 14-covariate design).
- Training data/feature version: this document, Sec 2-4, plus
  `docs/task1_training_report.json`'s `config` block.
- Causal out-of-fold forecasts: reproducible via the training command above
  (Sec 8); OOF predictions are not persisted separately, only their
  aggregated metrics (persisting full per-row OOF predictions was judged
  unnecessary extra artifact surface for a single-owner project).
- Per-horizon calibration/ranking metrics: `docs/task1_training_report.json`
  (`cv_brier_by_horizon`, `cv_log_loss_by_horizon`, `cv_concordance`); see
  Sec 5.2 for why these alone were not sufficient evidence and Sec 5.3 for the
  end-to-end planner-cost metrics that were.
- Every active battery receives a complete daily CDF: enforced by
  `batteryswap_solution.forecast.validate_forecast`, exercised end-to-end in
  `EndToEndSyntheticFitTests`.
- Observed-tail / unobserved-EOL mass: Sec 4.4 above.
- Runtime/memory: Sec 7 above; see also the fallback and real-mode benchmark
  commands below.
- Required packages: `pandas`, `numpy`, `scikit-learn`, `lifelines`, `scipy`
  — all already in `requirements.txt` / the official package list.
- Reproducibility seed and command: Sec 8 above (`seed=20260818`, matching
  the Task 2 planner's own default seed).

## 10. Commands

```powershell
python -m unittest discover -s tests -v
python tools/benchmark_task2.py --dataset-path data/raw/train --mode fallback --limit 3
python tools/benchmark_task2.py --dataset-path data/raw/train --mode real --limit 12
```

`--mode real` loads `models/risk_forecaster.pkl` into `CompetitionPlanner`
and scores it with the official evaluator on train scenarios — the actual
integration check that this artifact both satisfies the v1 contract under
real scenario data and improves total cost over the deterministic fallback
(Sec 5.3), not just that the submission path survives without it (which
`--mode fallback` alone would not have caught — see Sec 5.2).
