# Task 1 Implementation Guide

Status: implemented and locally verified, last updated 2026-08-19.

This document is the engineering specification for the Task 1 risk model in
this repository: `src/risk/`. It describes the causal feature pipeline, the
survival model, calibration, causal out-of-fold validation, the artifact
handoff to Task 2, and the exact commands needed to reproduce
`models/risk_forecaster.pkl`.

**Companion document:** [TASK1_MODEL_INVESTIGATION.md](TASK1_MODEL_INVESTIGATION.md)
records how the current design was arrived at — four calibration defects that
each passed the model's own statistical metrics and were only exposed by
end-to-end evaluation against the official cost function. Read that one for
the *why*; this one is the *what*.

Task 2's contract and planner (`batteryswap_solution/`) are unchanged by this
work. Task 1 only produces an object implementing `RiskForecaster.predict()`
per `batteryswap-risk-forecast/v1`, as documented in
[TASK2_IMPLEMENTATION.md](TASK2_IMPLEMENTATION.md) sections 4 and 12.

## 1. Scope and design choice

Task 1 must emit, for every active battery, a calibrated daily failure CDF
over the planning horizon plus the three-state evaluator-aligned tail split
(observed-after-horizon / unobserved-EOL / final in-horizon mass). See
TASK2_IMPLEMENTATION.md Sec 4 for the exact contract this satisfies.

**Pipeline in one line:**

```text
causal features -> AFT survival curve -> physical-prior blend
    -> horizon-conditional isotonic calibration -> v1 contract tables
```

**Model family: one censoring-aware parametric AFT model** (Weibull,
LogNormal, or LogLogistic — selected by causal out-of-fold Brier score), fit
on a small, curated covariate set, blended with a sharp deterministic
physical-extrapolation prior, then recalibrated by a **horizon-conditional
isotonic** map. The blend and the calibration are both load-bearing and their
*ordering* matters — see Sec 4.3-4.5 and the investigation log.

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

Grid search over AFT families × penalizers (`lifelines`), scored by mean
out-of-fold Brier across {7, 14, 21, 28, 35, 42}-day horizons under the exact
masked-label rule (Sec 3.1), with concordance as a secondary check.
`train.py` defaults to `families=weibull,lognormal` and
`penalizers=0.1,0.5`; LogLogistic is excluded by default because it was never
selected and scored within noise of LogNormal, so dropping it roughly halves
grid-search time. LogNormal/0.1 has been selected in every run to date. See
`docs/task1_training_report.json` for the full grid.

Note that this selection criterion (OOF Brier) is *weakly* discriminating
here — all nine configurations of the original 3×3 grid landed in a narrow
band (concordance 0.889-0.902, mean Brier 0.0097-0.0106). The family and
penalizer choice is not where this model's performance is decided;
calibration (Sec 4.3) is.

### 4.3 Calibration

Calibration is **selected, not assumed**. Four candidates are fit on the same
out-of-fold pool and the best OOF weighted log loss wins:

| Candidate | OOF weighted log loss |
|---|---:|
| identity (raw, uncalibrated) | 0.2039 |
| Platt (1-D logistic on logit) | 0.0942 |
| pooled isotonic | 0.0851 |
| **horizon-conditional isotonic** ← selected | **0.0783** |

The design spec's guidance was "Platt/beta calibration first, isotonic only
when sample support is adequate". That guidance turned out to be wrong for
this problem, for a reason worth recording: Platt is a two-parameter
shift/scale of the logit, and this model is *saturated* rather than merely
shifted — it pushes ~60 batteries/scenario to a high probability against ~9
real events. No (slope, intercept) pair can pull a saturated cluster down;
sweeping the slope from 0.711 to 1.5 moved the mean prediction by under 10%.
Isotonic maps each predicted level directly onto its observed frequency and
can. Support is in fact adequate: the pool holds 522,532 rows.

**Horizon-conditional** matters because the event rate is a strong function
of horizon — 0.32% at 7 days rising to 16.45% at 365 days, a ~50x spread. One
pooled map is calibrated to the pooled average (4.20%) and therefore
over-predicts by ~2x at the 42-day planning horizon (true rate 1.92%).
`HorizonIsotonicCalibrator` fits one isotonic map per horizon in
`CALIBRATION_HORIZONS = (7, 14, 21, 28, 35, 42, 60, 90, 120, 180, 240, 300,
365)` and interpolates linearly between them. The long horizons are not
decoration: `predict()` must evaluate out to `evaluation_observation_end` (up
to ~334 days past origin) to split observed-tail from unobserved-EOL mass, and
calibrating only to ≤42 days made that pure extrapolation.

**Monotonicity** is preserved throughout: each isotonic map is monotone
non-decreasing in the raw value, the raw AFT survival curve is monotone in
time, and `predict()` applies a running maximum over time afterwards — so the
emitted CDF satisfies the v1 contract even where two neighbouring horizon maps
disagree slightly.

**Tie-breaking.** Isotonic regression is a step function, so it maps whole
groups of batteries onto one identical probability — which silently destroys
the model's within-group ranking (AUC 0.93). That is not a cosmetic issue:
faced with 40 tied batteries just above its decision threshold, the planner
has no basis to prefer the riskiest few and swaps the entire block. The
calibrator therefore mixes a small weight (`tie_break_weight = 0.02`) of the
raw score back in. Both terms are monotone in the raw score, so this preserves
monotonicity and holds the calibrated level to within 0.01 while restoring a
strict ordering inside each plateau.

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

### 4.5 Physical-prior blend (`model.physical_blend`)

The AFT curve is not used alone. At every evaluated time point the raw
survival CDF is combined with a physical crossing-day CDF by pointwise
maximum, so whichever estimate considers failure more imminent wins:

```text
blended = max(raw_AFT_CDF, physical_CDF)      # then calibrate, then cummax
```

`_conditional_logistic_cdf` reuses the location-scale logistic form of
`forecast.VoltageTrendForecaster`, located at each battery's own
`crossing_days_extrapolated` (Sec 2.2) with a fixed 20-day scale. The blend
exists because with only 82 physical events the AFT's covariate coefficients
are heavily shrunk toward the population baseline, which under-predicts
near-term risk for batteries whose voltage has plateaued just above the
threshold rather than declining smoothly.

**Ordering is critical, and was the subject of a real bug.** The blend must be
applied to the *raw* curve **before** calibration, in both training
(`run_cross_validation`) and inference (`Task1Forecaster.predict`). An earlier
version blended *after* calibrating, which pushed values straight off the
calibrated scale the calibrator had just been fit to produce — silently
invalidating it. Because the blend now happens pre-calibration, the calibrator
sees and corrects the same blended distribution it will later be applied to.

**Cold start.** A battery with no crossing-day estimate gets a placeholder
distance far outside any horizon, so the physical term contributes ~0 and the
survival model alone determines its forecast — the right behaviour when there
is no voltage history to extrapolate from.

**Already past threshold.** `crossing_days_extrapolated` is bounded below at
−90 days rather than 0. Clipping at 0 made every already-below-threshold
battery numerically identical, erasing the distinction between "just dipped
below" and "far below and still falling" — another source of planner-visible
ties. This feature feeds only the blend (`crossing_days_log` is computed but
is not in `CURATED_FEATURES`), so the bound does not disturb the AFT fit.

`PhysicalPriorBlendTests` in `tests/test_task1_forecast.py` locks in both
directions: a battery in clear physical decline gets lifted to high risk even
when a stub AFT model predicts exactly zero, and a battery with flat voltage
(and therefore a huge crossing-day estimate) stays near zero — the blend does
not spuriously inflate risk for batteries with no real signal.

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

**Selected model**: LogNormal AFT, penalizer=0.1 (the same family every grid
search selected). Causal building-grouped out-of-fold concordance: **0.906**.
Brier/log loss by horizon, and — critically — the ratio against the trivial
"predict the base rate" baseline:

| Horizon (days) | Brier | Log loss | Base rate | Brier / trivial |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.00309 | 0.01353 | 0.32% | 0.97 |
| 14 | 0.00645 | 0.02518 | 0.64% | 1.01 |
| 21 | 0.01052 | 0.03737 | 0.97% | 1.10 |
| 28 | 0.01501 | 0.04954 | 1.30% | 1.17 |
| 35 | 0.01998 | 0.06281 | 1.61% | 1.26 |
| 42 | 0.02517 | 0.07704 | 1.92% | 1.34 |

**Read that last column before trusting the Brier scores.** With a ~2% base
rate, predicting a constant zero scores ~0.019 at 42 days. The raw Brier
numbers look excellent and are largely an artefact of class imbalance. This
was the single most misleading metric in the whole project — an earlier model
posted a *better-looking* 42-day Brier of 0.0167 while being roughly twice as
expensive as doing nothing under the official cost function. Always report the
trivial-baseline ratio alongside.

**Calibrator**: horizon-conditional isotonic, selected over identity, Platt
and pooled isotonic by out-of-fold weighted log loss (0.0783 vs 0.2039 /
0.0942 / 0.0851). Full grid in `docs/task1_training_report.json`.

**Calibration quality against ground truth** (8 sampled scenarios, comparing
the shipped v4 against the earlier Platt-calibrated model):

| Metric | Platt model | **v4 (shipped)** | Truth |
| --- | ---: | ---: | ---: |
| mean predicted in-horizon risk | 0.0945 | **0.0346** | 0.0234 |
| predicted / true ratio | 4.05x | **1.48x** | 1.00x |
| Brier vs trivial baseline | 1.97 (worse) | **0.93 (better)** | <1 |
| mean `prob_unobserved_eol` | 0.658 | **0.830** | 0.908 |
| batteries predicted >10% risk | 81.4 | **35.3** | ~9.9 real events |

**Ranking quality** (unchanged by calibration, and the reason the fixes
worked): AUC **0.932**; top-20 by predicted risk captures **5.0 of ~9.9** real
in-horizon failures, against 0.43 for random ordering.

### 5.2 Why out-of-fold statistical metrics were not enough

The single most important methodological finding from this project. Every
model version below passed the v1 contract, held OOF concordance at
0.90-0.91, and posted Brier scores that looked excellent in isolation — while
differing by a factor of **two** in the only metric that counts.

**[TASK1_MODEL_INVESTIGATION.md](TASK1_MODEL_INVESTIGATION.md) is the full
record**, with evidence for each of the four calibration defects found. In
brief:

1. Platt scaling cannot calibrate a *saturated* model (it is a two-parameter
   logit shift/scale; sweeping the slope 0.711→1.5 moved the mean under 10%).
2. The physical blend ran *after* calibration, invalidating it.
3. One pooled calibration map spanned a ~50x horizon-dependent event-rate
   spread, and did not cover the long horizons the tail split needs.
4. Isotonic plateaus collapsed the AUC-0.93 ranking into ties, so the planner
   swapped entire tied blocks. (Attempted fix rejected — see Sec 5.4.)

### 5.3 End-to-end results against the official evaluator

All 48 train scenarios, official evaluator, via
`tools/benchmark_task2.py --mode real --limit 0`. Every row is recorded in
`docs/local_benchmark_log.csv`.

| Model | `total_cost` | swaps | `early_swap` | `late_swap` |
| --- | ---: | ---: | ---: | ---: |
| v1/v2 — Platt, blend-after-calibration | 6023.61 | 71.1 | 4562.57 | 278.13 |
| v3 — pooled isotonic, blend-before | 4005.33 | 43.1 | 2597.77 | 712.92 |
| **v4 — horizon-conditional isotonic (shipped)** | **3128.88** | 28.0 | 1589.72 | 1072.50 |
| v5 — v4 + tie-break + negative crossing-days | 3168.27 | 28.3 | 1621.54 | 1050.00 |
| *baseline* — all-defer | 3324.68 | 0 | 0 | — |
| *ceiling* — oracle EOL | ~78 | ~13.5 | ~9.6 | 0 |

v4 is the first version to beat the all-defer baseline, and roughly halves
the cost of the model that produced the 2026-08-18 leaderboard submission
(7389.39, rank 11/14).

### 5.4 Rejected experiment (v5)

v5 targeted Defect 4 with two changes: a calibrator tie-break, and bounding
`crossing_days_extrapolated` at −90 rather than 0 so already-below-threshold
batteries stay ordered. Both worked mechanically (top-50 distinct values in
`s_24` went from ~5 to 29) but measured **1.3% worse** end-to-end. Both were
reverted to inert defaults (`tie_break_weight = 0.0`,
`MIN_CROSSING_DAYS = 0.0`) so the shipped code reproduces v4's benchmark
exactly; the mechanisms remain as documented opt-in knobs.

The lesson: the plateau is a *symptom* of the model having exhausted its
discriminating signal at the top of the ranking, not an independent defect.
With 82 events the top ~30 batteries genuinely are near-indistinguishable,
and imposing an ordering on them adds no information. Further gains need a
better-discriminating model, not better post-processing.

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

Ordered by how much they should worry you.

- **The end-to-end benchmark is in-sample — this is the biggest gap.** The
  shipped artifact is fit on all 461 train devices and then benchmarked on
  train scenarios. Model and calibrator *selection* is honest (grouped
  building folds, no fold sees its own validation buildings), but the
  `total_cost` figures in Sec 5.3 are **not** out-of-fold.
  `docs/SOLUTION_DESIGN_SPEC.md` Sec 8.1 explicitly requires generating OOF
  forecasts per scenario and evaluating Task 2 only against those; that
  harness was not built. Expect real generalisation to be worse than 3128.88.
  Building it (train K fold models, route each battery to the model that did
  not see its building, re-run the benchmark) is the highest-value next step.
- **Residual over-prediction remains**: predicted/true in-horizon ratio is
  ~1.48, and the model swaps ~28 per scenario against an oracle ~13.5. The
  remaining gap is model discrimination, not calibration (Sec 5.4).
- **Raw Brier/concordance are misleading here** and must never be quoted
  without the trivial-baseline ratio (Sec 5.1). A model with a *better*
  42-day Brier was twice as expensive under the official cost function.
- **`physical_uncertainty_days=20.0` is a fixed constant**, not fit from
  data. It matches `VoltageTrendForecaster`'s scale (18) and was never
  independently tuned. Worth a sensitivity check.
- **The AFT component's practical contribution is smaller than originally
  intended** — for batteries with a clear physical decline signal the blend is
  dominated by the deterministic term. The AFT still does real work (the
  calibrated level, the tail/censoring split, and the sole signal for
  cold-start batteries), but this is a different balance than Sec 1 intended
  and is worth revisiting if more physical events become available.
- **Only 82 unique observed EOL events** (82.2% censoring) is the binding
  constraint on everything above. Every AFT covariate has p > 0.25 even when
  fit alone, under a robust sandwich estimator that correctly accounts for
  ~461 independent devices behind 48,059 person-period rows.
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
  `model_version = "task1-aft-horizon-isotonic/v4"` (bump on any retrain with
  a materially different configuration). Reflects the 6-covariate feature set,
  blend-before-calibration, and horizon-conditional isotonic calibration.
  Earlier artifacts are kept alongside for comparison:
  `risk_forecaster_v2_backup.pkl` (the Platt model that produced the 7389.39
  leaderboard entry), `_v3`, `_v4`, `_v5`.
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
