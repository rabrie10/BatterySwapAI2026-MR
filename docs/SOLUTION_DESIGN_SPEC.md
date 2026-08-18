# BatterySwapAI 2026 Solution Design Specification

**Status:** v0.9 implementation draft; freeze after Phase P0 clock/schema tests  
**Owners:** Task 1 (RUL/risk model) and Task 2 (work-order planner)  
**Primary objective:** Minimize the official evaluator's mean `total_cost` across unseen scenarios under the CPU, memory, and runtime constraints.  
**Secondary objective:** Preserve enough uncertainty and diagnostics to improve the complete prediction-to-planning system rather than optimizing an isolated ML metric.

## 1. Executive decision

The solution will be a decision-focused, scenario-native pipeline:

```text
raw time series at scenario cutoff
    -> causal daily feature extraction
    -> calibrated conditional EOL distribution
    -> expected timing cost for every battery/day
    -> building/day selection and room batching
    -> route ordering and capacity repair
    -> exact evaluator replay
    -> valid complete plan
```

The primary Task 1 output is not a point RUL or a hard deadline. It is a calibrated conditional failure CDF for every day in the 42-day planning horizon, with summary statistics and quality flags. Task 2 chooses service dates by minimizing expected official cost jointly with travel, labor, room/building changes, overtime, and weekly capacity.

The final system must always retain a deterministic greedy planner. It is both a benchmark and the production fallback if a mathematical solver times out or returns an invalid plan.

### 1.1 Five-day scope rule

The architecture describes the best complete system, but implementation is governed by P0/P1/P2 priorities in Section 10. P0 must work end-to-end before any P1 work starts. P2 is attempted only when measured held-out results show that the existing component is a bottleneck. A simple, evaluated, valid pipeline is more valuable than several unfinished model and solver variants.

## 2. Known facts and assumptions

### 2.1 Confirmed from the saved EDA

- 461 devices across 24 buildings.
- 8,520,098 time-series readings.
- 82 observed EOL events and 379 right-censored devices (82.2% censored).
- Median observed readings per device: 18,389; range: 7,213 to 32,093.
- Eight voltage values and eight temperature values are missing; no duplicate `(device_id, end_time)` rows were found.
- The population Kaplan-Meier median is not reached.
- Pooled voltage-temperature correlation is 0.341, while per-device correlations vary widely. Temperature effects must therefore be modeled within-device and over time, not treated as a single global linear correction.

### 2.2 Confirmed from `batteryswap_public==0.3.4`

- The planning horizon defaults to 42 days. Scenario settings can override defaults and must be read dynamically.
- The score is the mean scenario cost. Lower is better.
- Cost components are battery work, building change, room change, travel, overtime, daily/weekly-limit penalties, early replacement, and late replacement.
- Default timing penalties are `0.5` hour-equivalent per day early and `10.0` per day late.
- Default work costs are 0.25 hours per battery, 0.5 hours per room change, and 1.0 hour per building change.
- The technician returns to base at the end of each worked day.
- Plan row order is operational route order within a day.
- A submitted plan must contain every active battery exactly once, have no duplicates, use normalized dates, be sorted by date, and contain no action before scenario start.
- The evaluator validates completeness before dropping rows after the planning horizon. A battery intentionally not serviced in-horizon must therefore appear once on a date after the horizon.
- An observed EOL inside the horizon that is not serviced in-horizon causes an individual emergency visit in the evaluator.
- Current limit fields are implemented as penalties, despite being described as hard limits. The planner will support both strict-cap and score-aligned modes so evaluator changes do not require a redesign.
- In version 0.3.4, both planned swaps and observed EOL are included when their timestamp is exactly `start_time + planning_window_days`; the current evaluator endpoint is inclusive.

### 2.3 Verified from the downloaded train split

- Train contains 48 scenarios from 2025-09-01 through 2026-07-27, stepped forward exactly seven days at a time.
- Every scenario starts Monday at 00:00, uses a 42-day horizon, and therefore has 43 candidate dates under the evaluator's inclusive endpoint (`day 0..42`).
- For all 48 scenarios, the maximum timestamp in the active cut equals `scenario_start` exactly. The official planner adapter can therefore recover the train planning origin as `max(battery_data.end_time)` without rounding.
- All numerical evaluation settings are constant. `base_location` and `base_room` vary by scenario and must never be cached as global constants.
- There is one shared 24-by-24 travel matrix across all scenarios. It is complete, exactly symmetric, and has no triangle-inequality violations.
- Travel time on the diagonal is 0.0333 hours rather than zero. Preserve the evaluator's values exactly.
- Train has 79 rooms. Active batteries range from 381 to 458 per scenario.
- Every scenario contains observed EOL inside its window: 2 to 19 events, with 454 battery-scenario event instances in total.
- The official loader reads the full train split in approximately 4.6 seconds in the current local environment.

These facts become automated regression assertions so a dataset or evaluator revision fails loudly.

### 2.4 Initial local P0 reference scores

Mean train cost across 48 scenarios using the public evaluator:

| Policy | Mean total cost | Main observation |
|---|---:|---|
| All defer | 3324.68 | Dominated by 3056.25 late-swap cost plus emergency visits. |
| Latest voltage, one battery per day | 3980.78 | Worse than defer; 2198.51 early-swap cost overwhelms reduced late cost. |
| Oracle EOL date, simple building/room batching | 205.24 | Zero timing penalty, but still 151.39 mean overtime/daily/weekly penalties plus route/work cost. |

These are engineering references, not leaderboard results. The oracle policy is not an optimizer or lower bound: scheduling every due battery on its exact EOL date can create avoidable route and capacity penalties. A stronger oracle planner may move some swaps earlier and batch visits.

The three-policy run took approximately 138 seconds because the public `iterate_scenarios()` implementation rescans/copies the raw history at each cutoff. Production feature extraction must preaggregate once and reuse compact daily state.

### 2.5 Still to verify externally or during final packaging

- Whether the final competition image uses the same evaluator behavior and `batteryswap_public==0.3.4`.
- Exact package versions available in the final CPU-only execution image.
- Public/private active-set, building, travel, and event-distribution shift.
- Full end-to-end runtime and peak memory after the real Task 1 and Task 2 implementations exist.

These checks can change constants and solver settings, but not the architecture or API contract below.

## 3. Optimization target

For a battery with EOL date `T` and proposed service date `d`, the evaluator's timing term is:

```text
C_timing(d, T) = c_early * max(T - d, 0)
               + c_late  * max(d - T, 0)
```

Ignoring logistics, the Bayes-optimal service date is the `q`-quantile of the conditional EOL distribution where:

```text
q = c_early / (c_early + c_late)
```

With default costs, `q = 0.5 / 10.5 = 0.0476`. This explains why `p10` must not be a fixed universal deadline: the best risk threshold comes from scenario costs, route economics, and capacity, and defaults place the isolated optimum near `p05`.

Task 2 will minimize:

```text
expected timing cost
+ battery work
+ room/building transition cost
+ travel and return-to-base cost
+ overtime and capacity penalties
+ risk-adjustment term for calibration uncertainty
```

The official `evaluate_plan()` result is the final model-selection metric. C-index, Brier score, calibration error, and log loss are diagnostic metrics, not the competition objective.

## 4. System architecture

### 4.1 Offline training

1. Load raw train data once and normalize timestamps.
2. Construct training examples at official scenario cutoffs plus safe synthetic historical cutoffs.
3. Generate causal features using only readings at or before each cutoff.
4. Split by held-out buildings and time before fitting any transform, model, or calibrator.
5. Produce out-of-fold daily CDFs for planner development.
6. Fit candidate models and calibrators.
7. Select the calibrated Task 1 model with proper prediction metrics and held-out end-to-end evaluator cost; tune Task 2 risk policy separately.
8. Fit the final Task 1 artifacts on all permitted training data.
9. Package Task 1 and Task 2 inside one serializable `Planner` implementation.

### 4.2 Scenario inference

1. Derive and validate the prediction origin and planning clock without assuming they are identical.
2. Identify exactly the active batteries supplied by `locations`.
3. Aggregate raw readings to compact daily series.
4. Compute causal cutoff features.
5. Predict and calibrate conditional EOL curves.
6. Validate the Task 1 contract.
7. Generate a score-aligned plan and repair any capacity or validity issue.
8. Append every deferred battery once after the horizon.
9. Run `check_plan_valid()` before returning.

## 5. Task 1 specification: conditional EOL risk

### 5.1 Learning unit and labels

The learning unit is `(battery_id, cutoff_time)`, not an hourly sensor row.

For each cutoff `t`, include only batteries alive at that time. For each prediction horizon `h`, define the multi-horizon label exactly:

```text
label(t, h) = 1       if an observed EOL satisfies T <= t + h
label(t, h) = 0       if the device is observed alive through t + h
label(t, h) = MASKED  if follow-up ends before t + h without an observed EOL
```

An EOL after `t + h` is a valid zero because the device is known to survive the horizon. A right-censored device is a valid zero only for horizons ending on or before its censoring time. Masked targets contribute no loss. This rule must have unit tests at the exact EOL/censoring boundary.

Official scenario cutoffs are the highest-value examples because they match inference. Add synthetic cutoffs to increase event coverage, subject to:

- No feature may use post-cutoff data.
- Synthetic cutoffs must respect deployment and observation boundaries.
- Repeated examples from one battery receive weights so long-lived devices do not dominate.
- All cutoffs from one building stay in the same outer validation fold.
- Calibration is fit only on out-of-fold predictions.

### 5.2 Feature families

Compute features from a daily robust series, preferably using the official `smooth_series()` behavior as one candidate preprocessing branch.

**State and age**

- Age since deployment and observed-history length.
- Latest valid voltage and temperature.
- Distance to the EOL voltage threshold and threshold bands.
- Time since last valid reading.

**Voltage degradation**

- Robust level statistics over 1, 3, 7, 14, 28, 56, 90, and 180 days.
- Robust slopes over 7, 14, 28, 56, 90, and 180 days.
- Curvature or slope change between recent and long windows.
- Recent minima, lower quantiles, range, IQR, and volatility.
- Fraction of readings below clinically relevant voltage bands.
- Extrapolated threshold-crossing time from robust linear and local-curvature fits.

**Temperature-conditioned state**

- Recent temperature level, range, and volatility.
- Voltage residual relative to an age/temperature baseline learned without validation leakage.
- Trend and lower quantiles of the residual.
- Stable-temperature voltage features to separate transient measurement effects from degradation.

**Data quality**

- Reading count and expected-count ratio by window.
- Longest and recent data gaps.
- Missingness and smoothing coverage.
- Feature-window availability flags; missing windows are imputed with fold-fitted values plus indicators.

**Context without identity leakage**

- Do not use raw `device_id`, `building_id`, or `room_id` as predictive categories.
- Causal building aggregates may be used because unseen test buildings still contain contemporaneous peers: leave-one-device-out median voltage, slope, age, low-voltage fraction, and device count.
- Room aggregates are optional and must pass held-out-building validation.

### 5.3 Candidate models

Use a small, prioritized CPU-safe model set:

1. **P0 baseline:** one censoring-aware CoxPH or Weibull AFT model that also supplies an initial long-tail estimate.
2. **P0/P1 primary model:** discrete-time hazard or direct multi-horizon classifier using runtime-available tree boosting (`HistGradientBoostingClassifier`) or ExtraTrees.
3. **P1 trajectory feature/model:** robust extrapolation to the EOL threshold, retained when it adds stable near-failure value.
4. **P2 ensemble:** only after one calibrated primary model works end-to-end and an additional model improves out-of-fold scenario cost enough to justify its runtime and packaging risk.

Do not rely on LightGBM, XGBoost, or scikit-survival until the official execution image is proven to contain them. Added local requirements do not guarantee evaluator availability.

Any P2 ensemble combines out-of-fold predictions. Weight selection must preserve calibration and improve held-out planner cost. Complexity is accepted only when the score gain is stable across building/time folds.

### 5.4 Calibration and curve repair

- Calibrate probabilities on out-of-fold predictions using a low-variance method appropriate to the event count: Platt/beta calibration first, isotonic only when sample support is adequate.
- Evaluate calibration at 7, 14, 21, 28, 35, and 42 days and across low-, medium-, and high-risk strata.
- Enforce `0 <= CDF(d) <= 1` and monotonicity over day using cumulative maximum or isotonic projection.
- Task 1 emits only its best estimate of the real conditional distribution. It must not alter probabilities to express planner risk aversion.
- Task 2 may derive an effective decision-risk curve from the calibrated CDF, but it must preserve the raw CDF and log the policy separately.
- Model the survival tail beyond 42 days. At minimum, emit survival probability through the forecast horizon and conditional mean excess lifetime after it. Fit the tail to the longest horizon supported by censoring and validate it separately.

### 5.5 Task 1 acceptance criteria

- No feature leakage in automated cutoff tests.
- Exactly one forecast curve for every active battery.
- CDF finite, bounded, and monotone for every battery.
- Tail probability is consistent with the last CDF value within `1e-6`, and conditional tail means are finite and non-negative.
- Deterministic output for a fixed model, input, and seed.
- Better than no-information and Cox baselines on out-of-fold integrated Brier score or log loss.
- Most importantly, improves held-out total planner cost or supplies complementary ensemble value.

## 6. Task 1 -> Task 2 API contract

### 6.1 Contract version

The contract identifier is:

```text
batteryswap-risk-forecast/v1
```

Any incompatible schema or semantic change increments the major version. Task 2 rejects unknown major versions immediately.

### 6.2 Python interface

```python
from dataclasses import dataclass
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class ForecastMetadata:
    contract_version: str
    model_version: str
    prediction_origin: pd.Timestamp
    forecast_end_date: pd.Timestamp
    horizon_days: int


@dataclass(frozen=True)
class RiskForecast:
    metadata: ForecastMetadata
    curves: pd.DataFrame
    tail: pd.DataFrame
    summaries: pd.DataFrame


class RiskForecaster(Protocol):
    def predict(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        *,
        prediction_origin: pd.Timestamp,
        horizon_days: int,
    ) -> RiskForecast: ...
```

The official `Planner.plan(...)` adapter derives `prediction_origin`, constructs a separate planning clock, calls this interface, validates the result, and sends it to Task 2. Task 1 may not read travel costs, alter probabilities for planner utility, or inspect planner decisions. `generated_at` is intentionally excluded so identical inputs produce identical contract objects.

### 6.3 Canonical identifiers

The shared canonical identifier is always `battery_id: str`.

- At Task 1 input, the adapter renames raw `device_id` to `battery_id`.
- At the official locations boundary, the adapter renames `battery` to `battery_id`.
- At final plan output, the adapter renames `battery_id` back to the required `battery` column.
- Task 1 and Task 2 implementation code must not guess among these names or perform independent mappings.
- The adapter asserts exact set equality after every mapping.

### 6.4 `curves` table

Long-form DataFrame with one row per active battery and daily forecast date covered by the model:

| Column | Type | Meaning |
|---|---|---|
| `battery_id` | string | Canonical ID matching the active location set exactly. |
| `forecast_date` | datetime64 | Normalized calendar date, timezone-consistent. |
| `failure_cdf` | float64 | `P(T <= forecast_date | T > prediction_origin, history <= prediction_origin)`. |

Required invariants:

- Keys `(battery_id, forecast_date)` are unique.
- Battery IDs equal the active location set exactly.
- Every battery has the same contiguous sequence of daily forecast dates with no gaps.
- The date sequence covers every candidate service date constructed by the planning clock.
- `failure_cdf` is finite, bounded, and non-decreasing.

Task 2 derives `failure_pmf` by differencing the repaired CDF and derives survival as `1 - failure_cdf`. These redundant columns are not serialized across the ownership boundary.

### 6.5 `tail` table

One required row per active battery:

| Column | Type | Meaning |
|---|---|---|
| `battery_id` | string | Canonical join key. |
| `prob_survive_horizon` | float64 | Probability of EOL after `forecast_end_date`; equals `1 - final failure_cdf`. |
| `mean_excess_rul_days_given_survival` | float64 | `E[T - forecast_end_date | T > forecast_end_date]` in days. |

This table makes expected early-replacement cost identifiable beyond the 42-day decision window. Task 1 should estimate the tail using a survival/AFT component or a validated long-horizon extrapolation, preferably through 180-365 days or the longest defensible range. It must document any cap. Task 2 must not interpret a capped mean as a claim that all batteries fail at the cap.

### 6.6 `summaries` table

Optional diagnostic table with one row per active battery:

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `battery_id` | string | yes | Join key. |
| `q05_days` | float64 | no | Conditional 5th percentile RUL. |
| `q10_days` | float64 | no | Conditional 10th percentile RUL. |
| `q25_days` | float64 | no | Conditional 25th percentile RUL. |
| `q50_days` | float64 | no | Conditional median RUL. |
| `data_quality` | float64 | yes | Score in `[0,1]`; higher means more reliable input history. |
| `cold_start` | bool | yes | Insufficient history for normal feature path. |

Quantiles beyond the fitted range are `NaN`, never an invented finite deadline. Task 2 must use `curves` and `tail` as its mathematical inputs. Summaries are for diagnostics, initialization, and fallback selection only.

### 6.7 Planning-clock and time semantics

- `prediction_origin` is the latest instant known to Task 1 and all forecasts are conditional on survival through that instant.
- `scenario_start` is the evaluator's planning origin. It is conceptually distinct from `prediction_origin` even if train data proves they coincide.
- The current evaluator includes actions on `scenario_start + planning_window_days`. Candidate service dates are therefore all normalized dates satisfying `date >= scenario_start` and `date <= scenario_start + planning_window_days`.
- Do not hard-code 42 or 43 candidate dates. The count depends on endpoint semantics and whether `scenario_start` is midnight.
- The official `Planner.plan()` signature does not expose `scenario_start`. In all 48 train scenarios, `max(battery_data.end_time)` equals the true scenario start exactly, so v1 uses that value as both `prediction_origin` and the inferred planning start.
- Assert that the inferred start is normalized to midnight. If a future split violates this assumption, fail over to a documented conservative date rule rather than silently shifting the window.
- Train has exactly 43 candidate dates, but code still constructs the date set from timestamps and settings instead of hard-coding 43.
- EOL on the service date has zero timing penalty in the current evaluator, so `failure_cdf` uses `T <= forecast_date`.
- All timestamps use one explicit timezone convention end-to-end. Conversion to timezone-naive values is allowed only at the official evaluator boundary if required by the package.

### 6.8 Failure behavior

- A single bad battery must not crash the scenario. Task 1 returns a conservative population-prior curve with `cold_start=True` and low `data_quality`.
- If Task 1 fails globally, Task 2 switches to a versioned no-information forecast fixture.
- NaN/Inf values, missing batteries, duplicate keys, or non-monotone curves are contract failures and are repaired only by the adapter's documented conservative fallback, never silently passed through.

### 6.9 Parallel-development fixture

Task 2 development uses a deterministic mock forecast with the exact v1 schema and three profiles:

- `urgent`: rising from meaningful 7-day risk to high 42-day risk.
- `medium`: low near-term risk with material late-horizon risk.
- `defer`: near-zero 42-day risk.

Every profile includes a consistent tail row. The fixture must also include one cold-start battery and one building with multiple rooms. The fixture lives in tests and is owned jointly; either side changing it requires contract review.

## 7. Task 2 specification: score-aligned planning

### 7.1 Decision set

For each active battery, choose exactly one action:

- Service on one feasible date in the planning window; or
- Defer by assigning a date strictly after the planning window.

For in-horizon actions, choose the ordered sequence of buildings, rooms, and batteries for each day. The output row order is therefore part of the decision, not presentation formatting.

### 7.2 Expected battery timing cost

For each battery and candidate service date, derive daily PMF from the calibrated CDF and compute expected asymmetric timing loss. For service date `d`, the tail contribution to early-replacement cost is:

```text
prob_survive_horizon
* early_penalty_daily
* ((forecast_end_date - d).days + mean_excess_rul_days_given_survival)
```

This assumes `d <= forecast_end_date`, as required for in-window candidates. Unit-test the complete calculation against hand-computed discrete distributions.

For defer, estimate the expected emergency cost for EOL within the horizon, including:

- Late timing penalty at the evaluator's emergency-service date approximation.
- Individual base-to-building-to-base travel.
- Building/room transition and battery work.
- Expected overtime or weekly penalty where material.

Validate the analytical cost table against the exact evaluator on deterministic micro-scenarios. Monte Carlo validation is P2 and is added only if analytical tail approximations remain a measured source of error.

### 7.3 Decision-risk policy

Task 1 probabilities remain calibrated estimates of reality. Task 2 may apply a separate, explicit decision policy such as conservative probability scaling, an uncertainty margin, or a low-data-quality premium. Policy parameters are tuned only on out-of-fold scenario cost.

The planner must retain both `failure_cdf` and its derived `effective_risk`; logs and reports must never present the latter as a calibrated probability. This separation lets the team change risk appetite without retraining or corrupting Task 1.

### 7.4 Optimization decomposition

The deterministic greedy planner is P0. Local search is P1. CP-SAT, exact route dynamic programming, and OR-Tools routing are P2 and are implemented only after profiling shows a material planner gap.

**Optional P2 Stage A: battery-day assignment**

- Binary decision `x[i,d]` for battery `i` on day `d`, plus `defer[i]`.
- Building activation `y[b,d]` and room activation `z[b,r,d]`.
- Link battery assignments to their building and room activations.
- Include exact expected timing cost and conservative lower-bound visit costs.
- Respect configurable strict daily/weekly limits or evaluator penalty mode.
- Solve with CP-SAT under a fixed time budget and deterministic seed.

**Optional P2 Stage B: daily route ordering**

- Group batteries contiguously by building and room.
- Solve the daily building route from base and back to base using exact enumeration/dynamic programming when the activated building count is small; otherwise use OR-Tools routing or deterministic local search.
- Within a building, group rooms contiguously; within a room, battery order is irrelevant to evaluator cost.

**P1 Stage C: exact repair and local search**

- Replay the exact evaluator mechanics.
- Repair invalid dates, overload, and ordering.
- Explore moving a battery/day, activating or removing a building visit, bundling a room, swapping day clusters, and deferring/reinstating a battery.
- Accept moves by exact or analytically equivalent score delta.
- Stop deterministically on time budget and return the best valid incumbent.

### 7.5 Required greedy planner

Start with every battery deferred. Repeatedly choose the feasible insertion with the best expected score reduction:

```text
benefit = avoided expected emergency/timing cost
        - incremental route/work/capacity cost
```

After activating a building/day visit, reconsider every other battery in that building because its marginal travel cost has fallen. Then run day-shift, bundle, remove, and route-order local search.

This planner must be able to complete every scenario without OR-Tools. The final solver result is accepted only when it beats the greedy incumbent under the same expected-cost function and passes validation.

### 7.6 Plan construction and validation

- Every active battery appears exactly once.
- Deferred batteries use one common normalized date strictly after the planning clock's inclusive end timestamp.
- Rows are sorted by `day`, then by the computed route order; do not re-sort alphabetically afterward.
- No action occurs before scenario start.
- Call official `check_plan_valid()` before returning.
- During local evaluation, call official `evaluate_plan()` and record every cost component.
- If any final check fails, return the last known-valid greedy plan.

### 7.7 Task 2 acceptance criteria

- Valid on empty-risk, all-urgent, one-building, multi-building, sparse-room, and cold-start fixtures.
- Never loses a battery or schedules duplicates.
- Exact travel replay matches evaluator cost on hand-computed micro-scenarios.
- Greedy planner beats the official one-swap-per-day baseline on held-out scenarios.
- Optimized planner is never worse than its greedy incumbent according to its selection objective.
- Oracle-EOL plus planner establishes a strong upper bound and diagnoses planner limitations.
- OOF-risk plus planner improves held-out total cost consistently across folds, not only in aggregate.

## 8. Validation and experiment design

### 8.1 Outer validation

Use grouped temporal evaluation:

- Hold complete buildings out of Task 1 fitting to emulate unseen-building generalization.
- Within each fold, train only on information causally available before its validation cutoffs.
- Balance folds by observed EOL count as far as the 24-building structure allows.
- Evaluate only with out-of-fold forecasts when tuning Task 2 or calibration.

No random row split is permitted. No cutoff from a validation building may contribute to feature normalization, imputation, model fitting, calibration, or planner hyperparameter tuning.

### 8.2 Prediction diagnostics

- Horizon-specific Brier score and log loss at 7/14/21/28/35/42 days.
- Calibration intercept/slope and reliability tables.
- Precision/recall among the top `k` urgent batteries, where `k` reflects capacity.
- Time-dependent concordance as a secondary ranking metric.
- Performance by building, age, voltage band, history length, and data-quality bucket.

### 8.3 End-to-end diagnostics

For every fold and scenario, log:

- `total_cost` and all official cost components.
- Number of planned in-horizon swaps and deferred batteries.
- Buildings/rooms visited per day and total route hours.
- Overtime and capacity-penalty counts.
- Predicted risk captured by scheduled batteries.
- Oracle gap, calibration gap, and planner gap.

Evaluate four predictor tiers with the same planner:

1. Oracle EOL/distribution upper bound.
2. Out-of-fold Task 1 forecasts.
3. No-information population prior.
4. Stress forecasts with under-confidence, over-confidence, and missing history.

Evaluate each Task 1 candidate with both greedy and optimized planners. This separates prediction value from optimizer quality.

### 8.4 Model and planner selection gate

A change is promoted only when:

- It improves mean held-out official cost.
- The improvement is not driven by one scenario/building alone.
- Tail-risk regressions are understood.
- Runtime and peak memory remain inside a safety margin.
- The full run is deterministic or seed-stable.

Use paired scenario-level score differences and bootstrap confidence intervals. Prefer the simpler candidate when the improvement is within noise.

## 9. Team parallelization and repository boundaries

### Task 1 owner

- Owns feature generation, labels, model fitting, calibration, forecast validation, and serialized model artifacts.
- Must emit only `batteryswap-risk-forecast/v1` at the integration boundary.
- Supplies calibrated OOF forecasts, explicit tail estimates, and the no-information fallback fixture.
- Does not encode planner risk aversion in the forecast probabilities.

### Task 2 owner

- Owns expected-cost construction, greedy policy, CP-SAT assignment, routing, repair, and plan validation.
- Develops exclusively against the contract fixture until real OOF forecasts arrive.
- Must not depend on Task 1 implementation classes or private feature columns.
- Owns any decision-risk adjustment and keeps it distinct from the calibrated Task 1 CDF.

### Shared ownership

- Contract dataclasses and validators.
- Exact evaluator harness and scenario score reports.
- End-to-end submission packaging and runtime profiling.
- Contract changes require both owners and a version update when incompatible.

Suggested module boundaries:

```text
src/
  data/              # loading and scenario-cutoff normalization
  features/          # causal daily features
  risk/              # Task 1 models, calibration, contract producer
  planning/          # Task 2 cost model, greedy, solver, routing
  contracts/         # shared v1 dataclasses and validators
  evaluation/        # exact local evaluator harness and reports
  submission/        # official Planner adapter and serialization
tests/
  fixtures/          # mock v1 forecasts and micro-scenarios
```

Task 1 and Task 2 branches can modify their owned directories independently. Shared contract changes are integrated first and kept small.

## 10. Implementation sequence

### P0: must work end-to-end

**Shared**

- Download train data and inspect every scenario setting and start timestamp.
- Pin and hash the public evaluator source/version.
- Encode the verified inclusive horizon behavior and prediction-origin/scenario-start mapping as regression tests.
- Freeze the canonical ID adapter and minimal v1 contract only after those clock tests pass.
- Reproduce the official baseline score on all train scenarios.
- Build deterministic micro-scenarios for timing, routing, overtime, deferral, and emergency behavior.

**Task 1**

- Build causal scenario-cutoff features and exact masked multi-horizon labels.
- Produce one no-information prior, one censoring-aware baseline/tail model, and one strong short-horizon risk model.
- Emit schema-valid out-of-fold curves and tail tables.

**Task 2**

- Run the official evaluator harness.
- Establish oracle, all-defer, and official/naive baselines.
- Implement a valid defer-aware greedy planner with building/room batching.
- Return a valid complete plan for every scenario and mock fixture.

### P1: likely score gain after P0 is stable

- Task 1: calibrate the primary model, improve long-tail estimates, and add temperature/voltage residual and trajectory features.
- Task 2: implement exact expected timing cost with tail, decision-risk policy, defer logic, and route-aware local search.
- Shared: tune only on OOF scenario cost, run the highest-value ablations, and profile clean-process runtime/memory.

### P2: only after a measured bottleneck

- Add a Task 1 ensemble or extra survival model only if the OOF forecast gap is limiting end-to-end score.
- Add CP-SAT, exact route DP, OR-Tools routing, or advanced repair only if oracle plus greedy exposes a material planner gap.
- Add Monte Carlo validation and bootstrap confidence intervals only when scenario-level results are too unstable for a decision.
- Prepare conservative and travel-efficient submission variants only after the balanced primary is frozen and reproducible.

### Stop/go rules

- Do not start P1 until P0 generates valid plans and score decompositions on all train scenarios.
- Do not start a P2 Task 1 experiment when oracle-to-OOF gap is already small relative to the planner gap.
- Do not start a P2 optimizer when oracle plus greedy is already close to the best observed oracle score.
- Stop any addition that does not show paired scenario-level improvement, fits poorly inside the runtime margin, or complicates submission packaging without stable gain.

## 11. Runtime and reliability requirements

- Target at most 20 minutes end-to-end locally to preserve margin under a 30-minute evaluator limit.
- Target peak memory below 20 GB under a 32 GB limit.
- Aggregate 8.5M raw rows once per split; do not repeatedly scan raw history per battery/candidate day.
- Cache only scenario-independent daily aggregates inside one process; never leak future values into a cutoff feature.
- Fix all random seeds and solver worker settings needed for reproducibility.
- Package no network calls, dynamic downloads, or unverified dependencies.
- Log phase timings and return a valid greedy result if the optimizer reaches its deadline.

## 12. Highest-value experiments

Run in this priority order before broad hyperparameter search:

1. **P0:** Exact evaluator baseline versus all-defer, earliest-voltage, risk-only, and risk-plus-building-batching policies.
2. **P0:** Oracle EOL plus greedy planner to quantify the Task 2 ceiling.
3. **P0:** Fixed first-60-day features versus cutoff-relative recent-window features.
4. **P1:** Raw voltage features versus smoothed and temperature-residualized features.
5. **P1:** Censoring-aware baseline versus primary tree hazard, both evaluated with the same planner.
6. **P1:** Uncalibrated versus calibrated CDFs under official total cost.
7. **P1:** Fixed `p10` deadline versus expected-cost scheduling with explicit tail.
8. **P1:** Battery-level ordering versus building/room batching and local search.
9. **P1/P2:** Strict capacity versus evaluator penalty mode.
10. **P2:** Greedy versus CP-SAT/routing, including runtime and stability, only if oracle results justify it.

## 13. Definition of done

The solution is competition-ready when:

- The minimal v1 contract, canonical ID adapter, planning clock, and tail semantics are implemented and tested by both owners.
- All forecasts are causal, calibrated, monotone, and complete.
- Every generated plan passes official validation and contains all active batteries once.
- End-to-end selection uses only out-of-fold or genuinely held-out scenario predictions.
- Greedy, fallback, and oracle benchmarks, plus any P2 solver actually used, are recorded with component costs.
- A clean CPU-only run completes inside resource limits without network access.
- The final serialized planner can generate public/private submissions through the official entry point.
- The README documents one-command training, evaluation, and submission generation.

## 14. Immediate unresolved risks

1. Public/private distribution shifts cannot be measured from train. The solution must avoid train-building identity features and hard-coded base/travel assumptions.
2. Tail behavior is weakly identified by only 82 unique observed failures and heavy censoring. The long-tail estimator and cap require held-out calibration checks before its expected cost is trusted.
3. The current preprocessing uses only the first 60 days after deployment. That is unsuitable as the final scenario-cutoff feature strategy.
4. The current Cox artifact is a baseline, not a validated winner; the other modeling/evaluation notebooks are currently empty files.
5. The simple oracle reference still averages 151.39 in overtime/daily/weekly penalties. We need an oracle greedy planner with early moves and route-aware batching to establish a meaningful Task 2 ceiling.
6. The planner needs automated evaluator edge tests for inclusive horizon dates, post-horizon deferral, emergency visits, non-zero diagonal travel, changing bases, and capacity penalties.

These are execution risks, not reasons to change the core architecture.
