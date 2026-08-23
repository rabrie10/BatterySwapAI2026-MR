# Task 2 Implementation Guide

Status: implemented and locally verified, last updated 2026-08-18.

This document is the engineering specification and operating guide for the
BatterySwapAI 2026 work-order planner. It describes the code currently in this
repository, the contract it expects from Task 1, the optimization model, the
validation strategy, and the experiments required before a final submission.

The goal is not to predict a convenient replacement date for every battery.
The goal is to produce the complete, ordered work plan with the lowest expected
official `total_cost`, while remaining valid and fast on unseen scenarios.

## 1. Scope

Task 2 decides all of the following for every scenario:

- whether each active battery should be replaced inside the planning window;
- the calendar day of every planned replacement;
- which replacements should be deferred beyond the planning window;
- which batteries should be grouped into the same trip, building, and room;
- the execution order within each day;
- how much forecast risk to accept in exchange for lower early-replacement and
  operational cost.

Task 2 can be developed before the final Task 1 model is ready. The repository
supports three forecast sources:

1. A train-only oracle used to measure the Task 2 engineering ceiling.
2. A fitted Task 1 artifact implementing the versioned forecast contract.
3. A deterministic voltage-trend fallback used for integration and failure
   recovery. It is not intended to be the final leaderboard model.

## 2. Official Planner Interface

The evaluator calls a `Planner` instance once per scenario:

```python
class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        timeseries: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings: EvaluationSettings,
    ) -> pd.DataFrame:
        ...
```

The production implementation is `CompetitionPlanner` in
`batteryswap_solution/planner.py`.

### Inputs

| Input | Meaning | Task 2 use |
| --- | --- | --- |
| `timeseries` | Causal voltage and temperature history available at the scenario cutoff | Passed to Task 1; latest timestamp defines the scenario start |
| `locations` | Active batteries and their building, room, and observation metadata | Completeness, grouping, proxy EOL handling, and logistics |
| `travel_costs` | Directed building-to-building travel times | Route construction, exact replay, and emergency costs |
| `settings` | Scenario-specific work times, limits, penalties, base, and horizon | Used directly; competition constants are not hard-coded |

### Output

The returned `DataFrame` has exactly these columns:

```text
day,battery
```

Every active battery must appear exactly once. A date inside the inclusive
planning horizon means "replace this battery". A date strictly after the
horizon means "do not replace it during this scenario". Rows sharing a day are
executed in their returned order, so row order is part of the route decision.

## 3. System Architecture

The production data flow is:

```text
scenario data
    -> Task 1 probabilistic failure forecast
    -> forecast contract validation
    -> expected early/late/defer cost tables
    -> CP-SAT service/defer and day assignment
    -> daily building and room routing
    -> evaluator-aligned replay and local improvement
    -> validity and official-evaluator consistency checks
    -> complete work-order plan
```

The implementation is split by responsibility:

| File | Responsibility |
| --- | --- |
| `batteryswap_solution/forecast.py` | Task 1 contract, strict validation, canonical ordering, and fallback forecasting |
| `batteryswap_solution/costs.py` | Expected early, late, unobserved-EOL, deferred, and emergency costs |
| `batteryswap_solution/optimizer.py` | Joint service/defer and service-day assignment using OR-Tools CP-SAT |
| `batteryswap_solution/routing.py` | Daily building route and room/battery execution order |
| `batteryswap_solution/replay.py` | Fast replay of the official operational cost semantics |
| `batteryswap_solution/planner.py` | End-to-end orchestration, robust local search, validation, and fallbacks |
| `script.py` | Official submission entry point and artifact/configuration loading |
| `tools/benchmark_task2.py` | Train-only oracle and fallback benchmarks |
| `tests/test_task2.py` | Contract, cost, routing, replay, boundary, and plan-validity tests |

## 4. Task 1 to Task 2 API Contract

Task 2 must consume a distribution, not only one RUL point estimate. A median
RUL cannot represent the asymmetric cost of early and late replacement or the
probability that a deferred battery creates an emergency visit.

The contract version is:

```text
batteryswap-risk-forecast/v1
```

Task 1 serializes an object implementing `RiskForecaster.predict()`:

```python
def predict(
    battery_data: pd.DataFrame,
    locations: pd.DataFrame,
    *,
    prediction_origin: pd.Timestamp,
    horizon_days: int,
    evaluation_observation_end: pd.Timestamp,
) -> RiskForecast:
    ...
```

### 4.1 Metadata

`RiskForecast.metadata` is a `ForecastMetadata` with:

| Field | Required meaning |
| --- | --- |
| `contract_version` | Exactly `batteryswap-risk-forecast/v1` |
| `model_version` | Immutable identifier for the fitted model/configuration |
| `prediction_origin` | Scenario start date used by the forecast |
| `forecast_end_date` | Last forecast date |
| `horizon_days` | Number of day offsets after the origin |
| `evaluation_observation_end` | Observation boundary used for evaluator-unobserved EOL handling |

### 4.2 Daily curves

`RiskForecast.curves` contains one row for every active battery and every
candidate date:

| Column | Type | Meaning |
| --- | --- | --- |
| `battery_id` | string | Must match an active battery exactly |
| `forecast_date` | normalized date | Candidate event date |
| `failure_cdf` | float in `[0, 1]` | Probability that observed EOL occurs on or before this date |

The CDF must be complete and monotone non-decreasing. The validator repairs
only tiny floating-point monotonicity errors. Missing batteries, missing dates,
duplicates, non-finite values, out-of-range values, or inconsistent probability
mass cause the primary forecaster to be rejected.

### 4.3 Tail probabilities

`RiskForecast.tail` contains one row per active battery:

| Column | Meaning |
| --- | --- |
| `battery_id` | Active battery identifier |
| `prob_observed_after_horizon` | Probability of an observed EOL after the planning horizon |
| `mean_excess_rul_days_given_observed_after_horizon` | Conditional mean days from the final forecast date to that EOL |
| `prob_unobserved_eol` | Probability that EOL is evaluator-unobserved/censored |
| `prob_no_observed_eol_by_horizon` | Sum of observed-tail and unobserved mass |

For each battery, final CDF mass plus observed-tail mass plus unobserved mass
must equal one within tolerance. `prob_no_observed_eol_by_horizon` must equal
the sum of the two tail components.

`RiskForecast.summaries` may contain diagnostic columns such as quantiles,
uncertainty, cold-start status, or data quality. Task 2 v1 does not require
those fields for optimization.

### 4.4 Artifact handoff

The default artifact location is:

```text
models/risk_forecaster.pkl
```

The object must be loadable with the competition Python environment and must
not depend on unavailable packages or files. `script.py` also accepts a custom
path through `BATTERYSWAP_FORECASTER_PATH`.

If loading, prediction, or validation fails, `CompetitionPlanner` logs the
failure and uses `VoltageTrendForecaster`. This keeps the submission valid, but
forecast failure should still be treated as a release blocker because the
fallback is not expected to be leaderboard-competitive.

## 5. Expected-Cost Model

Let `F_i(t)` be Task 1's daily failure CDF for battery `i`. The discrete event
probability for day `t` is:

```text
p_i(t) = F_i(t) - F_i(t - 1)
```

For a planned replacement on day `d`, the event-day loss is:

```text
L_i(t, d) = early_penalty * max(t - d, 0)
          + late_penalty  * max(d - t, 0)
```

The configured late penalty is multiplied by `late_risk_multiplier`. This is a
deliberate robustness control for forecast miscalibration and private-split
risk. It is not a replacement for probability calibration.

The planned-service cost includes:

1. Expected event loss over all in-horizon event dates.
2. Expected early-replacement cost for observed EOL after the horizon, using
   its conditional mean excess RUL.
3. Expected loss for evaluator-unobserved EOL mass, using the official proxy
   date `locations.end_time + settings.unobserved_eol_days`.

Deferral is not free when an event may occur inside the horizon. The planner
models:

- expected late cost from the evaluator's sorted emergency queue;
- each battery's expected position in that queue;
- isolated emergency travel, building, room, battery, overtime, and limit cost;
- robust sampled emergency combinations during local search.

This formulation is the central design decision: Task 1 probabilities are
converted directly into the same early/late trade-off that the competition
scores, instead of being converted to arbitrary hard deadlines.

## 6. Planning Algorithm

### 6.1 Planning clock and normalization

The latest available timeseries timestamp defines the normalized scenario start.
Candidate service dates include both the start and the final horizon date. The
defer date is exactly one day after the inclusive horizon.

All battery identifiers are converted to strings and all dates are normalized
before optimization. The final-horizon Sunday is retained in the forecast but
excluded from service assignment because `batteryswap_public==0.3.4` cannot
close a plan containing work on that evaluator boundary.

### 6.2 Forecast validation

The primary Task 1 forecaster runs first. Its output is validated, reordered to
the exact active-battery/date grid, and converted into canonical tables. Any
exception activates the deterministic fallback forecast.

### 6.3 CP-SAT assignment model

For every battery `i` and candidate day `d`, the model creates a binary service
variable `x[i,d]`. It also creates one binary defer variable `z[i]`:

```text
sum_d x[i,d] + z[i] = 1
```

Additional binary variables activate each day, building/day pair, and room/day
pair. These let the objective reward grouping and charge shared setup costs
only once.

The objective approximates total expected score with:

- forecast-derived timing cost for each battery/day choice;
- expected emergency cost for deferral;
- per-battery replacement time;
- per-room and per-building change time;
- a directed base round-trip travel proxy;
- overtime penalty.

The assignment model includes conservative daily and weekly work constraints.
A small capacity margin avoids schedules that become invalid after exact route
ordering. Solver arithmetic uses scaled integers for deterministic CP-SAT
behavior.

The solver is configured with one worker, a fixed seed, and a bounded runtime.
If OR-Tools is unavailable or no feasible solution is returned, a deterministic
greedy assignment is used.

### 6.4 Daily routing

Once service days are selected, each day is routed independently:

- the base building is visited first when it contains work;
- up to 10 non-base buildings use exact Held-Karp dynamic programming;
- larger building sets use deterministic cheapest insertion followed by 2-opt;
- batteries in one room remain contiguous;
- rooms and batteries are ordered by event-risk priority with stable ID
  tie-breaking.

The route starts and ends at the base. Directed travel costs are used, so the
algorithm does not assume a symmetric matrix.

### 6.5 Evaluator-aligned replay

The CP-SAT objective necessarily uses logistics proxies. Candidate plans are
therefore scored again with a fast replay matching the installed
`batteryswap_public==0.3.4` operational semantics:

- battery, room, and building work time;
- directed travel and return-to-base travel;
- overtime;
- strict daily-limit behavior;
- weekly-limit behavior;
- emergency visits for deferred batteries.

The replay intentionally reproduces current evaluator boundary behavior,
including return travel being carried into the next `time_of_day` calculation.
This is version-sensitive and must be re-audited if `batteryswap_public`
changes.

### 6.6 Robust local search

The routed CP-SAT plan becomes the first incumbent. Local search then evaluates
score-aware moves such as:

- inserting a deferred battery into a planned visit;
- moving a battery or full building group to another day;
- merging neighboring visits;
- trying nearby dates and existing visit dates;
- deferring low-value planned work;
- repairing days near daily or weekly thresholds;
- applying compound moves across two problematic days.

For uncertain forecasts, local search uses stratified common-random-number
samples of in-horizon emergency events. The same samples score competing plans,
which reduces ranking noise. Duplicate emergency outcomes are removed.

The all-defer plan is also evaluated as a safety incumbent. Search only accepts
strict score improvements and uses a deterministic evaluation budget.

### 6.7 Final validation and fallback

Before returning, the planner:

1. Runs the official `check_plan_valid()`.
2. Computes operational cost with the fast replay.
3. Calls the official `evaluate_plan(..., eol_times=None)`.
4. Requires fast and official operational totals to agree within `1e-8`.

Any unhandled planning, validation, or replay error returns a complete all-defer
plan. This fallback protects submission validity; it is not expected to produce
a competitive score.

## 7. Default Configuration

### Planner configuration

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `late_risk_multiplier` | `1.0` | Robustness multiplier on late-swap loss |
| `local_search_evaluations` | `160` | Search budget for deterministic/low-uncertainty cases |
| `uncertain_local_search_evaluations` | `70` | Search budget when multiple emergency outcomes are sampled |
| `robust_emergency_samples` | `4` | Number of emergency outcomes used in robust scoring |
| `random_seed` | `20260818` | Reproducible sampling and tie behavior |

### CP-SAT configuration

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `solver_seconds` | `2.0` | Maximum CP-SAT solve time per scenario |
| `random_seed` | `20260818` | Deterministic CP-SAT seed |
| `cost_scale` | `1000` | Integer precision for objective costs |
| `time_scale` | `1000` | Integer precision for work hours |
| `capacity_margin_hours` | `0.05` | Buffer below daily and weekly limits |
| `objective_roundtrip_fraction` | `0.55` | Weight of base round-trip proxy in assignment objective |

These defaults are a reproducible baseline, not universal optima. Tune them on
out-of-fold scenario simulations and keep one locked configuration for private
leaderboard selection.

### Environment variables

| Variable | Effect |
| --- | --- |
| `BATTERYSWAP_PLANNER_PATH` | Load a fully serialized `Planner`; takes precedence over all other planner construction |
| `BATTERYSWAP_FORECASTER_PATH` | Load the Task 1 artifact; defaults to `models/risk_forecaster.pkl` |
| `BATTERYSWAP_LATE_RISK_MULTIPLIER` | Override late-risk robustness multiplier |
| `BATTERYSWAP_SOLVER_SECONDS` | Override CP-SAT time budget |
| `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS` | Override normal local-search budget |
| `BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS` | Override uncertain-case search budget |
| `BATTERYSWAP_ROBUST_SAMPLES` | Override emergency sample count |
| `BATTERYSWAP_DATASET_PATH` | Dataset root used by `script.py` |
| `BATTERYSWAP_SPLITS` | Comma-separated splits; official default is `public,private` |

## 8. Running and Verification

Run commands from the repository root in the same environment used for local
competition testing.

### Unit and evaluator-parity tests

```powershell
python -m unittest discover -s tests -v
```

The current suite covers:

- forecast CDF repair and structural rejection;
- asymmetric early/late expected cost;
- exact small-day building routing;
- plan completeness and validity;
- emergency queue replay equality with the official evaluator;
- evaluator-unobserved EOL behavior;
- inclusive horizon handling;
- daily and weekly threshold semantics.

### Train-only Task 2 ceiling

```powershell
python tools/benchmark_task2.py --mode oracle --limit 12
```

Oracle mode uses train EOL labels and must never be used by `script.py` or in a
competition submission. It isolates planning quality from forecast error.

The 2026-08-18 12-scenario oracle regression produced:

| Metric | Mean |
| --- | ---: |
| `total_cost` | `77.8319` |
| all-defer `total_cost` | `4885.4567` |
| battery swaps | `13.5` |
| `battery_swap` | `3.3750` |
| `building_change` | `11.5833` |
| `room_change` | `6.4167` |
| `travel` | `22.6174` |
| `overtime` | `24.2561` |
| `daily_limit` | `0.0000` |
| `weekly_limit` | `0.0000` |
| `late_swap` | `0.0000` |
| `early_swap` | `9.5833` |
| planner runtime per scenario | `11.73 s` |

The earlier simple oracle baseline scored `205.24`. The improvement supports
the planning architecture, but `77.83` is an engineering ceiling on known train
labels, not an estimate of public or private leaderboard performance.

### End-to-end fallback smoke test

```powershell
python tools/benchmark_task2.py --mode fallback --limit 3
```

This checks that the submission path works without a Task 1 artifact. It does
not validate final forecast quality.

### Local submission generation

For a local train smoke test:

```powershell
$env:BATTERYSWAP_DATASET_PATH = (Resolve-Path dataset).Path
$env:BATTERYSWAP_SPLITS = "train"
python script.py
```

For an official-like local run, use the available public/private data root and
restore:

```powershell
$env:BATTERYSWAP_SPLITS = "public,private"
python script.py
```

Confirm that `submission.csv` exists, every expected scenario is represented,
runtime is comfortably below the competition limit, and repeated runs produce
identical output.

## 9. Acceptance Criteria

A Task 2 release candidate is acceptable only when all of these hold:

- every scenario returns one and only one row per active battery;
- all deferred dates are strictly after the inclusive planning horizon;
- no candidate plan fails official validation;
- fast replay and official operational cost agree;
- deterministic reruns create identical plans and scores;
- all unit tests pass in the pinned competition environment;
- the full local public/private-equivalent run stays within the compute limit;
- no oracle label, future observation, or split-specific identifier leaks into
  planning;
- the fitted Task 1 artifact passes the v1 contract on every validation fold;
- configuration changes improve a locked out-of-fold metric, not only the
  public leaderboard.

## 10. Competition Tuning Protocol

The private leaderboard is best protected by treating planner tuning as an
offline model-selection problem.

1. Create causal scenario cutoffs from train data and keep entire buildings or
   groups out where possible.
2. Generate out-of-fold Task 1 distributions for those scenarios.
3. Score Task 2 with the official evaluator and retain every cost component.
4. Tune `late_risk_multiplier`, CP-SAT time, search budgets, robust sample
   count, capacity margin, and travel proxy weight on aggregate validation cost.
5. Track mean, median, worst-case, and tail scenario cost. A small mean gain
   that creates catastrophic scenarios is not private-safe.
6. Run ablations against all-defer, simple threshold, CP-SAT-only, no-routing,
   and no-local-search baselines.
7. Freeze the planner configuration before choosing final submissions.

Public leaderboard feedback can diagnose gross mistakes, but repeated tuning
to a small public split risks selecting noise. The primary selection signal
should be causal out-of-fold total cost and stability across scenario groups.

## 11. Current Limitations and Next Improvements

The implementation is strong but should not be described as mathematically
globally optimal. Important remaining work includes:

- Replace `VoltageTrendForecaster` with calibrated out-of-fold Task 1 survival
  distributions.
- Audit the forecast's calibration by horizon and building group; Task 2 cannot
  recover information that Task 1 does not provide.
- Tune robustness and search parameters on generated validation scenarios.
- Consider correlated building-level failure samples. Current emergency samples
  primarily use battery-level marginal risk.
- Compare the CP-SAT travel proxy with stronger route-aware lower bounds if the
  runtime budget permits.
- Re-audit replay parity whenever the official evaluator package changes.
- Measure complete public/private generation runtime with the final artifact,
  not only train oracle runtime.

No algorithm can guarantee first place without access to private labels. The
competitive strategy is to optimize the official cost exactly where possible,
model forecast uncertainty honestly, validate causally, preserve deterministic
reproducibility, and avoid leaderboard overfitting.

## 12. Task 1 Handoff Checklist

Before integrating a new Task 1 artifact, the forecasting owner should provide:

- the serialized forecaster and immutable model version;
- the training-data and feature version;
- causal out-of-fold forecast files or a reproducible generation command;
- per-horizon calibration and ranking metrics;
- confirmation that every active battery receives a complete daily CDF;
- observed-tail and evaluator-unobserved probability mass;
- runtime and memory measurements in the official CPU environment;
- a list of required Python packages and artifact files;
- a reproducibility seed and exact training command.

After integration, rerun unit tests, oracle regression, out-of-fold end-to-end
evaluation, full runtime testing, and submission generation before committing a
final submission SHA.
