# Task 2 Planner Reference

Status: working reference, last updated 2026-08-18.

This document is the Task 2-focused reference for building the work-order
planner. It translates the official challenge rules into implementation
requirements and optimization priorities.

## Objective

Task 2 receives scenario data and must return a complete battery-swap schedule
that minimizes official `total_cost`.

The planner should decide:

- which batteries to swap inside the 42-day planning window;
- which batteries to defer beyond the planning window;
- what service day each in-window swap should happen;
- the row order within each day, which defines the technician route;
- how to trade late risk against early replacement, travel, room/building
  changes, overtime, and worker-day limits.

## Official Interface

The submitted code must expose a `Planner` implementing:

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

Inputs:

- `timeseries`: causal data available up to the scenario start/cutoff.
- `locations`: active battery locations for the scenario.
- `travel_costs`: scenario-specific building-to-building travel time table.
- `settings`: scenario-specific evaluation settings, including planning window,
  base location/room, work times, and penalties.

Output:

- pandas DataFrame with exactly two columns: `day`, `battery`.
- one row per active battery.
- rows on the same day are executed in the row order returned.

## Plan Validity Rules

The output must be complete and valid:

- every active battery in `locations` appears exactly once;
- no duplicate batteries;
- no missing active batteries;
- no action before scenario start;
- `day` values are normalized dates;
- rows are sorted/grouped by day in execution order;
- batteries intentionally not swapped in the planning window are placed after
  the last planning-window day.

Important evaluator behavior from local inspection:

- Horizon is inclusive: a plan/EOL exactly at `start_time + 42 days` is inside
  the window.
- To defer a battery, schedule it strictly after the inclusive horizon, for
  example `start_date + 43 days`.
- Completeness is checked before the evaluator cuts rows after the horizon, so
  deferred rows still satisfy completeness.
- If a battery reaches EOL inside the horizon and is not planned in-window, the
  evaluator adds an emergency visit.
- The technician starts each worked day at the base and returns to base at the
  end of that day.

## Cost Components

Official cost components:

- battery swap work time;
- room change time;
- building change time;
- travel time between buildings;
- overtime beyond the normal workday;
- early replacement penalty;
- late replacement / downtime penalty;
- total worker-day availability limits.

Observed local evaluator constants for `batteryswap_public==0.3.4`:

- planning window: 42 days;
- early penalty: 0.5 per battery-day early;
- late penalty: 10 per battery-day late;
- battery swap time: 0.25 hours;
- room change time: 0.5 hours;
- building change time: 1.0 hour;
- overtime threshold: 8 hours/day;
- overtime factor: 2;
- daily max: 24 hours, with large penalty if exceeded;
- weekly max: 24 hours, with large penalty if exceeded.

These values should always be read from `settings` in submitted code, not
hard-coded, except in tests that intentionally assert known public behavior.

## Data Details That Matter For Planning

Local train diagnostics:

- 461 devices.
- 24 buildings.
- 79 rooms.
- 8,520,098 hourly readings.
- 82 unique observed EOL failures.
- 379 censored batteries.
- 48 scenarios.
- Active batteries per scenario: 381 to 458.
- EOL events inside horizon per scenario: 2 to 19.
- Every train scenario has at least 2 in-window EOL events.
- Scenario bases vary by building and room.
- Travel matrix is symmetric and complete.
- Travel diagonal is nonzero, so same-building travel assumptions must match
  evaluator behavior.
- No triangle inequality violations were found in train.

Planning implication: travel/route cost is real, but timing penalties dominate
when a high-risk battery is missed. A strong planner must combine risk selection
with batching and route ordering.

## Prediction-To-Planning Contract

Task 2 should not require a single point RUL estimate. It should consume a
calibrated risk forecast.

Required Task 1 forecast tables:

### `curves`

Columns:

- `scenario`
- `battery_id`
- `prediction_origin`
- `forecast_date`
- `failure_cdf`

Meaning:

- `failure_cdf = P(EOL <= forecast_date | alive at prediction_origin, observed data)`
- one row per battery/date inside the planning horizon;
- CDF must be monotone non-decreasing for each battery;
- values must be clipped to `[0, 1]`;
- dates should cover every service date Task 2 might consider.

### `tail`

Columns:

- `scenario`
- `battery_id`
- `prob_survive_horizon`
- `mean_excess_rul_days_given_survival`

Meaning:

- probability the battery survives beyond the planning horizon;
- expected extra lifetime beyond horizon, conditional on survival.

This is needed because early replacement cost depends on how much useful life is
lost, including batteries that survive past the horizon.

### Optional diagnostics

Useful columns:

- `q05`
- `q10`
- `q25`
- `q50`
- `data_quality`
- `cold_start`
- `model_version`

## Expected Timing Cost

For each battery and candidate service day `d`, Task 2 should compute expected
timing cost:

- early cost if service happens before true EOL;
- late cost if service happens after true EOL;
- tail early cost if EOL is likely beyond the planning horizon.

Core idea:

```text
expected_total_cost(day)
  = expected_early_replacement_cost(day)
  + expected_late_replacement_cost(day)
  + incremental_logistics_cost(day, route)
  + capacity_penalty(day/week)
```

Ignoring logistics, the optimal day under asymmetric absolute timing loss is the
CDF quantile:

```text
c_early / (c_early + c_late) = 0.5 / 10.5 = 0.0476
```

That means a fixed `p10` deadline is often too aggressive. The service threshold
should emerge from expected official cost, not from a hard-coded percentile.

## Defer Decision

Every battery must be represented, but not every battery should be swapped
inside the window.

Use a common defer date:

```text
defer_day = scenario_start_date + settings.planning_window_days + 1 day
```

For each battery, compare:

- best in-window service cost;
- expected emergency/late cost if deferred;
- logistics savings from batching;
- risk forecast uncertainty.

A good planner starts with all batteries deferred and then inserts only swaps
whose expected avoided timing/emergency cost exceeds incremental operational
cost.

## Route And Batching Strategy

Within each day, row order is route order. The planner should exploit this.

High-value batching rules:

- group same-building visits when possible;
- inside a building, group same-room batteries together;
- when a high-risk battery activates a building/day visit, reconsider other
  batteries in that building and nearby rooms;
- sort or locally optimize building order to reduce base-to-route-to-base travel;
- avoid creating a workday for a single low-risk battery unless late risk is
  clearly high.

The simplest valid strong approach:

1. Start with all batteries deferred.
2. Compute each battery's expected timing cost for each candidate day.
3. For each candidate insertion, estimate avoided deferred cost minus
   incremental route/work/capacity cost.
4. Insert the best positive move.
5. After inserting, recompute affected same-day/same-building candidates.
6. Repair route order by building and room.
7. Apply local search moves: shift day, bundle same building, remove weak swaps,
   and reorder route.

## Capacity Strategy

The evaluator charges overtime and large penalties for excessive daily/weekly
work.

Practical rules:

- keep normal days near but preferably below 8 hours unless late-risk savings
  justify overtime;
- avoid exceeding 24 hours/day;
- avoid exceeding 24 hours/week;
- prefer moving lower-risk same-building batteries earlier/later to smooth
  workload;
- treat capacity as a score tradeoff during search, not just a final repair.

## Emergency Cost Awareness

If a battery fails in-window and is not planned in-window, the evaluator adds an
emergency replacement. This can be very costly because it can create isolated
travel and timing penalties.

Task 2 should estimate emergency exposure from the forecast distribution:

- high probability of in-window EOL increases the value of scheduling;
- nearby planned visits reduce marginal cost of adding a battery;
- deferring high-risk isolated batteries may still be bad if late penalty is
  large.

## Validation Baselines

Keep these baselines for sanity checks:

- `all_defer`: valid plan where all batteries are scheduled after horizon.
- `latest_voltage_one_per_day`: simple risk proxy baseline.
- `oracle_exact_batched`: uses true train EOL dates to estimate planning ceiling.

Previously measured train means:

- all-defer mean total cost: 3324.68.
- latest-voltage one-per-day mean total cost: 3980.78.
- oracle exact batched mean total cost: 205.24.

The oracle still had capacity penalties, so a route/capacity-aware oracle should
be built to estimate the true attainable Task 2 ceiling.

## Local Tests We Need

P0 tests:

- output has exactly `day` and `battery`;
- every active battery appears exactly once;
- defer date is strictly after inclusive horizon;
- no planned row before start date;
- route order affects cost as expected;
- same-building/same-room batching reduces cost;
- base building/room changes are respected per scenario;
- travel diagonal behavior matches evaluator;
- emergency behavior is reproduced for deferred batteries that fail in-window.

P1 tests:

- expected timing cost matches hand-calculated toy examples;
- monotone CDF and tail validation catches bad Task 1 outputs;
- local search never returns an invalid plan;
- fallback plan is always valid if forecast data is missing or broken.

## Implementation Priorities

P0:

- exact evaluator wrapper;
- contract loader/validator for Task 1 forecasts;
- valid all-defer and simple risk baselines;
- greedy insertion planner with building/room batching;
- route ordering by building and room;
- local metric report across all train scenarios.

P1:

- expected-cost decision rule using full CDF and tail;
- route-aware local search;
- workload smoothing;
- stronger synthetic scenario validation;
- submission-time safety fallback.

P2:

- OR-Tools / CP-SAT for selected subproblems if measured runtime allows;
- exact small-day route DP;
- ensemble or bootstrap risk uncertainty transforms;
- multiple final submission profiles with different risk aversion.

## Common Mistakes To Avoid

- Returning only batteries we want to swap; the plan must include every active
  battery exactly once.
- Scheduling deferred batteries on the inclusive horizon day instead of after
  it.
- Using `building_id` memorization as a model feature without validating unseen
  buildings.
- Optimizing only RUL accuracy instead of official total cost.
- Hard-coding train scenario bases or travel behavior.
- Ignoring row order within each day.
- Letting a local optimizer produce invalid duplicate/missing batteries.
- Depending on network access during official evaluation.
- Adding dependencies unavailable in the fixed competition runtime.
