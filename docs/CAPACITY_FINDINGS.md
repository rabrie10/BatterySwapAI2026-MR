# Capacity post-pass: findings (final, 2026-08-22)

## Verdict

The pass is correct, deterministic, tested, and nearly free (0.088 s/scenario
mean, 0.475 s max) — but its true value is **-2.5/scenario realized**, not the
-30+ the mission banked on. The capacity pool the mission targeted mostly does
not exist on the plan side: it is emergency-day cost from missed forecasts
plus irreducible far-cluster geometry. Recommendation: **ship the pass**
(default `capacity_repair=True`; it provably never worsens the planner's
deterministic objective and is insurance against fixable overloaded days on
unfamiliar public splits) — **do not bank a -30 improvement**.

## What was built

`CompetitionPlanner._capacity_repair` (batteryswap_solution/planner.py), a
deterministic post-pass between `_local_search` and `_restore_excluded`,
gated by `PlannerConfig.capacity_repair: bool = True` (kill switch: construct
the planner with `capacity_repair=False`; script.py untouched).

Per round it replays the incumbent exactly (`replay_operational_cost`,
`include_details=True`), collects days over the daily limit and week buckets
at/over the weekly limit, and enumerates moves of building groups, contiguous
route segments (prefix/suffix of the day's route — the only split shape that
can fix a chained multi-building day), whole days, and single batteries to
±1..3/±7/±14 days, the group's timing-optimal day, and the nearest existing
workdays (week fixes restricted to out-of-bucket targets). When nothing is
over a limit it merges adjacent light days. Acceptance is doubly strict:
exact-replay **operational delta < 0** AND operational + timing delta
(`costs.service_cost`) **< 0**. Timing-funded moves are rejected — measured on
a cross-run A/B they realize badly (two accepts bought +100 penalties against
an expected-timing credit and realized +225/+171). Steepest descent,
tie-break (delta, target day, sorted batteries); caps 40 rounds / 120
candidate replays per round / 600 total; candidates re-route only touched
days (day routes are independent in the evaluator), so one candidate costs a
small routing call plus a ~ms replay. No randomness anywhere.

## The definitive measurement (paired, same incumbent)

CP-SAT's 1 s wall-clock termination re-rolls the search incumbent between
processes: 20/48 scenarios drift between *identically configured* runs, and a
no-op rerun of the ship config moved the 48-scenario mean by **-52.1** by
itself (outputs/capacity_baseline_rerun.json vs outputs/val_ship_final.json;
even daily/weekly components drift ±100 on 6 scenarios). Cross-run validation
diffs therefore cannot attribute a ±10-20 effect, and the headline runs below
are reported with that caveat.

`tools/capacity_pass_paired.py` (outputs/capacity_paired_ab.json) kills the
drift: one incumbent per scenario, official `evaluate_plan` scored with and
without the pass in the same process, all 48 scenarios.

| component (mean/scenario) | pass effect |
|---|---:|
| **total_cost** | **-2.50** |
| weekly_limit | -4.17 |
| overtime | -0.16 |
| travel | -0.08 |
| building_change | -0.06 |
| daily_limit | 0.00 |
| late_swap (realized) | +2.29 |
| early_swap (realized) | -0.32 |

9/48 scenarios changed; 5 improved (s_6 -89.3 weekly fix, s_39 -14.9 weekly
fix, s_0 -10.6, s_5 -8.0, s_25 -1.1 op-saving merges), 1 worsened (s_40 +3.0,
an op-correct merge whose realized early went against it). Pass runtime
0.088 s/scenario mean, 0.475 s max.

## Full-config validation runs (ship config, 48 scenarios OOF)

| run | mean total | daily | weekly | overtime | late | early | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| outputs/val_ship_final.json (baseline, quiet box) | 2056.52 | 52.08 | 64.58 | 62.21 | 1153.12 | 657.93 | 5.92 s |
| outputs/capacity_baseline_rerun.json (no-op control, loaded box) | 2004.45 | 45.83 | 58.33 | 59.85 | 1124.17 | 651.02 | 10.97 s |
| **outputs/val_capacity_pass.json (pass on, quiet box)** | **2042.60** | 56.25 | 58.33 | 63.71 | 1140.83 | 656.36 | **6.62 s** |

Runtime with the pass: 6.62 s/scenario mean, 10.02 s max — inside the 8 s
budget; the pass itself accounts for ~0.09 s of it. The spread between the
three totals is dominated by incumbent re-rolls, not the pass (see paired
number above).

## Why the -30 target was unreachable (measured)

1. **Emergency-side penalties.** A large share of the validation's
   daily/weekly component is post-horizon emergency days from missed
   forecasts. s_4: 300 daily + 100 weekly in validation, only 100 daily on
   the plan replay. No plan repair can touch the rest.
2. **Irreducible far-cluster mega-days.** The historical worst days (s_4
   28.1 h, s_21 35.6 h/13 batteries/8 buildings) have every building
   7.3-8.1 h from base. Any split buys a second ~16 h-travel day that
   inherits the first day's return carry (evaluate.py carries return travel
   into the next workday) and trips its own daily limit and the weekly
   bucket. `tools/capacity_pass_probe.py` enumerates every candidate on
   these incumbents: best available delta +3.9 (s_4), +7.6 (s_21). The
   mega-day is the optimum; the local search had already converged, not run
   out of budget.
3. **Fixable hits are rare.** On a given incumbent draw, ~2 weekly hits per
   48 scenarios are profitably repairable (-100 each), plus a few
   op-saving merges of a couple hours each; that arithmetic is the -2.5.

## Tests

66/66 (`./.venv/Scripts/python.exe -m unittest discover -s tests`): the
original 62 plus 4 new in tests/test_capacity_pass.py — a fabricated 25.5 h
chained day that must split (and must pick the split that dodges the weekly
bucket and the return carry: exact target day asserted), a 25.3 h week bucket
that sheds its earliest day +7 via the documented tie-break, an
adjacent-light-day merge onto the earlier day, and bitwise determinism plus
battery/planned-set preservation.

## Artifacts

- batteryswap_solution/planner.py — `_capacity_repair` + `capacity_repair`
  flag + call in `plan()`
- tests/test_capacity_pass.py — 4 deterministic fixtures
- tools/capacity_pass_probe.py — single-scenario anatomy + candidate deltas
- tools/capacity_pass_paired.py — drift-free paired A/B (the number above)
- tools/capacity_pass_ab.py — no-op control runner (drift measurement)
- tools/capacity_pass_report.py — component diff of two validation JSONs
- outputs/val_capacity_pass.json, outputs/capacity_paired_ab.json,
  outputs/capacity_baseline_rerun.json, outputs/capacity_smoke.json
