# V6 implementation notes

What was built, what it measures, and what did not work. Written 2026-08-21
against `claude/v6-forecast-rebuild`.

Every number here is out-of-fold by building, scored with the official
`evaluate_plan()` over all 48 train scenarios, unless stated otherwise.

Anchors: all-defer **3324.7**, naive oracle **205.2**, the existing planner with
oracle risk **77.8** (scenarios 0-11 only).

---

## 1. Where the branch landed

| configuration | mean | late | early | capacity | served | recall |
|---|---:|---:|---:|---:|---:|---:|
| all-defer | 3324.7 | 3056.3 | 0.0 | 213.4 | 0 | 0 |
| shipped v3 (local) | 2644.9 | 1700.0 | 610.8 | — | 11.0 | 0.296 |
| **V6, joint search (default)** | **2526.0** | 1407.5 | 747.6 | 284.8 | 17.2 | 0.449 |
| V6, per-battery decision | 2668.4 | 1366.9 | 764.8 | 429.9 | 19.1 | 0.465 |
| V5 prototype (scratch, weaker model + naive router) | 1567.6 | 911.7 | 272.4 | 297.7 | 23.8 | 0.659 |

`capacity` is overtime + daily_limit + weekly_limit.

**The branch beats the shipped solution locally but does not beat the V5
prototype.** That is stated plainly because it decides what should be submitted
and what should not.

Runtime: **5.65 s per scenario**, projecting to **11.3 minutes** for the 96
public and private scenarios including 2.3 minutes of harness overhead. The
previous solution projected 25.8-27.6 minutes.

## 2. The defect that mattered, and why local totals hide it

The shipped v3 scored 2644.9 locally and **4252.3 on public** — a 1.6x
degradation with `early_swap` 2434.99 and roughly 41 swaps per scenario. That
profile is the signature of one specific failure, and it is now measured.

Predicted against actual due batteries, out-of-fold, by scenario regime:

| scenarios | predicted | actual | ratio | served |
|---|---:|---:|---:|---:|
| early (s_0-15) | 11.56 | 13.25 | 0.87 | 16.8 |
| mid (s_16-31) | 12.53 | 8.56 | 1.46 | 21.1 |
| late (s_32-47) | 14.49 | 6.56 | **2.21** | 40.6 |

The over-prediction grows monotonically as the observation window closes. In
`s_41` the model predicted 19.3 due, 6 were, and the planner serviced 52.

**Cause.** The scored target is not "will this battery cross 2.4 V" but "will an
EOL *record* exist", and a record can only be filed while the device is still
observed. `stack_horizons` treated a row with fewer than `h` days of remaining
observation as censored and dropped it. Those rows are not censored — they are
genuine negatives. Dropping them removed exactly the population that dominates
the closing scenarios.

**Fix.** No censoring drops, and `remaining_observation_days` becomes a feature.
It is known at plan time from `locations.end_time`, it is monotone-constrained
(more observation time can only add records), and without it the model cannot
express "no time left for a record to be filed".

Reliability at scenario cutoffs, before and after:

| predicted p | n | actual | ratio before | ratio after |
|---|---:|---:|---:|---:|
| 0.05-0.10 | 600 | 0.078 | 1.31 | 0.81 |
| 0.10-0.20 | 604 | 0.146 | 1.33 | **0.99** |
| 0.20-0.35 | 412 | 0.231 | 1.27 | **1.12** |
| 0.35-0.50 | 185 | 0.287 | 1.57 | 1.46 |
| 0.70-1.00 | 96 | 0.427 | 2.23 | 2.08 |
| **overall** | | | **1.36** | **1.07** |

Predicted due per scenario fell from 12.86 to 10.14 against an actual 9.46, and
servicing from 26.2 to 17.2. **The local total barely moved** (2559.7 -> 2526.0)
because recall fell as calibration improved. Local validation cannot score the
difference; only a submission can.

## 3. The remaining gap is the objective, not the search

`_expected_score` is what the local search optimises. On the first twelve
scenarios:

```
mean believed   1176.8
mean realised   2328.2
mean gap       +1151.3
correlation        0.613
```

The objective is roughly twice as optimistic as the evaluator, and only loosely
correlated with it. A search that is good at optimising a number this wrong will
confidently assemble plans whose believed savings do not exist. **This is the
single highest-value thing left to fix**, and it explains why so many knobs
below moved the plan without moving the score.

## 4. Fixes that are in

- **The weekly limit was a hard constraint.** The evaluator charges a flat 100
  for breaching it; a missed due battery costs 200-400. One visit to the 10.25 h
  building consumes 20.5 h of a 24 h week, so the solver could not schedule.
  Both limits are now priced in the objective with a loose hard bound.
- **Candidate reduction.** Every search evaluation walked all ~420 batteries when
  about 9.5 are ever due. Now capped at 150 by expected gain. With the above:
  **17.3 s -> 5-6 s per scenario**, and 1449 -> 1290 on the first four scenarios.
- **The emergency-queue rank** was summed over the whole fleet (~6.4) rather than
  the batteries actually missed (~3.6), inflating the cost of deferring by
  10 hours per day of rank. Now an explicit `emergency_rank_scale`.
- **Cold-start gate.** At a 21-day staleness cut, 15.9 % of alive
  device-scenarios had no usable row and were auto-deferred, carrying **11 % of
  all due batteries**. Their due rate (0.012-0.019) is close to fresh devices'
  (0.024), so the gate was removed and staleness handed to the model.
- **`Dockerfile` did not copy `bsai/`.** The joblib artifact resolves
  `bsai.hazard.HazardModel` on load, so the submission would have silently
  fallen back to the voltage-trend forecaster.
- **Incremental smoothing**, pinned bit-exact to the official `smooth_series`
  across single-pass, incremental and truncated-prefix paths, and **10x faster**
  on a full split (6 s against 26 s).
- **A fallback guard in validation.** `CompetitionPlanner.plan` catches
  everything and returns all-defer, which is right in a submission and terrible
  in validation: a crash reads as a mediocre score. A renamed parameter once
  turned every scenario into all-defer and the run still reported a number.
  `tools/validate_v6.py` now aborts if any scenario falls back.

## 5. What was tried and did not work

Recorded so the next person does not repeat them.

| tried | result |
|---|---|
| CP-SAT seed vs per-battery seed | **Identical plans** in every cost component. The local search dominates the construction. |
| Refitting calibration on the scenario-cutoff population | Ratio 1.36 -> 1.40. No effect: the shift is between buildings, and a calibrator fitted on the other four folds cannot see it. |
| Global probability shrink (0.70 / 0.45 / 0.25) | Destroys the early scenarios: `s_0` 1250.9 -> 1872.4 at 0.70. Early and late regimes want opposite corrections. |
| `emergency_rank_scale` 0.0 / 0.35 | Same trade in reverse: `s_0` 1250.9 -> 1872.4 / 1607.7. |
| Candidate margin 24 -> 6 -> 0 | 2526.0 -> 2522.1 -> 2506.8. Inside the noise floor. |
| `capacity_roundtrip_fraction` 1.0 -> 0.55 | 2526.0 -> 2525.9. The replay-based repair already handles it. |
| Per-battery decision layer (no joint search) | 2668.4, and capacity cost 429.9 against 284.8. The joint search earns its place. |
| `max_planned_rate` cap | Operates on the *reduced candidate* count, so 0.05 collapses to 1-2 swaps. Needs an absolute cap to be meaningful. |

The pattern in the first six rows: every global knob trades the early regime
against the late one, because the two were miscalibrated in opposite directions.
Fixing the label removed that tension; none of the knobs are needed.

## 6. What to do next, in order

1. **Fix `_expected_score`** (section 3). Everything the planner does follows
   from it, and it is twice as optimistic as reality. The sample-average
   approximation in `PLAN_V6_MAXIMUM.md` section 4.1 is the intended
   replacement: the in-window operational cost of a fixed plan is deterministic,
   the timing penalty is separable and exact, and only the emergency tail needs
   sampling — which vectorises.
2. **Recover the discrimination the label fix cost.** Horizon-42 PR-AUC fell
   0.4932 -> 0.4052 while calibration improved. The seasonal-physical head
   (`PLAN_V6_MAXIMUM.md` section 3.2) targets the residual misses, which are
   seasonally structured.
3. **Submit all-defer once.** Its public score is unknown, and without it every
   public number is uninterpretable.

## 7. Reproduction

```bash
python tools/train_v6.py          # ~18 min: model + fold models + report
python tools/validate_v6.py       # ~5 min: the only number that justifies a submission
python tools/calibration_v6.py    # ~2 min: reliability at scenario cutoffs
python tools/belief_v6.py         # believed vs realised cost
python -m unittest discover -s tests -v
```

Seeds are fixed in `bsai/hazard.py` (`random_state=20260821`) and
`batteryswap_solution/optimizer.py` (`random_seed=20260818`).
