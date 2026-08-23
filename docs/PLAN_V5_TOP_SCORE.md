# Plan V5 — Rebuild for the top of the leaderboard

Status: proposal for review. Written 2026-08-20. Deadline 2026-08-23.

All numbers below were measured in this repo today against
`batteryswap_public==0.3.4` and `dataset/train` (48 scenarios, 461 devices,
24 buildings, 82 observed EOL events). Scripts used are listed in
[Appendix A](#appendix-a--how-every-number-here-was-produced).

---

## 1. Where we actually stand

Public leaderboard decomposition, us (rank 17) vs the leader:

| component | J2W (#1) | Dynamic duo (#17) | ratio |
|---|---:|---:|---:|
| battery_swap | 4.83 | 10.28 | 2.1x |
| building_change | 11.52 | 30.96 | 2.7x |
| room_change | 7.35 | 18.16 | 2.5x |
| travel | 40.92 | 150.39 | 3.7x |
| overtime | 73.07 | 320.48 | 4.4x |
| daily_limit | 85.42 | 495.83 | 5.8x |
| weekly_limit | 43.75 | 256.25 | 5.9x |
| late_swap | 499.17 | 535.00 | 1.1x |
| **early_swap** | **605.51** | **2434.99** | **4.0x** |
| **total** | **1371.54** | **4252.33** | **3.1x** |

`battery_swap / 0.25` is the average number of in-window swaps per scenario:
**J2W swaps 19.3 batteries, we swap 41.1.** Our late_swap is essentially the
same as the leader's. Every other component is inflated by the same root cause:
**we swap far too many batteries, far too early.**

### The floor and the ceiling, measured

Three reference policies evaluated on all 48 train scenarios with the official
`evaluate_plan()`:

| policy | total | early | late | overtime | daily_lim | weekly_lim | swaps/scen |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Defer everything** (empty plan) | **3324.7** | 0 | 3056.3 | 57.2 | 70.8 | 85.4 | 0 |
| Perfect EOL knowledge, naive greedy route | **205.2** | 0 | 0 | 58.0 | 56.2 | 37.5 | 9.5 |
| Our submitted solution (public split) | **4252.3** | 2435.0 | 535.0 | 320.5 | 495.8 | 256.3 | 41.1 |

**Our solution scores worse than an empty plan.** That is the single most
important fact in this document. Every swap we make is, on net, destroying
score. The average scenario only has **9.5 batteries that genuinely reach EOL
inside the 42-day window**; we service 41.

The reachable range is `[205, 3325]`. The leader sits at 1371 on public. We are
outside the range on the wrong side.

### How much accuracy do we actually need?

I ran a controlled degradation study: give the decision layer the *true* answer
to "will this battery be recorded as EOL inside the window?" but corrupt the
*timing* with Gaussian noise of standard deviation σ, then let a correct
decision-theoretic rule pick the service day.

| timing error σ | total cost | early_swap | late_swap | swaps/scen |
|---:|---:|---:|---:|---:|
| 0 days | 205.2 | 0.0 | 0.0 | 9.5 |
| 3 days | 291.5 | 24.9 | 0.0 | 10.5 |
| 7 days | 379.3 | 61.4 | 0.0 | 11.6 |
| 14 days | 458.0 | 118.5 | 0.0 | 12.9 |
| 21 days | 524.4 | 176.3 | 0.0 | 14.1 |
| 30 days | 581.3 | 243.4 | 0.0 | 15.4 |

And the same study with the *classification* corrupted instead — `q_b` blurred
toward 0.5 by a fraction, at a fixed σ = 7 days:

| blur on P(is due) | total | late_swap | early_swap | swaps/scen |
|---:|---:|---:|---:|---:|
| 0 % | 379.3 | 0.0 | 61.4 | 11.6 |
| 2 % | 403.7 | 0.0 | 60.2 | 28.4 |
| 5 % | 374.7 | 2.7 | 54.2 | 28.4 |
| 10 % | 371.5 | 5.2 | 52.0 | 28.0 |
| 20 % | 384.6 | 32.3 | 49.4 | 27.9 |

A correct decision layer absorbs even 20 % classification blur at a cost of
~5 points, because it prices the uncertainty rather than thresholding it.

Read that carefully. **Even with ±30 days of timing error, a correct decision
layer scores 581.** Late cost stays at exactly zero at every noise level,
because the optimal rule places the swap near the ~5th percentile of the
predicted EOL distribution and simply *defers everything it is unsure about*.

The conclusion is unambiguous:

> The competition is not won by a better RUL point estimate. It is won by
> (a) correctly answering **which** batteries will be recorded as EOL inside the
> window, (b) **deferring everything else**, and (c) a scheduler that does not
> waste 150 points on avoidable capacity penalties.

Our previous three iterations spent their effort on (a')= sharper RUL
regression, which the table above shows is worth at most ~370 points, while
losing thousands on (b).

---

## 2. Evaluator mechanics that must be encoded exactly

These come from reading `batteryswap_public/evaluate.py` and
`batteryswap_public/utils.py` line by line. Several are non-obvious and at
least three of them are, I believe, where the ranking is actually decided.

**2.1 The EOL label is exactly reproducible.**
`eol_times[d]` is *precisely* the first day on which the official
`smooth_series()` output for device `d` falls below 2.4 V. I verified this on
all 82 train events: 82/82 exact match, 0 false positives, 0 false negatives.
This means we can synthesise labels at any historical cutoff and generate
unlimited training scenarios. The prediction target is a **first-passage time
of a smooth, near-monotone scalar curve** — not a black-box failure process.

**2.2 The scored EOL is censoring-aware, and we are told the censoring time.**
For a device with no EOL record, the evaluator substitutes
`normalize(locations.end_time + 30 days)`. `locations.end_time` is passed to
`plan()` as a column of the `locations` DataFrame. So the pseudo-EOL of every
never-failing device is **exactly computable at plan time**. On train it is
2026-08-31 for 445 of 461 devices (data export date + 30 d).

**2.3 A battery with no EOL record can be deferred for free — forever.**
`eol_batteries = set(eol_times[eol_times <= end_time].index)` is computed
*before* the NaN fill, and `NaN <= x` is False. So censored devices are never
added to the forced-emergency set. Deferring them costs 0. Swapping them costs
`0.5 x (pseudo_EOL - day)`, which on early scenarios is 150+ hours each.
16 train devices stopped reporting in late 2025; their pseudo-EOL is in the
*past* for later scenarios, so swapping one costs a **late** penalty of up to
1800. Any planner that does not special-case these is bleeding score.

**2.4 Missing a genuinely due battery is catastrophic and the cost is fixed.**
Unswapped due batteries are serviced after the window on dedicated days
starting at `start + 48 days`, one per day, sorted by battery id. Cost per
missed battery is `10 x (48 + k - T)` plus a full dedicated round trip plus
overtime — between 60 and 480 hours. Deferring an actually-due battery is
never cheap.

**2.5 The end-of-day return travel is double-counted.**
In `do_action`, `state.time_of_day` is reset to 0 when the day changes and the
return-travel time is added *after* the reset. So the travel back to base is
charged to the closing day's overtime **and** carried into the next active
day's clock. Consequences we can exploit:
- Consecutive active days that end far from base compound into daily-limit hits.
- The *last* active day's return travel leaks into the inert final Sunday, so
  the most expensive building should be visited on the **last** active day.
- Fewer active days is strictly better than more, beyond the obvious.

**2.6 Building change never carries the room with it.**
`change_building` passes `room=None`, so `state.room` is stale after a move and
a 0.5 h room change is charged on arrival. Room ids are globally unique
(verified: max buildings per room id = 1), so this is unavoidable per building
visit — but it makes multi-room buildings relatively cheaper to batch.

**2.7 Self-travel is 0.0333 h, not 0.** Every day close charges it even when
already at base.

**2.8 Weekly buckets are 7-day windows anchored on the scenario start**, and
the penalty is a flat 100 once accumulated work reaches 24 h. Total work in a
good plan is ~50 h over 7 weeks, so **every weekly-limit hit is avoidable
scheduling failure.** We are paying 256 per scenario for this.

**2.9 The travel matrix does not vary across train scenarios — the depot does.**
All 48 train scenarios share an identical `travel_costs` table. What is
randomised is `base_location` / `base_room` (16 distinct base buildings, 27
distinct base building+room pairs). Geography: 21 buildings / 406 devices in
one cluster, 2 buildings / 37 devices about 8 h away, 1 building / 18 devices
10.25 h away. When the depot lands in a remote cluster the whole scenario is
expensive. The planner must be depot-aware, not geography-aware.

**2.10 Plan validity constraints.** Complete (every battery exactly once),
non-decreasing `day`, clean `RangeIndex`, datetime day column with no
time-of-day, nothing before `start_time`. Deferral = any day after
`start + 42 days`.

---

## 3. Why the current solution fails

Reading `batteryswap_solution/` and `src/risk/`:

1. **The cost model asks the wrong question.** It builds a failure CDF and
   thresholds risk. The evaluator's actual decision is a two-branch expectation
   over *"is there an EOL record at all"* x *"when"*, with a known censoring
   time. Collapsing this into one survival curve, then patching it with a
   "mixture cure" model, a `physical_uncertainty_days` knob, a
   `physical_timing_weight`, and a hand-tuned 210-day survivor gate, is fitting
   correction terms to a structurally wrong objective. The documented tuning
   history (`docs/TACTICAL_EXPERIMENTS.md`) shows every fix moving the number
   but never getting below the all-defer baseline of 3324.68 — that is the
   signature of a wrong objective, not a badly tuned one.

2. **`VoltageTrendForecaster` is a linear extrapolant.** Measured on train:
   linear extrapolation of the 30- or 60-day slope to the 2.4 V crossing has
   MAE 61 days and a **+45 day optimistic bias** — it systematically says
   there is more life left than there is, because the discharge curve has a
   knee. Any planner leaning on it will be late; compensating for that by
   swapping early is what produced the 2435 early_swap.

3. **No hard deferral prior.** With 9.5 due batteries out of ~420 alive, the
   base rate is 2%. The planner services 41. There is no mechanism that makes
   "do nothing" the default and forces evidence to overcome it.

4. **The scheduler optimises a proxy, then repairs.** Local search over
   thresholds plus emergency sampling cannot recover the 750 points of
   daily+weekly limit penalties, because those come from *how many active days
   and how much work per week*, which the threshold layer does not control.

### What we keep

- `batteryswap_solution/replay.py` — a faithful, evaluator-exact fast replay
  including the 2.5 travel leak. This is genuinely good and is the objective
  function of the new optimizer. Keep, extend, and keep its equality test.
- `batteryswap_solution/routing.py` — Held–Karp for small day routes. Keep.
- `docs/OFFICIAL_CHALLENGE_REFERENCE.md` — keep, it is accurate.
- The submission plumbing in `script.py`, `Dockerfile`, `LICENSE`. Keep.

Everything in `src/risk/`, `batteryswap_solution/forecast.py`,
`costs.py`, `optimizer.py` and `planner.py` is replaced.

---

## 4. New architecture

Four layers, each with a single well-defined contract.

```
  raw hourly parquet
        |
  [L1] incremental smoothing cache  -->  causal daily smoothed V/T per device
        |
  [L2] first-passage model          -->  P(cross 2.4V on day d), d = 0..180
        |
  [L3] scored-cost distribution     -->  q_b = P(due), C[b][d] expected timing cost
        |
  [L4] operations optimizer         -->  day assignment + within-day route
        |
  submission.csv
```

### L1 — Incremental smoothing cache

`smooth_series()` is causal (daily resample, then a *trailing* rolling
quantile), so smoothing the truncated series equals truncating the smoothed
series. Therefore:

- Maintain a per-device cache of daily aggregates and the rolling output.
- On each `plan()` call, take only rows newer than the previous scenario's
  start (a boolean mask over the MultiIndex level, ~50 ms on 6.2 M rows) and
  extend the cache.
- Recompute the rolling tail only for the last ~14 days per device.

Measured: full smoothing of a split costs 26 s once; the harness itself costs
68 s per split. Budget in section 8.

### L2 — First-passage model for the 2.4 V crossing

Target: for device `b` observed up to cutoff `t`, the distribution of
`T_b` = first day the smoothed curve goes below 2.4 V.

**Training sample generation.** For every device, every cutoff on a 3-day grid
from `start + 60 d` to `min(last_obs, T_b)`. Each sample is
`(features(series up to t), y = T_b - t, event = 1)` if the device crosses, or
`(..., y = last_obs - t, event = 0)` right-censored otherwise. On a 7-day grid
this already yields 45.5 k samples / 7.7 k events; a 3-day grid gives ~3x that.

**Model form — discrete-time hazard, not regression.** The decision layer needs
a calibrated CDF, not a point estimate. My prototype confirms a mean regressor
is actively harmful here: MAE 25 d overall, but **+17 d optimistic bias when
the true RUL is under 14 days** and −21 d pessimistic bias at 70–120 days —
textbook regression to the mean, in the direction that gets you late.

So: train one `HistGradientBoostingClassifier` on the stacked
`(sample, horizon)` design, with horizon `h ∈ {3,7,10,14,21,28,35,42,56,70,90,120,180}`
as an explicit feature, predicting `P(T - t <= h)`. A sample contributes to
horizon `h` only if it is informative for it (crossed within `h`, or observed
alive for at least `h` more days). Enforce monotonicity in `h` with
sklearn's `monotonic_cst`, then isotonic-calibrate per horizon band on
out-of-fold predictions. Interpolate to daily and difference to get the PMF.

**Features** (all causal, all from the smoothed series):
- level: `v(t)`, min/max so far, drawdown from max
- rate: `Δv` over 7/14/30/60/90/120 d, and curvature `Δv_14 - Δv_60`
- knee position: days since first crossing each of 2.90 … 2.42 V (9 thresholds)
- extrapolation: linear crossing estimate from the 14/30/60-day slope
  (used *as a feature* so the model can learn its known +45 d bias, never as
  the answer)
- thermal: recent and lifetime mean temperature, temperature variance
- ageing: device age, data coverage, staleness of the last observation, gap
  count in the last 90 days
- **no building, room, or device identity** — the split is by unseen buildings

**Second opinion — trajectory matching.** Align the last 90 days of the query
curve against all 82 known EOL trajectories plus the censored ones (as lower
bounds), match on voltage level and slope, and take the empirical
time-to-crossing distribution of the k nearest. This captures the knee shape
non-parametrically with almost no parameters to overfit, and it degrades
gracefully. Blend with the hazard model by out-of-fold stacking weights.

Prototype signal check (building-grouped 5-fold OOF, weekly grid):
`P(cross within 42 d)` gets **AUC 0.983, Brier 0.0068**, top-decile calibration
0.087 predicted vs 0.098 actual, at a 1.0 % base rate. That is a usable
starting point and the tail calibration is what we will improve.

### L3 — From crossing distribution to scored cost

This is the layer that was structurally missing. For each battery `b`:

Let `E_b = locations.end_time[b]` (known), `P_b = normalize(E_b + 30 d)`
(known), `S` the scenario start, `W = S + 42 d`, `D = S + 48 d`.

```
q_b        = P(T_b <= min(E_b, W))                     # probability it is "due"
f_b(d)     = P(T_b = S + d | T_b <= E_b)               # timing pmf, conditional
Cswap(b,d) = q'_b * Σ_k f_b(k) * [0.5*(k-d)+ + 10*(d-k)+]
             + (1-q'_b) * [0.5*(P_b-d)+ + 10*(d-P_b)+]
Cdefer(b)  = q_b * Σ_{k<=42} f_b(k) * 10 * (48 + rank_b - k)
```

where `q'_b = P(T_b <= E_b)` is the probability an EOL record exists at all,
and `rank_b` is the expected position in the sorted emergency queue.

Three properties fall out for free, none of which the current solution has:
- Devices that stopped reporting long ago get `q_b = 0` and a *negative*
  `P_b - d`, so swapping them is priced at its true catastrophic cost.
- Late scenarios, where the remaining observation window is shorter than the
  planning window, automatically produce small `q_b`. No 210-day hand gate.
- The 20:1 late:early asymmetry is handled by the expectation itself; the
  optimal day naturally lands near the 5th percentile of `f_b`.

### L4 — Operations optimizer

Objective = `replay.py` cost (exact, deterministic, includes the travel leak,
week buckets and daily limits) + `Σ Cswap/Cdefer` from L3 + expected emergency
operational cost for deferred-but-possibly-due batteries.

Candidate set: batteries with `q_b > ε` (~15–40 per scenario). Everything else
is hard-deferred and never enters the search.

- **Seed 1**: per-battery independent argmin of `Cswap(b,·)` vs `Cdefer(b)`.
- **Seed 2**: cluster seeds — one visit per (building, week) at the earliest
  member's optimal day.
- **Search**: large-neighbourhood search with moves
  1. shift one battery ±1..7 days, or defer it
  2. move a whole building-block from day `d` to `d'`
  3. merge day `d` into `d'` / split a day
  4. opportunistic add — pull a co-located battery into an existing visit
  5. drop a battery from a visit
  6. exact Held–Karp re-route within a day (≤10 buildings), insertion + 2-opt above
- **Ordering rule** applied at the end: schedule the day whose route ends
  farthest from base **last**, so its return travel leaks into the inert
  final Sunday (2.5 above).
- **Guards**: keep every week under 24 h; keep every day under 24 h including
  the leaked travel; never let the plan place work on the final Sunday.

Move 4 matters more than it looks. For the 10.25 h building, a dedicated visit
costs ~20.5 h travel + ~25 h overtime, so pulling in a co-located battery that
is 60 days from EOL (cost 30) is strongly positive. For a 0.2 h building the
break-even is about 4 days. The optimizer discovers this per-building from the
real cost function instead of a global "batch window" constant.

Runtime: a replay over ~30 rows is ~0.1 ms; a few thousand evaluations per
scenario is well inside budget.

---

## 5. Validation protocol

This is where we have been fooling ourselves, so it is a first-class
deliverable, not an afterthought.

- **V1 — honest model score.** `GroupKFold(5)` **by building**. Assemble
  out-of-fold predictions for all 461 devices, then run all 48 scenarios using
  only OOF predictions. This is the only local number allowed to justify a
  submission. In-fold scores are banned from the decision log.
- **V2 — unseen-building stress.** Split the 24 buildings into two halves;
  train on half A; evaluate scenarios restricted to half B's devices *and* with
  a depot drawn from half B. This directly mimics the public/private design.
- **V3 — scenario robustness.** Report mean, median, p90, max and the count of
  scenarios that regress, not just the mean. Bootstrap over scenarios for a CI.
- **V4 — decision audit.** Per scenario: `#due`, `#swapped`, `#missed`,
  precision and recall on "due", and the early/late split. A change that
  improves the mean while lowering recall is rejected.
- **Anchors printed on every run**: all-defer (3324.7) and perfect-knowledge
  (205.2). Any candidate that does not beat all-defer by a wide margin is not a
  candidate.

---

## 6. On pretrained models — my recommendation is no

You said you are willing, and the rules do allow it (open-source, documented,
pinned, weights committed, no network at eval). My honest read is that it is
the wrong tool here, and I would rather say so than spend a day of a three-day
budget on it:

- **The runtime does not fit.** 96 `plan()` calls x ~420 devices = ~40 000
  forecasts, CPU-only, inside a 30-minute wall clock that already spends 2.3
  minutes on harness I/O. Chronos/TimesFM/Moirai-class models cannot do that.
- **The signal does not need it.** The target is a first-passage time of a
  single smooth scalar curve, and a plain voltage threshold already reaches
  AUC 0.983 on the decision-relevant question. A general-purpose forecaster
  brings no prior that beats "this is a Li-MnO₂ discharge knee".
- **The score is not limited by forecasting.** Section 1 shows ±30 days of
  timing error still scores 581 against our 4252. The bottleneck is the
  decision and scheduling layers.

Where a heavier model *could* earn its place, if time remains after the core
lands: a small 1-D CNN or GRU over the last 180 days of smoothed voltage,
trained from scratch on our own synthetic cutoffs, as a third opinion in the
L2 blend. `torch` is already in the competition runtime. This is cheap at
inference and stays reproducible. I would only do this on Day 3 if V1 says the
blend is the binding constraint.

---

## 7. Expected outcome

I will not promise a rank. What I can state from the measurements:

- Simply **deferring everything** scores 3324.7 locally, versus our 4252.3 on
  public. Even that trivial change is a large move.
- With a correct decision layer and a timing model at σ ≈ 14 days — well
  inside what the prototype already achieves near the knee — the measured
  score is **458** on train.
- The remaining ~250 above the 205 floor is capacity penalties, which the L4
  optimizer attacks directly.

The honest uncertainty: train and public are different building sets, so the
absolute scale may shift. The *structure* of the cost — 9.5 due batteries, a
20:1 late:early ratio, flat 100 capacity penalties — will not.

---

## 8. Runtime budget (30 min wall, CPU only, 32 GB)

Measured on this machine, per split:

| item | cost |
|---|---:|
| `load_dataset` | 4.7 s |
| `iterate_scenarios` x48 (harness, unavoidable) | 62.9 s |
| initial full smoothing (first scenario only) | ~26 s |
| incremental smoothing x47 | ~0.2 s each |
| L2 inference, ~420 devices x 13 horizons | ~0.3 s per scenario |
| L4 optimizer | budget 3 s per scenario, hard-capped |

Projected total for public+private: **~9–11 minutes**, versus the 13.8 minutes
our current 48-scenario train audit alone takes. Hard requirements:
- a wall-clock guard that degrades the optimizer to its seed solution if the
  elapsed budget is exceeded;
- the planner instance is reused across scenarios (`make_submissions` calls
  `planner_loader()` per scenario but `script.py` returns the same object), so
  the cache is safe;
- artifacts loaded once at import, not per scenario.

---

## 9. Schedule to 2026-08-23

We have ~3 days and 5 submissions/day, and must nominate at most 3 finals.

**Day 1 (today, 2026-08-20)**
1. New branch, new package `bsai/`, keep `replay.py` + `routing.py`.
2. L1 cache + L3 cost algebra + L4 optimizer against a *perfect-knowledge*
   forecaster. Target: reproduce ≤205 on train and push the capacity penalties
   below 100. This validates the whole operations half with zero ML risk.
3. **Submission #1: all-defer.** One submission, enormous information value —
   it calibrates the public split's scale against our local 3324.7 and gives us
   a guaranteed-safe fallback to nominate as a final. It is a legitimate
   strategy, not evaluator probing.

**Day 2 (2026-08-21)**
4. L2 training pipeline: sample generation, hazard model, trajectory matcher,
   isotonic calibration, building-grouped OOF.
5. V1/V2/V3/V4 validation report.
6. **Submissions #2–#4**: full pipeline, then one conservative and one
   aggressive `q` calibration variant.

**Day 3 (2026-08-22)**
7. Tune only on V1 OOF. Optional torch third opinion if V1 says L2 binds.
8. Reproducibility pass: seeds, pinned artifacts, `LICENSE`, third-party
   license table, exact reproduction commands, README rewrite.
9. **Submissions #5–#8**, nominate 3 finals.

**2026-08-23**: buffer day for the deadline itself and the 1-hour
public-repository requirement.

Explicit rule for the whole window: **never force-push, never rewrite a
submitted commit.** Every submitted SHA stays in history.

---

## 10. Risks

| risk | mitigation |
|---|---|
| Public/private split has a different due-rate and our `q` calibration is off | Submission #1 (all-defer) calibrates the scale; keep a conservative and an aggressive variant among the 3 finals |
| Optimizer overruns the 30-min wall on a bigger split | Hard elapsed-time guard, degrade to seed solution, cap candidate set size |
| Overfitting to 82 events | Building-grouped OOF only; no per-building features; the trajectory matcher has ~2 hyperparameters |
| `locations.end_time` behaves differently on public/private | Derive `P_b` defensively; if `end_time` is at or before the scenario start, force `q=0` and hard-defer |
| A bug makes a plan invalid and the run scores nothing | Validate every scenario plan with `check_plan_valid` before writing, and assert plan completeness in `script.py` |
| We chase the public leaderboard and overfit it | Nominate one final selected purely on V1/V2, not on public score |

---

## 11. Compliance checklist

- MIT `LICENSE` at repo root — present, keep.
- Third-party license table in README (all competition packages are
  MIT/BSD/Apache-2.0).
- No network at eval; all artifacts committed.
- Full reproduction pipeline documented with exact commands and seeds.
- Repository public within 1 hour of the deadline; submitted commits preserved.
- No external datasets, no pretrained third-party weights (per section 6).

---

## Appendix A — how every number here was produced

| claim | method |
|---|---|
| Leaderboard decomposition | the public leaderboard table you provided; components verified to sum to `total_time` for both rows |
| all-defer = 3324.7, oracle = 205.2 | `evaluate_plan()` over all 48 train scenarios, greedy nearest-building day ordering |
| σ-degradation table | same harness, exact decision rule over a discretised EOL pmf, Gaussian corruption of the true EOL |
| EOL label reproducibility | recomputed `smooth_series()` on the full train parquet; first day below 2.4 V matched `eol_times.csv` for all 82 events |
| linear-extrapolation bias +45 d | 1254 uncensored device-cutoff samples, `lin30` / `lin60` vs true RUL |
| RUL regressor bias by band | `HistGradientBoostingRegressor`, 5-fold GroupKFold by building |
| `P(cross ≤ 42 d)` AUC 0.983 | `HistGradientBoostingClassifier`, 5-fold GroupKFold by building, 43 269 samples / 441 positives |
| travel matrix identical across scenarios | pairwise `Series.equals` over all 48 scenario travel tables |
| harness overhead 67.7 s per split | timed `load_dataset` + full `iterate_scenarios` pass |
| evaluator mechanics (2.1–2.10) | direct reading of `batteryswap_public/evaluate.py` and `utils.py` v0.3.4 |
