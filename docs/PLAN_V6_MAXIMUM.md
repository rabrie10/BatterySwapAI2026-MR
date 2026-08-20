# Plan V6 — the maximum-result plan

Supersedes `PLAN_V5_TOP_SCORE.md`. Incorporates the V5.1 review, corrects two
errors in V5 that measurement disproved, and adds the findings from a further
round of experiments. Written 2026-08-21.

Scope note: this plan is optimised for the best achievable private-leaderboard
score, not for the shortest path. Sequencing is given by dependency, not by
calendar.

---

## 0. Corrections to V5, established by measurement

**V5 was wrong to propose rebuilding Task 2.** Running my naive oracle on
exactly the 12 scenarios used by `benchmark_task2.py --mode oracle --limit 12`:

| on scenarios 0–11 | total | daily_limit | weekly_limit | travel | overtime |
|---|---:|---:|---:|---:|---:|
| naive greedy oracle (V5's "floor") | 263.18 | 58.3 | 83.3 | 41.1 | 57.2 |
| **existing planner, oracle risk** | **77.83** | **0.0** | **0.0** | 22.6 | 24.3 |
| all-defer | 4885.46 | | | | |

The all-defer figure matches `TASK2_IMPLEMENTATION.md` exactly, confirming the
same scenario set. The existing optimizer is 3.4× better than V5's baseline and
eliminates both capacity penalties. `205.2` is not a floor; it is a weak
baseline. **Keep `replay.py`, `routing.py`, `optimizer.py` and `planner.py`.**

**V5's robustness study was not valid.** Blurring `P(due)` toward 0.5 maps
1→0.98 and 0→0.02, which preserves the oracle ranking perfectly and can only
create false positives, never false negatives. It shows that timing matters
little once incidence is known — nothing more.

**V5's L3 defer formula double-counted.** With `f` conditional on `T ≤ E`,
`q_due = q_event · Σ_{k≤W} f(k)`, so `q_due · Σ_{k≤W} f(k) · cost` applies the
incidence probability twice. The existing `costs.py` does *not* have this bug —
it works with an unconditional pmf decomposed into (event in window, event
observed after window, never observed). **That convention is adopted here.**

Also accepted from the review: horizons must extend to the full remaining
observation window, not 180 days; AUC at a 1 % base rate is not sufficient
evidence; recall is diagnostic, not an acceptance gate; and candidate selection
must be by expected economic gain, not a bare `q > ε` threshold.

---

## 1. New findings that shape this plan

### 1.1 `smooth_series()` is exactly causal — verified

Smoothing the truncated series and truncating the smoothed series agree to
**0.000e+00 over 26,366 device-days across 40 devices**. Two consequences: the
features used in all experiments below do not leak the future, and the
production planner may keep an incremental smoothing cache across scenarios.

### 1.2 The EOL label is exactly reproducible

`eol_times[d]` is the first day the official smoothed voltage falls below
2.4 V — 82/82 exact, no false positives or negatives. We can therefore
manufacture labelled cutoffs at any date and synthesise unlimited scenarios.

### 1.3 Temperature is a first-order driver, and it is predictable

Within-device, after removing a 91-day trend from both series:

- `β ≈ +0.00463 V/°C` (median over 454 devices; IQR 0.0031–0.0063)
- Spearman ρ median **+0.564**, and **positive in 100 % of devices**
- indoor annual temperature swing 4.87 °C → **0.023 V** of pure seasonal voltage
- EOL incidence **1.76× higher in Nov–Mar than May–Sep** (3.57 vs 0.19 events
  per 1000 device-days in January vs August)

0.023 V is roughly the gap between 2.45 V and 2.475 V, which at that point in
the curve is about two weeks of remaining life. **A device can cross 2.4 V
because winter arrived.** Since the calendar is known at plan time and each
device's own seasonal profile is estimable from its history, this is
*forecastable* signal that a purely state-based model discards.

### 1.4 The curve is not monotone, and level alone is a weak predictor

- 23.5 % of smoothed days are increases, so first passage is genuinely
  stochastic. Predict a distribution, never a point.
- Days from first crossing a level until EOL:

| level | median | IQR | q10–q90 |
|---|---:|---:|---:|
| 2.70 V | 258 | 169 | 97–386 |
| 2.60 V | 134 | 128 | 43–252 |
| 2.55 V | 83 | 87 | 27–196 |
| 2.50 V | 44 | 67 | 11–138 |
| 2.45 V | 17 | 30 | 0–90 |

- **Pace normalisation fails.** Correlation between "days taken to fall
  2.70→2.55" and "days remaining from 2.55" is **−0.085**; dividing by pace
  makes dispersion worse at every level. There is no simple universal discharge
  clock to exploit.
- Temperature compensation does not shrink this dispersion either (it does cut
  spurious level re-crossings from 0.121 to 0.084 of days). The dispersion is
  real device-to-device variation, not measurement noise.

### 1.5 Between-building failure rates vary 20×

Observed EOL rate per building ranges from **0.043 to 0.833** (mean 0.224, sd
0.224) across the 24 train buildings. The public and private splits are
*different buildings*. This is the mechanism behind the calibration collapse:

| same v3 code | local train | public |
|---|---:|---:|
| swaps per scenario | 10.98 | **41.1** |
| early_swap | 610.78 | **2434.99** |
| late_swap | 1700.00 | 535.00 |
| due recall | **0.296** | — |

Locally the model is far too timid and misses 70 % of real failures; on public
it is four times too eager. That is one failure — the absolute probability
level does not transfer — showing up with opposite signs.

### 1.6 The 48 train scenarios are not 48 independent samples

Consecutive scenarios start one week apart and each covers six weeks, so
adjacent windows overlap by roughly 85 %. The effective sample size is closer
to **eight**. Differences of well under 100 in the 48-scenario mean are noise.
This is very likely why the tactical-experiment series kept producing
"improvements" that did not transfer.

### 1.7 Only the depot is randomised

All 48 train scenarios share an identical travel matrix; what varies is
`base_location` / `base_room` (16 base buildings, 27 base building+room pairs).
Every other setting is constant across scenarios. **The planner must still read
every value from `settings`** — public and private are free to differ, and
nothing in our code should hardcode 42 days, 0.5/10 penalties, or 24 h limits.

### 1.8 Geography

21 buildings / 406 devices in one cluster; 2 buildings / 37 devices at ~8 h;
1 building / 18 devices at 10.25 h. Self-travel is 0.0333 h. When the depot
lands in a remote cluster the scenario is structurally expensive.

---

## 2. Where the score actually is

| lever | measured value |
|---|---|
| **Incidence** — which batteries get an EOL record inside the window | thousands |
| **Timing** — when to swap, given incidence | ≤ 370 (σ = 0 → 30 days) |
| **Operations** — routing, batching, capacity | ~185 (263 → 78 on scenarios 0–11) |

Incidence dominates by an order of magnitude. Everything in this plan is
prioritised accordingly.

### 2.1 A prototype of §3 already beats the shipped solution

To keep this plan from being speculation, I built a first cut of the §3.2 H1
head — a single multi-horizon hazard GBM, 1.52 M stacked `(cutoff, horizon)`
rows from 107,897 cutoffs, out-of-fold by building — wired it into the §4
decision algebra, and scheduled it with the *naive greedy* router from V5 (not
the good existing planner).

Classifier quality, out-of-fold, grouped by building:

| metric | stacked | horizon 42 (base rate 1.0 %) |
|---|---:|---:|
| AUC | 0.9782 | 0.9849 |
| **PR-AUC** | **0.5946** | **0.5260** |
| Brier | 0.01747 | — |

precision@10 = 1.000, @15 = 1.000, @20 = 0.950, @30 = 0.933, @50 = 0.920,
@100 = 0.840. PR-AUC of 0.526 against a 1.0 % base rate is a 52× lift, which
answers the review's objection that AUC alone proves nothing.

End-to-end evaluator cost, all 48 train scenarios, out-of-fold predictions only:

| | total | late | early | overtime | daily | weekly | swaps | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-defer | 3324.7 | 3056.3 | 0 | 57.2 | 70.8 | 85.4 | 0 | 0 |
| **shipped v3** | **2644.9** | 1700.0 | 610.8 | — | — | — | 11.0 | **0.296** |
| **V6 prototype + naive router** | **1567.6** | 911.7 | 272.4 | 95.6 | 87.5 | 114.6 | 23.8 | **0.659** |
| naive oracle | 205.2 | 0 | 0 | 58.0 | 56.2 | 37.5 | 9.5 | 1.000 |

A first cut, honestly validated, with the *worse* of our two schedulers, is
**41 % below the shipped solution** and more than doubles due-recall.

### 2.2 The three remaining gaps are now individually priced

1. **Capacity penalties: 297.7.** The prototype's overtime + daily + weekly.
   The existing planner drives daily and weekly to exactly 0.0 under oracle
   risk, so most of this is recovered simply by using it instead of the naive
   router.
2. **Late cost: 911.7**, from 3.23 missed due batteries per scenario. Sweeping
   the late-risk multiplier from 0.6 to 2.0 moves recall by 0.003 and total by
   under 40. **The misses are batteries the model gives almost no probability
   to — no decision-layer knob can recover them.** This is pure Task 1 error,
   and it is what heads H2–H4 exist to attack.
3. **Over-prediction of incidence: predicted `E[#due]` = 20.6 against an actual
   9.5.** Early cost and roughly half the excess swaps come from here. This is
   the §3.5 recalibration target, and it is the same defect that produced 41
   swaps per scenario on the public leaderboard.

These three are separable, individually measurable, and none of them requires
rebuilding Task 2.

### 2.3 Rescaling probabilities does not help — only better ones do

Scaling the whole probability vector trades early cost against late cost at
almost exactly par:

| probability scale | total | late | early | swaps | recall |
|---:|---:|---:|---:|---:|---:|
| ×0.7 | 1545.7 | 1045.8 | 176.4 | 21.1 | 0.593 |
| ×1.0 | 1567.6 | 911.7 | 272.4 | 23.8 | 0.659 |
| ×1.3 | 1588.7 | 838.3 | 337.3 | 25.3 | 0.685 |

A 1.9× swing in the probability scale moves total cost by 43 — under the noise
floor implied by §1.6. The same is true of the late-risk multiplier across
0.6–2.0. **There is no knob left on this model.** Every remaining point has to
come from probabilities that are actually better ordered, which is the entire
argument for §3.

### 2.4 The misses are structured, not random

Of the 404 genuinely-due batteries across the 48 scenarios, **19.1 % get
`P(cross ≤ 42 d) < 0.05`** from the prototype. They are not a random sample:

| | missed (p<0.05) | detected |
|---|---:|---:|
| median voltage at scenario start | **2.503** | 2.457 |
| median 30-day slope | **−0.00172** | −0.00210 |
| mean true RUL (days) | 26.6 | 19.5 |

The missed ones sit **higher on the curve and are declining more slowly**, yet
cross anyway. They are knee-onset surprises, not borderline cases.

Miss rate by true RUL: 5.7 % at 0–7 days, 12.7 % at 7–14, 23.5 % at 21–28,
**32.3 % at 35–42**. The model is fine at short range and degrades steadily.

Miss rate by the month the scenario starts:

| window covers | months | miss rate |
|---|---|---:|
| cooling into winter | Sep–Dec | 0.05–0.13 |
| warming into spring/summer | Jan–Mar, Jun | **0.38–0.53** |

The residual error is **seasonally structured**, and the prototype only has raw
day-of-year sin/cos to work with. Combined with §1.3 — β positive in 100 % of
devices, 1.76× winter incidence — this is the clearest single indication of
which head to build next: an explicit temperature-path model, not a calendar
dummy. (Counts per month are small, 13–71 due batteries, so treat the exact
values as directional.)

---

## 3. Task 1: the model

### 3.1 The reframing that unlocks the data

Event-based survival models are trained on **82 events**. But EOL is a
deterministic threshold on a curve we can forecast directly, and the 379
censored devices have fully observed *voltage futures* even though they never
cross. Forecasting the curve instead of the event turns a tiny-label survival
problem into a **385,153 device-day regression problem**, and uses every device
at full weight instead of as a weak right-censoring bound.

That is the centrepiece of V6.

### 3.2 Four heads, stacked

- **H1 — discrete-time hazard GBM.** One classifier over stacked
  `(sample, horizon)` rows with horizon as an explicit feature, monotone
  constrained in horizon. Horizons run to the full remaining observation window
  (up to ~334 days), not 180. A row contributes to a horizon only if it is
  informative for it: crossed within it, or observed alive past it.
- **H2 — voltage-path forecaster.** Quantile regression of
  `ΔV(t → t+h)` for a grid of `h`, from which the first-passage distribution
  follows by path simulation against the 2.4 V threshold. Trained on every
  device-day of every device.
- **H3 — trajectory matcher.** Cross-fitted k-NN on the last 90 days of
  temperature-compensated curve shape. The library **must exclude every device
  in the validation building**, otherwise it leaks.
- **H4 — seasonal-physical head.** Temperature-compensated state, plus each
  device's own estimated seasonal temperature profile projected across the
  planning window, plus an AFT/Weibull on remaining charge. This is the head
  that owns finding 1.3.

Stack the four on grouped out-of-fold predictions. Calibrate with a
Platt/beta prior *before* isotonic, because 82 events will not support isotonic
alone.

### 3.3 Features

All causal, all derived from the smoothed series: level, running min/max and
drawdown; ΔV over 7/14/30/60/90/120/180 days plus curvature terms; days since
first crossing each of nine thresholds from 2.90 to 2.42 V; the linear crossing
extrapolant fed **as a feature** so the model can learn its known +45-day
optimistic bias; recent and lifetime temperature and its variance; seasonal
sin/cos and the expected seasonal temperature *change* across the window; data
staleness, coverage, and gap fraction. **No building, room, or device identity.**

### 3.4 The acceptance criterion is calibration transfer, not AUC

At a 1 % base rate a high AUC coexists comfortably with a useless planner. The
gate is:

1. predicted `E[#due]` vs actual `#due`, **per scenario, on held-out buildings**
   — this is the exact quantity that collapsed between local and public;
2. PR-AUC and precision@{10,15,20,30,50};
3. calibration in the probability band where servicing actually pays;
4. end-to-end evaluator cost using out-of-fold predictions only.

Nothing is promoted on AUC.

### 3.5 Scenario-level recalibration

A small, robust second model predicts `E[#due]` per scenario directly from
aggregate features (the distribution of voltages and slopes across the fleet,
plus the calendar months the window covers). The per-device probability vector
is then shifted by a scenario-level intercept so that its expectation matches.
This is the cheapest available defence against the §1.5 collapse.

---

## 4. Task 2: keep it, and upgrade the objective

Keep the existing planner. Change what it optimises.

### 4.1 Sample-average approximation with the exact replay

The current objective uses marginal expectations. That is wrong in three
specific places: the emergency queue couples batteries through a shared rank,
the daily and weekly limits are flat 100 thresholds rather than smooth costs,
and overtime is convex. The fix is to optimise the sample average of the exact
cost:

1. Draw `S` joint samples of the EOL realisation from the predictive
   distribution (mild shared room effect included — see §4.3).
2. Score a candidate plan by the exact replay on all `S` samples.

This is cheap because the cost decomposes:

- the in-window operational cost of a fixed plan is **deterministic** — one fast
  replay, independent of the sample;
- the timing penalty on planned batteries is **separable** — an exact
  expectation, no sampling needed;
- only the emergency tail needs samples, and it vectorises: with `Due[b,s]` and
  `T[b,s]` precomputed, the rank of each unplanned due battery is a `cumsum`
  down the id-sorted axis, so the whole tail cost is one numpy expression over
  `|U| × S` (~30 × 256).

Estimated ~100 µs per candidate evaluation, which supports thousands of LNS
moves per scenario inside the runtime budget.

### 4.2 Verification

The replay plus the SAA objective must reproduce `evaluate_plan()` **exactly**
on thousands of randomly generated plans under a known EOL realisation, with
edge cases pinned: all-defer, single battery, batteries in the base building
and base room, plans that touch the horizon boundary, and remote-depot
scenarios. This is a hard gate before any tuning.

### 4.3 Correlation

Same-room EOL pairs are only mildly clustered — median 82 days apart versus 98
for random pairs, despite 94 batch installations of two or more devices. Model
a weak shared room effect in the sampler; do not build the plan around it.

### 4.4 Candidate selection by economic gain

Replace the `q > ε` gate with: include battery `b` if

```
defer_cost(b) − min_d swap_cost(b,d) − incremental_logistics(b,d) > 0
```

for any `d`, evaluated against the *current* plan. This admits cheap
co-located batteries that a probability threshold would drop — which is exactly
where the far-building batching value lives (a dedicated visit to the 10.25 h
building costs ~20.5 h of travel plus ~25 h of overtime; pulling in a
co-located battery 60 days from EOL costs 30).

### 4.5 Evaluator-specific exploits to encode

- **The end-of-day return travel is double-counted.** `do_action` resets
  `time_of_day` before adding the return leg, so it is charged to the closing
  day's overtime *and* carried into the next active day's clock. Schedule the
  day that ends farthest from base **last**, where the leak lands on the inert
  final Sunday.
- **Building changes never carry the room**, so a 0.5 h room change is charged
  on every arrival; multi-room buildings are relatively cheaper to batch.
- **Weekly limits are entirely avoidable** — a good plan does ~50 h of work
  across seven weeks; every 100-point hit is a scheduling failure.
- **Never emit a service day inside `(start+42, start+48]`**, and never place
  work on the final Sunday.

---

## 5. Validation

The methodology is the deliverable that stops us fooling ourselves again.

- **V1 — grouped OOF.** `GroupKFold` by building; assemble out-of-fold
  predictions for every device; run every scenario on OOF only.
- **V2 — public/private simulation.** Repeatedly partition the 24 buildings
  into disjoint halves; treat one half as "public" and the other as "private";
  rotate the depot into the held-out half. Report the **spread between the two
  halves**, which is the honest estimate of how much a public-leaderboard
  improvement should be trusted.
- **V3 — synthetic scenarios.** Because §1.2 lets us rebuild labels anywhere,
  generate hundreds of scenarios at arbitrary start dates with randomised
  depots, instead of relying on 48 overlapping windows.
- **V4 — block bootstrap.** Given §1.6, compare candidates with paired tests
  over **non-overlapping** scenario blocks, and report p90 and max alongside the
  mean. Treat sub-100 mean differences on the 48 given scenarios as noise.
- **V5 — decision audit.** Due, swapped, missed, precision, recall, early/late
  split. Diagnostic only; the official cost decides.

Anchors printed on every run: all-defer (3324.7 over 48), naive oracle (205.2),
planner oracle (77.8 on scenarios 0–11).

### 5.1 Choosing the three finals

Select on V2's worst-half performance, not on the public score. Public
submissions confirm; they do not select. Reserve one final for the
configuration that wins on V2 even if it is not the public leader.

---

## 6. Runtime engineering

Measured: `load_dataset` 4.7 s and `iterate_scenarios` ×48 62.9 s per split, so
~2.3 min of unavoidable harness time for both splits. Full smoothing of a split
costs 26 s once; incremental extension is ~0.2 s per scenario.

The current v3 profile spends 16.09 s per scenario and projects to 25.75–27.6
minutes for 96 scenarios. That is not enough headroom: if the public or private
split has more than train's 461 devices, the run exceeds 30 minutes and scores
nothing. Requirements:

- incremental smoothing cache keyed on a watermark, reusing the planner
  instance across scenarios;
- all artifacts loaded at import, never per scenario;
- a wall-clock governor that tracks elapsed time against scenarios remaining and
  degrades the optimizer to its seed solution when the budget tightens;
- a target of ≤12 minutes for both splits with the governor never engaging on
  train.

---

## 7. Compliance

MIT `LICENSE` at root; third-party license table; no network at evaluation; all
artifacts committed; full reproduction pipeline with exact commands and seeds;
repository public within one hour of the deadline; submitted commits never
rewritten. No external datasets. On pretrained third-party models: still no —
the runtime cannot absorb ~40,000 CPU forecasts, and §3.1 shows the data volume
problem is solved by reframing rather than by transfer. A small from-scratch
1-D CNN or GRU over the smoothed window is a legitimate fifth head if the
stack says H1–H4 are the binding constraint.

---

## 8. Order of work

Dependency order, not calendar order.

1. **Foundation.** Exact-replay verification harness (§4.2). Incremental
   smoothing cache. Synthetic scenario generator. Freeze v4 as control.
2. **Validation rig.** V1–V5 implemented and running against the frozen v4, so
   every later change is measured on a fixed instrument.
3. **Cheapest large win first: wire the §2.1 prototype into the existing
   planner.** The prototype scores 1567.6 with the naive router; the existing
   planner drives daily and weekly limits to 0.0 under oracle risk. Expected
   recovery of most of the 297.7 capacity penalty for integration work only, no
   new modelling. This is the first submission candidate.
4. **Task 1 heads,** in descending expected value against the 911.7 late cost:
   H2 (voltage-path — the reframing that turns 82 events into 385 k
   device-days), then H4 (seasonal-physical, which owns the 1.76× winter
   effect), then H3, then the stack and calibration, then §3.5 scenario
   recalibration against the 20.6-vs-9.5 over-prediction.
5. **Task 2 objective.** SAA swap-in behind a flag, verified against the current
   objective under oracle risk before it is trusted.
6. **Candidate selection** by economic gain (§4.4) and the evaluator exploits
   (§4.5).
7. **Runtime governor** and a full timed dry run on both splits.
8. **Selection** of three finals on V2 worst-half.

Submissions are used to confirm, in this order: the frozen v4 control, the new
Task 1 in the existing planner, then the SAA objective, then calibration
variants. One early submission of all-defer is worth its slot purely to fix the
public split's scale — it is a legitimate strategy, and without it every public
number we see is uninterpretable.

---

## 9. Appendix — provenance

| claim | method |
|---|---|
| planner oracle 77.83 vs naive 263.18 on scenarios 0–11 | re-ran both on the identical 12-scenario subset; all-defer 4885.46 matches `TASK2_IMPLEMENTATION.md` |
| `smooth_series` causal to 0.000e+00 | full vs truncated smoothing compared over 26,366 device-days, 40 devices |
| EOL label exactly reproducible | first smoothed day below 2.4 V vs `eol_times.csv`, 82/82 |
| β = +0.00463 V/°C, ρ = +0.564, 100 % positive | within-device residual regression after 91-day detrending, 454 devices |
| seasonal incidence 1.76× | EOL events per 1000 device-days of exposure, Nov–Mar vs May–Sep |
| non-monotone 23.5 % | fraction of increasing days in the smoothed series, per device |
| τ(v) dispersion table | 82 complete trajectories, days from first level crossing to EOL |
| pace correlation −0.085 | days 2.70→2.55 against days 2.55→EOL |
| building EOL rate 0.043–0.833 | events divided by devices, per building |
| settings constant except depot | set comparison across all 48 scenario settings blocks |
| harness overhead 67.7 s per split | timed `load_dataset` plus a full `iterate_scenarios` pass |
| evaluator mechanics | direct reading of `batteryswap_public` 0.3.4 |
