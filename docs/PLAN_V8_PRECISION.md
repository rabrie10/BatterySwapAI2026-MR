# V8: close the last 1006 points

Written 2026-08-21 after the V7 submission scored **2167.11**, up from 2915.68,
moving us from 17th to 12th. First place is 1160.67.

---

## 1. The problem has completely changed shape

V7 was aimed at recall, and recall is now **solved**:

| | us | J2W (1st) |
|---|---:|---:|
| recall | **0.794** | 0.771 |
| precision | **0.473** | **0.744** |
| planned swaps | 20.4 | 12.6 |
| due batteries caught | 9.67 | 9.40 |
| missed | 2.51 | 2.24 |
| `late_swap` | 679.0 | 605.8 |

**We already catch more failing batteries than first place.** We just pay for
7.8 extra swaps per scenario to do it.

Gap decomposition, 1006.4 points:

| component | us | J2W | gap | share |
|---|---:|---:|---:|---:|
| **early_swap** | 950.1 | 302.7 | **+647.4** | **64.3%** |
| daily_limit | 185.4 | 85.4 | +100.0 | 9.9% |
| weekly_limit | 131.2 | 39.6 | +91.7 | 9.1% |
| late_swap | 679.0 | 605.8 | +73.1 | 7.3% |
| overtime | 128.7 | 68.3 | +60.5 | 6.0% |
| travel | 61.1 | 38.3 | +22.8 | 2.3% |
| building_change | 16.1 | 10.5 | +5.6 | 0.6% |
| room_change | 9.7 | 6.3 | +3.4 | 0.3% |
| battery_swap | 5.7 | 3.7 | +2.0 | 0.2% |

Two facts that decide the whole plan:

**Early cost is entirely explained by precision.** Solving the two teams jointly
gives a wasted swap costing **86.1** (172 days beyond the horizon) and a due swap
about 2.5. That reproduces both teams' early-per-swap exactly: ours
`86.1 x 0.527 + 2.5 x 0.473 = 46.5` (observed 46.5), theirs
`86.1 x 0.256 + 2.5 x 0.744 = 23.9` (observed 23.9). Neither team is placing
swaps at a better time than the other — the difference is *which batteries*.

**Capacity is a symptom, not a cause.** Building visits per planned swap: ours
0.79, J2W's 0.83 — we are marginally *better*. The 252 points of overtime and
limit penalties are volume, and volume is swap count.

So: **the entire 1006-point gap is one number, precision, and everything else
follows from it.**

## 2. The economics tell us exactly where to stop

A wasted swap costs 86.1; a missed due battery costs about 270. Swapping one
more battery is worth it while its probability of actually being due exceeds

```
86.1 (1 - p) = 270 p    =>    p = 0.242
```

Our *average* precision across 20.4 swaps is 0.473, so the first dozen or so are
clearly worth it. The question is where the **marginal** swap falls below 0.242 —
and at 20.4 swaps it plainly has, because J2W stops at 12.6 and catches the same
number of failures.

## 3. Why do we swap 20.4 when local tuning said 16.5?

Three candidates, in order of how cheaply they can be settled.

**H1 -- The submission runs a different model from the one we tune on.**
`tools/validate_v6.py` scores the five out-of-fold models; `script.py` ships the
production model trained on all five buildings. More data could mean sharper,
more confident probabilities and therefore more swaps.

**MEASURED, AND REFUTED.** Across all 48 train scenarios the production model
predicts *fewer* due batteries than the out-of-fold models, not more:

| | predicted due/scenario | ratio to actual | over p>0.26 |
|---|---:|---:|---:|
| out-of-fold | 8.83 | 0.934 | 11.21 |
| production | 8.45 | 0.894 | 10.44 |

Production runs at 0.957x the out-of-fold level, so if anything it should swap
slightly *fewer*. This is not the cause.

**H2 -- The planner services batteries the ranking would never pick. CONFIRMED.**
The same measurement settles it. At the shipped setting the out-of-fold model
puts only **11.21 batteries per scenario above the break-even probability of
0.26**, yet the planner swaps **16.5**. Those extra **5.3 swaps per scenario are
ones the economics say to skip.** The optimizer adds them because its expected
cost includes trip batching, so a low-probability battery becomes attractive
purely by being co-located with a visit that is happening anyway.

They are also the worst mistakes available. A battery nobody ranks highly is
precisely one whose effective end of life is distant, so it pays close to the
full wasted-swap cost. Locally that is 5.3 x 60.6 = **321 points per scenario**;
on public, where a wasted swap costs 86 rather than 60, the same leak is worth
**over 450**.

This also explains 20.4 against 16.5 without needing H3: the leak scales with how
many visits a plan makes.

**H3 -- The public split is simply different from train.** Possible as a residual,
no longer needed to explain the gap, and the least actionable. The split is not
obviously harder: production precision there is 0.473 against 0.502 out-of-fold
locally at a comparable operating point.

## 4. Plan

### Phase 0 -- Diagnostics before any modelling (about 1 hour)

The V6 generation cost a week because a two-line control was not run first, so
nothing gets built until these numbers exist. Two are already in.

1. ~~Production against out-of-fold level.~~ **Done -- H1 refuted, H2 confirmed.**
2. ~~Does the planner swap more than the economics justify?~~ **Done -- 16.5
   against 11.2, a leak of 5.3 swaps per scenario.** What remains is to price the
   leak exactly by comparing the planner's chosen set against the ranking's, per
   scenario, rather than by counts alone.
3. **Characterise the false positives.** For batteries we swap that were not due:
   margin, drift, within-day resistance, distance to substitute EOL. If they
   cluster, that cluster is a feature. This feeds Phase 2c directly.
4. **Marginal-precision curve in fine steps around 12-21 swaps**, to locate where
   the marginal swap crosses the 0.242 break-even, and confirm Phase 1's target.

### Phase 1 -- Stop the batching leak (highest certainty, ~350-500)

H2 is already confirmed, so this does not wait on Phase 0.

Batching is sound reasoning when a battery is nearly due and nearly free to
reach. It is wrong when the battery's effective end of life is 172 days out,
which is the average case for the ones the optimizer is picking up. The fix is to
require a **standalone** economic justification, and let batching decide only
*when* a swap happens, never *whether*:

- Require every serviced battery to clear the break-even probability derived from
  its own cost table -- about 0.26, but computed per battery rather than
  hard-coded, since the wasted-swap cost varies with distance to substitute EOL.
- Tighten `select_candidates`, generous by design, so the local search cannot
  reach batteries that fail that test.
- Re-tune `volatility_scale` afterwards: with the leak closed, the optimal level
  moves.

Expected: swaps 20.4 toward 13-15, early down 400-600, capacity down 150-170,
late up 150-250. **Net 350-500**, and it is economics rather than modelling, so
confidence is high.

### Phase 2 — Precision (the real prize, ~300-700)

Four independent lines, each validated out-of-fold before it is kept.

**2a. Per-device adaptive drift — the literature's biggest single lever.**
Our drift is a pure function of features: two devices with identical current
features get identical predicted drift, even if one has been degrading steadily
for two years and the other has been flat. Every Wiener RUL paper treats
unit-to-unit heterogeneity plus Bayesian updating of the drift as the primary
accuracy gain. Concretely: give the model each device's own realised drift over
several windows, and shrink the predicted drift toward it with a weight set by
how much history that device has (empirical Bayes). This directly separates "this
looks like a dying battery" from "this specific battery has always looked like
this and is stable" — which is precisely the false-positive failure mode.

**2b. Within-day features aimed at the knee.** The current set is trailing means
plus three ratios. Missing, in rough order of expected value:
- the *trend* and *acceleration* of the resistance proxy, not just its level;
- a night-versus-day voltage contrast, which is a cleaner resistance probe than a
  regression across all 24 hours;
- interaction with margin — a resistance rise matters far more at 0.05 V of
  headroom than at 0.4 V;
- longer baselines (90, 180 days) so the "rise" is measured against a stable
  reference.

**2c. Error-targeted features.** Whatever Phase 0.3 finds. This is the user's
"target the weaknesses" instinct and it is the right method: fix what is actually
breaking rather than what is theoretically improvable.

**2d. Stacking.** We have three genuinely decorrelated views: the Wiener
first-passage model, the margin-quantile model (built, measured at PR-AUC 0.4464,
not shipped), and the physics control. A stack fitted on out-of-fold predictions
typically buys 5-15% of ranking quality. Cheap, because all three already exist.

### Phase 3 — Capacity, only if a residual gap survives Phase 1

Once swap count falls, re-measure. If daily and weekly penalties remain above
about 1.3x J2W's per swap, look at day packing; otherwise leave it alone.

### Phase 4 — Full-strength retrain

The shipped model uses stride 4 and 250 iterations, chosen for a fast first
result. Once the feature set and structure are settled, retrain at stride 2 with
400 iterations. Do this **last** — retraining a design that is still moving wastes
the compute.

## 5. What this realistically achieves

| | expected | confidence |
|---|---:|---|
| Phase 1 alone | ~1650-1850 | high — economics, not modelling |
| Phase 1 + 2 | ~1350-1550 | medium |
| matching J2W (precision 0.744) | ~1160 | low without a further insight |

Phases 1 and 3 are close to arithmetic. Phase 2 is genuine research: raising
precision from 0.473 to 0.744 at equal recall is a large ask, and I will not
promise it. What I will promise is that each line is validated out-of-fold before
it ships, and anything that does not beat the current model gets dropped and
recorded.

## 6. Two methodology rules, both learned the hard way

1. **Run the cheap control first.** The physics baseline invalidated a week of V6
   tuning in four minutes. Phase 0 exists for this reason.
2. **Calibrate on the population you deploy on.** This has now bitten twice — V6's
   isotonic fitted on training cutoffs, V7's volatility scale fitted on the same.
   H1 is the same error a third time, in a new place: tuning on out-of-fold
   models and shipping a production one.

## 7. Explicitly not doing

- **Rebuilding Task 2.** Building visits per swap already match first place.
- **A deep sequence model.** 82 events; the variance is the wrong direction when
  the goal is fewer mistakes.
- **Chasing `late_swap`.** At 679 against 606 it is 7% of the gap and our recall
  already exceeds first place's. Any further push there costs precision, which is
  the thing we cannot afford.
