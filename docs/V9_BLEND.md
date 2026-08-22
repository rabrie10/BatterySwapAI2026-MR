# V9: the passage model was right about shape and wrong about level

Out of fold by building over all 48 train scenarios:

| configuration | local | delta | t | wins |
|---|---:|---:|---:|---:|
| V8 phase 1 (`de261f5`) | 2145.16 | — | | |
| blend, single-horizon head | 1923.96 | −221.20 | 3.82 | 36/48 |
| blend, multi-horizon bagged head | 1855.28 | −289.88 | 4.27 | 34/48 |
| **+ out-of-fold level correction (shipped)** | **1806.71** | **−338.46** | **5.30** | **42/48** |

The block-mean delta over six non-overlapping blocks is **−338.5 ± 79.3**, so this
clears the noise floor on the conservative measure as well as the optimistic one.
Five of six blocks improve by 331 to 518; the sixth is +16.6.

Each step is smaller than the whole. The multi-horizon head is worth −68.7 on its
own (t = 1.60) and the level correction −52.2 (t = 1.57); neither is significant
in isolation, and they are kept because they point the same way as the measurement
that motivated them and because the combination is.

| | V8 | V9 shipped |
|---|---:|---:|
| early_swap | 763.39 | **558.05** |
| late_swap | 1026.46 | 985.00 |
| weekly_limit | 114.58 | 79.17 |
| daily_limit | 87.50 | 56.25 |
| overtime | 79.52 | 63.31 |
| travel | 45.69 | 39.11 |
| swaps served | 17.65 | **14.65** |
| missed | 3.94 | 3.62 |
| **precision** | **0.313** | **0.398** |
| **recall** | 0.584 | **0.617** |
| seconds per scenario | 9.85 | 12.78 |

**Three fewer swaps, fewer misses, and every single cost component down.**
Precision and recall both rise, which is the thing three generations of work had
not managed: every previous gain in one cost the other. That is the shape of the
public gap -- of the 942 points between us and first place, `early_swap` was +634
and the capacity penalties +248.

---

## 1. What the leaderboard table gave away

The public board publishes ten cost columns per team, and two of them are almost
direct readouts of a plan.

**`battery_swap / 0.25` is the exact number of swaps performed**, planned plus
emergency. Verified against our own out-of-fold run: 5.40/0.25 = 21.6 = 17.65
served + 3.94 missed. So first place performs 15.08 swaps per scenario and we
performed 21.69.

**`early_swap` divided by that is what an average swap costs in earliness**: 20.6
for first place, 43.6 for us. That factor of two survives every assumption you
can vary about how many batteries are due, which is what says the gap is not only
"how many" but "which".

One structural fact makes the arithmetic exact. `devices.csv` gives 445 of 461
devices the *same* `end_time`, so the evaluator's substitute end of life is a
fixed calendar date and a wasted swap costs
`0.5 x (that date - scenario start - swap day)`. It ranges from about 182 in the
opening scenarios to about 17 in the closing ones.

## 2. The defect: a Gaussian tail on a process that does not have one

Measured on the cached scenario-cutoff population, realised deaths against the
probability mass the shipped model assigns, bucketed by margin (`smooth_v - 2.4`):

| margin | rows | deaths | predicted | ratio |
|---|---:|---:|---:|---:|
| ≤ 0.05 V | 495 | 164 | 289.0 | 0.6 |
| 0.05-0.10 | 934 | 141 | 118.2 | 1.2 |
| 0.10-0.15 | 1190 | 74 | 35.8 | 2.1 |
| 0.15-0.20 | 1353 | 42 | 7.0 | 6.0 |
| 0.20-0.30 | 3016 | 25 | 1.3 | 19.1 |
| 0.30-0.50 | 6391 | 8 | 0.06 | 99.2 |

**30% of all deaths happen at more than 0.10 V of margin, and the model predicts
39 of those 137.** The first-passage probability is
`PHI((-m + drop)/sigma) + reflection`; with `drop` about 0.02 V and `sigma` about
0.03 V, a margin of 0.25 V gives `1.6e-14`. The realised rate there is 0.0094.

Two consequences, both measured:

* The due batteries the plan misses carry a **median predicted probability of
  0.008**. They are not ranked low, they are declared safe. Only 3.2% of misses
  were ranked above the lowest-ranked battery the planner *did* swap, so the
  decision layer is not at fault.
* Inside the population the model calls safe, plain `voltage` separates the
  deaths at **AUC 0.913** -- median 2.539 V, dying in a median of 27 days.

A multiplicative calibration cannot repair this: a factor of a hundred on
`1e-14` is still zero. The level has to come from somewhere else.

## 3. The fix: geometric mean with a discriminative head

`bsai/blend.py`. A gradient-boosted classifier is fitted on the 64 features plus
the remaining observation window, on the **scenario-cutoff population** -- one row
per (scenario, device) the forecaster is actually asked about -- and combined with
the passage probability as a geometric mean at the 42-day decision.

The head is fitted at seven horizons rather than only at the decision, so both
halves of the blend carry a shape and the geometric mean is taken column by
column across the whole horizon grid.

Out of fold by building, on the same folds, same features:

| | AUC | PR-AUC |
|---|---:|---:|
| Wiener passage | 0.9503 | 0.3083 |
| head alone, single horizon | 0.9550 | 0.3627 |
| head alone, multi-horizon bagged | 0.9609 | 0.3677 |
| geometric mean, single horizon | 0.9581 | 0.3825 |
| **geometric mean, multi-horizon bagged** | **0.9590** | **0.3953** |

**The head is fitted at seven horizons, not just the 42-day decision.** Stacking
the same rows at 14 to 126 days takes the positives from 454 to 1114 and gives
the head a horizon axis under a monotone constraint. That buys two things: a
better ranking at the decision, and a shape -- so the blend applies at every
horizon rather than being imposed on the passage model's shape at one point. The
geometric mean of two functions monotone in the horizon is monotone, so the CDF
contract the planner reads survives.

Heads are bagged over five seeds because a single head moves the timing screen by
±34 between seeds, larger than most of the effects worth chasing here. The five
heads are evaluated as one tall design rather than one call per horizon, which
turns 1.6 seconds per scenario into 1.0; `tests/test_blend.py` pins the two paths
to identical answers.

The blend beats *both* ends of the weight sweep -- pure passage scores 1709 on the
timing screen and pure head 1725, while the blend scores 1598 -- which is the
signature of two genuinely decorrelated views rather than one dominating.

Blending in probability space rather than by rank is what makes it shippable:
rank depends on the whole scored set, and `predict_grid` is called one building
at a time. The two are equivalent in quality (PR-AUC 0.3889 against 0.3876).

It also removes the calibration inversion that trap 3 was written about. Block
ratios of predicted to actual, before correction, were 0.54 / 1.01 / 1.64 for the
passage model -- a bias that changes sign, which no scalar can fix. For the blend
they are a uniform under-prediction, and after the out-of-fold correction they
are **0.83 / 0.94 / 0.96** against the passage model's 0.78 / 1.08 / 1.31.

## 4. The planner's local search was a one-way ratchet

Separately, and worth recording even though its own effect is small:

With the shipped settings the local search evaluates **fifteen** general moves.
`robust_emergency_samples = 4` makes `due_samples` longer than one row, which
caps `search_budget` at `uncertain_local_search_evaluations = 35`, of which
`repair_reserve` takes 20.

Those fifteen were spent entirely on the first move class -- reinserting
*deferred* batteries, of which there are easily a hundred candidates. The only
move that can **remove** a scheduled swap, `moves.append((batteries, None))`, is
appended last and was never reached. The search could add swaps and never drop
them.

`PlannerConfig.move_order = "interleaved"` round-robins the move classes and puts
removals first, including per-battery removals ordered by ascending predicted
probability, which is where the batching leak lives. Measured on the same 48
scenarios it is worth **−15** against the legacy order (1919.48 against 1934.43)
and it is **faster** (13.2 s per scenario against 14.6), because removing swaps
early makes every later replay cheaper. Kept for the speed and the structure; the
score difference on its own is inside the noise floor.

Raising the budget outright does work -- 400 evaluations scored **−71.9** on 24
paired scenarios, 21/24 wins, with an *identical* swap set, so the gain is pure
day packing -- but it costs 26 s per scenario, which projects to 44 minutes for 96
and blows the 30-minute cap. Not shipped. **If the evaluation budget ever grows,
this is the first knob to turn.**

## 4b. The level was 11% low out of fold, and it mattered in exactly one place

The multi-horizon blend left one block worse: scenarios 32-39 went from 1970.4 to
2210.9, serving 9.50 against 8.00 due and missing 5.25 against 3.88. It was
**under**-swapping there.

The cause is measurable and general. `RemainingCalibration` is fitted on four
buildings and applied to the fifth, and across the five held-out folds the
corrected prediction lands at **0.89** of the realised count. The shipped model is
fitted on all five buildings and deployed on none of them, so it inherits the same
gap -- an in-sample calibration is optimistic about a building it has never seen.

`tools/fit_calibration.py` now measures that shortfall and writes it onto the
artifact as `BlendedModel.level_scale`. It came out at **×1.119**, and applying it
scores **1806.71** against 1855.28 -- with the gain concentrated almost entirely in
the block that was wrong (−247.8 there, ±16 in every other block). A hand-set 1.15
scored 1803.07, so the derived value and the swept value agree; the derived one
ships because it is a measurement rather than a knob.

This is not swap-count tuning, and the distinction matters because local and the
leaderboard disagree about swap count: locally the optimum wants more swaps, the
leaderboard says fewer. A *calibrated* probability lets the planner's own
expected-cost rule find the operating point for whichever split it is given --
and the two splits genuinely differ, because the substitute end of life is a fixed
calendar date and its distance from the scenario changes what a wasted swap costs.

## 5. Runtime

The shipped configuration trades a little search for headroom:
`solver_seconds 0.5` and `candidate_margin_hours 12` against the previous 1.0 and
24. Measured effect on the score: **+4.5**, which is noise.

The end result is **12.78 seconds per scenario, projecting to 22.7 minutes for
96** against the previous submission's 18.0. The soft deadline in
`bsai/runtime.py` is deliberately left at 17 minutes, so the last quarter of the
run will plan with the cheap search: that is what the governor is for, and the
search is measurably worth little now that the model is better -- interleaving the
move classes is worth −15 and quadrupling the budget −72, against −338 for the
model. The machinery that stops a run scoring zero should not be loosened to
recover a fraction of that.

**Do not try to buy the time back by shrinking the search.** At
`uncertain_local_search_evaluations = 20` the budget arithmetic gives
`repair_reserve = 20` and therefore `general_budget = 0`: no general move is ever
evaluated, the plan stays at the raw CP-SAT seed, and the repair loop grinds over
its limit-hit days. Measured: **1827.85 and 16.56 seconds per scenario**, with a
worst scenario of 63 -- worse *and* slower than the 80/35 that ships. The cliff is
at `uncertain <= repair_reserve`, which is floored at 20.

## 6. What this does not fix

The ranking is still the whole problem, and it is still hard.

* With a perfect ranker the timing cost would be about 95 per scenario. The best
  ranking measured here reaches about 1590. **Essentially all of the remaining
  gap is which batteries get chosen.**
* Ranking by margin alone still matches the shipped passage model at 12 swaps
  (precision 0.370 against 0.363). The blend finally beats the two-line control
  (0.377 at 12, 0.406 at 10), which is the first time anything in this project
  has -- but only just.
* Deaths are spread almost uniformly across margin bands: 1.5 to 1.9 per scenario
  in each of the first five. Margin barely concentrates them.

## 7. Measured and rejected in this generation

| tried | result |
|---|---|
| **Expected-cost ranking** using the AFT's predicted EOL date to price a wasted swap by *when* the battery actually dies | Worse at every operating point. The AFT over-predicts time to EOL by a consistent factor of 2-3 (predicted median 65 days against a true 20; 155 against 76), so subtracting it adds noise. |
| **Peer contrast** rebuilt properly -- within-scenario building and room medians of margin and slope, plus rank and peer count, at 100% coverage | Blend PR-AUC 0.3825 -> 0.3891 and the timing screen improves at four of six operating points and worsens at two. Inside noise; not shipped. See `HANDOVER.md` section 4 for the previous attempt. |
| **Head hyperparameters** (depth 7/15/31, 600 iterations, strong regularisation) | A shallower head has a better PR-AUC (0.4029 against 0.3627) and the *same* timing at the operating points. Re-running the shipped head across three seeds moves timing by ±35, which is larger than any of the differences. Left alone. |
| **Oracle-free swap-day policy** -- place a swap on a day chosen from its predicted probability | The planner puts the median swap on day 1 of a 42-day window, and the naive headroom looks like 333 per scenario. Fitted on three blocks and scored on the other three it is **56**, below the noise floor. The day-1 choice is close to correct: with an asymmetric 0.5/10 loss the optimum is the 4.8th percentile of the predicted failure time. |
| **`end_time` as a leak** -- does a device's data stop when it dies? | No. 445 of 461 devices share one export date and dying devices keep reporting for a median of 204 days after their EOL. |
| **Raising the local-search budget to 400** | −71.9 on 24 paired scenarios but 44 minutes projected for 96. See section 4. |
| **Shrinking the local search** to 50/20 to buy deadline headroom | 1827.85 at 16.56 s per scenario against 1806.71 at 12.78 -- worse *and* slower. `repair_reserve` is floored at 20, so `general_budget` becomes zero and no general move is ever evaluated. |
| **A tighter hard bound on one day's work** (`max_daily_hours_factor` 1.0), aimed at the three scenarios whose base is 10.25 hours from everything | **+18.75, 1/48 wins.** Forbidding a 41-hour day does not split it, it defers the batteries -- and s_23 gets *592 worse*, because a deferred due battery in a far building buys its own emergency trip. This is the reasoning already written into `optimizer.py`, now measured. The knob stays, defaulting to the shipped behaviour. |

## 8. Reproduce

```bash
python tools/train_wiener.py --stride 4 --max-iter 250
python tools/fit_calibration.py --volatility-scale 1.0
python tools/build_scenario_frame.py --folds outputs/v7_folds.joblib
python tools/train_blend.py
python tools/fit_calibration.py --folds outputs/v9_blend_folds.joblib \
    --model models/v9_blend.joblib --volatility-scale 1.0
python tools/validate_v6.py --folds outputs/v9_blend_folds.joblib \
    --model models/v9_blend.joblib --volatility-scale 1.0 \
    --solver-seconds 0.5 --candidate-margin 12
python -m unittest discover -s tests
```

`tools/build_scenario_frame.py` writes the cache that makes ranking experiments
cost seconds instead of thirteen minutes; `tools/rank_lab.py` scores any
candidate against it at the swap counts the leaderboard charges.
