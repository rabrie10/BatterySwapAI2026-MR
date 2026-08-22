# V9: the passage model was right about shape and wrong about level

Out of fold by building over all 48 train scenarios:

| configuration | local | delta |
|---|---:|---:|
| V8 phase 1 (`de261f5`) | 2145.16 | — |
| **V9 blend, shipped config** | **1923.96** | **−221.20** |

Paired over the same 48 scenarios: **t = 3.82**, **36/48 wins**, and the block-mean
delta over six non-overlapping blocks is **−221.2 ± 88.2**, so this clears the
noise floor on the conservative measure as well as the optimistic one. Five of
the six blocks improve.

| | V8 | V9 |
|---|---:|---:|
| early_swap | 763.39 | **591.05** |
| late_swap | 1026.46 | 1040.21 |
| weekly_limit | 114.58 | 85.42 |
| daily_limit | 87.50 | 72.92 |
| overtime | 79.52 | 67.63 |
| travel | 45.69 | 40.75 |
| swaps served | 17.65 | **14.79** |
| missed | 3.94 | 3.90 |
| **precision** | **0.313** | **0.376** |
| recall | 0.584 | 0.588 |

**Three fewer swaps, the same misses, and 172 points less earliness.** That is the
shape of the public gap: of the 942 points between us and first place,
`early_swap` was +634 and the capacity penalties +248.

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

The passage model keeps its job of supplying the *shape* across horizons, because
the planner needs a full CDF to choose a service day; the blended level is
imposed on that shape. Where the passage probability is too small for its own
shape to mean anything, a fleet median shape recorded at fit time is used.

Out of fold by building, on the same folds, same features:

| | AUC | PR-AUC | AUC below 0.12 V |
|---|---:|---:|---:|
| Wiener passage | 0.9503 | 0.3083 | 0.7589 |
| head alone | 0.9550 | 0.3627 | — |
| **geometric mean** | **0.9581** | **0.3825** | **0.7957** |

The blend beats *both* ends of the weight sweep -- pure passage scores 1709 on the
timing screen and pure head 1725, while the blend scores 1598 -- which is the
signature of two genuinely decorrelated views rather than one dominating.

Blending in probability space rather than by rank is what makes it shippable:
rank depends on the whole scored set, and `predict_grid` is called one building
at a time. The two are equivalent in quality (PR-AUC 0.3889 against 0.3876).

It also removes the calibration inversion that trap 3 was written about. Block
ratios of predicted to actual, before correction, were 0.54 / 1.01 / 1.64 for the
passage model -- a bias that changes sign, which no scalar can fix. For the blend
they are 0.55 / 0.67 / 0.93, a uniform under-prediction, and after the
out-of-fold correction 0.84 / 0.92 / 1.02 against the passage model's
0.78 / 1.08 / 1.31.

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

## 5. Runtime

The shipped configuration trades a little search for headroom:
`solver_seconds 0.5` and `candidate_margin_hours 12` against the previous 1.0 and
24. Measured effect on the score: **+4.5** against the expensive configuration
(1923.96 against 1919.48), which is noise. Measured effect on time: **11.29 s per
scenario against 13.2**, projecting to **20.3 minutes for 96** against the
30-minute cap.

For comparison the previous submission projected 18.0 minutes and scored, so this
is a 13% increase on a run that already fit. The soft deadline in
`bsai/runtime.py` is deliberately left at 17 minutes -- it was not changed for
this, because the machinery that stops a run scoring zero should not be loosened
for a marginal quality gain.

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
