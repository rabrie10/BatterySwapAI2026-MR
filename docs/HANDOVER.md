# Handover: what has been tried, what worked, and the traps

Written 2026-08-21 for a fresh session picking this up cold. Team is *Dynamic
duo*. Lower score is better.

The single most useful thing in this document is section 5, the traps. Nine of
them are recorded because each one cost real time, and most were not obvious.

---

## 1. Where the score stands

| submission | public | rank |
|---|---:|---|
| v3, discrete-time hazard classifier | 4252.33 | 17 of 19 |
| V6, censoring-aware label | 2915.68 | 17 of 19 |
| V7, Wiener first passage + within-day features | **2167.11** | **12 of 12** |
| J2W, first place | 1160.67 | 1 |

Local out-of-fold mean over all 48 train scenarios, which is the only number
allowed to justify a submission:

| configuration | local |
|---|---:|
| all-defer (service nothing) | 3324.7 |
| v3 | 2644.9 |
| V6 | 2526.0 |
| V7 | 2293.2 |
| **V8 phase 1, on branch `claude/v8-precision`** | **2145.1** |
| perfect knowledge, same planner (scenarios 0-11) | 77.8 |

V8 phase 1 is committed at `db85121`, a clean fast-forward from `main`, 62 tests
passing, submission path verified. **It has not been submitted yet.**

## 2. The gap to first place, decomposed

From the leaderboard columns, using `battery_swap = 0.25 h` per swap (which
includes the emergency swap for every missed battery) and `late_swap = 10` per
day late at about 27 days per miss:

| | us | J2W |
|---|---:|---:|
| planned swaps per scenario | 20.4 | 12.6 |
| due batteries caught | 9.67 | 9.40 |
| missed | 2.51 | 2.24 |
| **recall** | **0.794** | 0.771 |
| **precision** | **0.473** | **0.744** |

**We already catch more failures than first place.** The entire 1006-point gap
is that we buy them with 7.8 extra swaps. Solving the two teams jointly gives a
wasted swap costing **86.1** and a due swap about 2.5, which reproduces both
teams' early-per-swap exactly (ours 46.5 observed 46.5, theirs 23.9 observed
23.9). So neither team places swaps at a better *time*; the difference is
*which batteries*.

Break-even: swapping one more battery pays while its probability of being due
exceeds `86.1 / (86.1 + 270) = 0.242`.

Capacity is a symptom, not a cause. Building visits per planned swap are 0.79
for us and 0.83 for J2W -- we are marginally better. The 252 points of overtime
and limit penalties are volume, and volume is swap count.

**Everything reduces to precision.**

## 3. What worked, and why

1. **Censoring-aware labels.** The scored event is not "does the voltage cross
   2.4 V" but "does an EOL *record* exist", and a record can only be filed while
   the device is still observed. Rows with less remaining observation than the
   horizon were being dropped as censored; they are genuine negatives. Dropping
   them removed exactly the population that dominates the closing scenarios.
   Adding `remaining_observation_days` as a monotone feature completed the fix.

2. **Wiener first-passage model** (`bsai/wiener.py`). The margin
   `smooth_v - 2.4` is non-monotonic, which rules out the gamma and
   inverse-Gaussian *processes* and points at the Wiener process, whose first
   passage has a closed form. Drift and volatility are fitted on 668,000
   observed windows rather than 82 events, and the horizon axis becomes a
   formula instead of 24 stacked horizons under a bolted-on monotone constraint.

3. **Within-day features** (`bsai/shape.py`). `smooth_series` collapses
   8,520,098 hourly readings into 360,847 daily medians. The daily voltage
   response to the daily temperature swing -- a proxy for internal resistance,
   the documented knee precursor -- separates due from not-due with **AUC 0.871**
   on exactly the population the smoothed series cannot rank.

4. **Remaining-observation calibration** (`bsai/calibrate.py`, phase 1, −148).
   See trap 3.

5. **Soft capacity limits.** The optimizer treated the 24 h weekly limit as a
   hard constraint. The evaluator charges a flat 100; a missed due battery costs
   200-400, and one visit to the 10.25 h building consumes 20.5 h of a week.

6. **Candidate reduction.** Search evaluations were linear in the fleet (~420)
   when about 9.5 are ever due. Runtime 17.3 s -> 5.6 s per scenario.

## 4. What failed, measured

Do not repeat these. Every one was validated out-of-fold and rejected.

| tried | result |
|---|---|
| Eight V6 knob sweeps (late multiplier, emergency rank, candidate margin, capacity fraction, probability shrink, planned-swap cap, seed choice, per-battery decision layer) | All inside the ~100 noise floor. See trap 1 for why. |
| Refitting isotonic calibration on the scenario population | Ratio 1.36 -> 1.40. No effect. |
| Global probability shrink (0.70 / 0.45 / 0.25) | `s_0` 1250.9 -> 1872.4 at 0.70. See trap 4. |
| Per-battery decision layer instead of the joint search | 2668.4 against 2526.0. The joint search earns its place. |
| Quantile regression on the running minimum (`bsai/margin.py`) | Target verified exact (1.00000000 agreement over 271,063 rows) and PR-AUC 0.4464 against the classifier's 0.4052, but the Wiener model reached 0.4725 on less data in 87 s against 70 min. Kept as a documented alternative, not shipped. |
| **Direct variance regression** (gamma loss on squared residuals, replacing mean-absolute × √(π/2)) | 2170.3 against 2148.8, PR-AUC 0.4281 against 0.4303. See trap 5 -- the motivating measurement was wrong. |
| **Stacking** a logistic layer on `logit(p)` plus the most separable features | PR-AUC 0.2913 against the raw 0.3083. A control given only the probability scored 0.2591, so the wrapper itself destroys information. The passage model already extracts the signal. |
| **Per-device adaptive drift** (random effects) | Disqualified by the stacking result before building: the features it would add are already used. See trap 6. |
| **Peer contrast** (margin and slope against room/building medians) | 2319.3 against 2145.1, recall −0.036, precision −0.023. See below. |

The peer-contrast failure is worth understanding, because the prior evidence was
the strongest in the whole phase. Measured *inside the swapped group* -- so
conditional on the model already assigning similar probability -- a false
positive falls at `+0.00000` against its roommates' median and a true positive
at `−0.00061`. False positives are in cold rooms, not dying. That signal is real
and it is genuinely new information: every other feature is within-device.

It still failed as a feature. The likely reasons are that 82 events cannot carry
five more degrees of freedom, and that most rooms have fewer than three devices
so the contrast is NaN for most rows. **If you retry this, fix those two things
first** -- weight by peer count, or pool to building only, or use it as a
post-hoc veto rather than a feature.

## 5. The traps

These cost the most time. Each is a general lesson, not a detail.

**1. Run the cheapest control before building anything.**
A two-line rule -- rank by `margin / -slope`, no model -- matched the entire
51-feature gradient-boosted V6: precision 0.309 against 0.300 at twelve swaps,
best timing 1823.4 against 1813.5. Fifty-one features and 2.8 M stacked rows
were worth nothing over a straight line. This explains why all eight V6 knob
sweeps landed in the noise: the bottleneck was the *representation*, not the
model class, not the 82-event sample size. `tools/physics_baseline.py` runs in
four minutes and should be re-run whenever the feature set changes.

**2. Calibrate on the population you deploy on.** This bit three times.
V6 fitted isotonic on the stride-sampled training cutoffs; V7 fitted the
volatility scale on the same. A scenario asks about every alive device on one
date, which is not the same population. Symptom: a model that looks calibrated
in training and predicts 13.23 due per scenario against a realised 9.46 at
scenario cutoffs.

**3. A healthy pooled metric can hide two large errors of opposite sign.**
V7's predicted-to-actual due ratio was 0.93 overall, which looks fine. By
scenario block it was **0.54 / 1.01 / 1.64** -- it predicted the most failures
where there were the fewest. This is the real reason every global knob traded
the opening scenarios against the closing ones: **a single scalar cannot correct
a bias that changes sign.** Always break a calibration number down by whatever
axis the deployment varies along.

**4. Watch for confounding before trusting a correction's axis.**
The phase 1 calibration keys on remaining observation window. On this split the
48 scenarios run chronologically September to July, so remaining and calendar
month move together by construction, and incidence is 1.55x higher in
November-March. Remaining correlates better (Spearman 0.839 against 0.557) and
the low end is mechanically explained -- a scenario with 12 days of observation
left can only record failures in 12 of its 42 window days -- so it was kept. **If
a submission shows this correction hurting, the month of the window is the
alternative axis, and it is the more physical one.**

**5. Check the diagnostic before building on it.**
I reported that sigma was under-estimated 15-62 % with a horizon-dependent
error, called it a structural defect, and built a gamma-loss variance regression
to fix it. It barely moved anything, which prompted a re-check: I had compared
`std(residual)` against `mean(sigma)` where `rms(sigma)` is correct. By Jensen's
inequality that inflates the ratio. Measured properly, `std(r/sigma)` is
**0.93 to 1.10** -- the volatility model was right all along. A whole cycle spent
fixing nothing.

**6. Separability does not mean under-weighting.**
`age_days` separates false positives from true positives at AUC 0.712 (FP median
898 days, TP 786). The natural reading is "the model should use this more". The
opposite is true: the separation exists *because* the model already uses it. The
stacking probe settled it in five minutes and disqualified an entire planned
workstream.

**7. A silent fallback turns a crash into a mediocre score.**
`CompetitionPlanner.plan` catches everything and returns all-defer -- correct in
a submission, terrible in validation. A renamed parameter once made every
scenario fall back and the run still printed a number.
`tools/validate_v6.py` now aborts if any scenario falls back. Keep that.

**8. Suspiciously perfect null results are usually plumbing.**
The first calibration attempt returned figures identical to three decimals. It
had been stored on the fold models and read from the `OofHazardModel` wrapper,
which does not carry the attribute. It is now applied inside
`WienerModel.predict_grid`, where the out-of-fold dispatcher has already picked
the right fold.

**9. The Dockerfile must copy whatever the artifact unpickles into.**
`models/v7_wiener.joblib` resolves as `bsai.wiener.WienerModel` and now also
`bsai.calibrate.RemainingCalibration`. `COPY bsai/ ./bsai` covers it, but this
was missing once and the submission would have silently fallen back to the
voltage-trend forecaster. Also: **never bake `ENV BATTERYSWAP_SPLITS` into the
image** -- the official checklist's own `docker run` passes no override, so a
baked-in `train` would have produced a train-only submission.

## 6. What is still unexplored, ranked

1. **The planner's own objective, roughly 400-500 points.** `_expected_score`
   believes **1176.8** where the evaluator charges **2328.2**, correlation 0.613
   (`tools/belief_v6.py`). The analytic ranking says the best achievable timing
   at 16 swaps is about 1450; the planner delivers 1961. Nobody has attacked
   this. It is the largest measured, unexploited gap in the whole project.
2. **Full-strength retrain.** Everything ships at stride 4 and 250 iterations,
   chosen for fast iteration. Stride 2 with 400 iterations has never been run.
3. **Peer contrast, done properly.** See section 4.
4. **Season as the calibration axis** instead of remaining observation.

## 7. Tools you should reuse rather than rebuild

| tool | what it answers |
|---|---|
| `tools/validate_v6.py` | The only number that justifies a submission. Out-of-fold by building, official `evaluate_plan`, prints anchors, aborts on planner fallback. |
| `tools/physics_baseline.py` | The control. Does the model beat two lines of arithmetic? |
| `tools/ranking_v7.py` | Precision and recall at the swap counts the leaderboard actually charges, plus an analytic timing estimate. No planner, so it is fast. |
| `tools/calibration_v6.py` | Reliability at scenario cutoffs, bucketed. |
| `tools/swap_audit.py` | Which batteries the planner swaps against what its own cost tables say. |
| `tools/error_profile.py` | What distinguishes false positives from true positives, feature by feature. |
| `tools/stack_probe.py` | Is a signal present but under-weighted, or already extracted? |
| `tools/importance.py` | Permutation importance on the drift model, grouped. |
| `tools/belief_v6.py` | Believed against realised cost. Start here for item 1 above. |
| `tools/fit_calibration.py` | Fits the remaining-observation correction, out-of-fold by building. |

## 8. Ground rules that have earned their place

- **Out-of-fold by building, always.** The observed failure rate per training
  building spans 0.043 to 0.833 and none of those buildings exist in the public
  data. A random split leaks that and flatters every number.
- **The 48 train scenarios are not 48 samples.** They start a week apart and
  each covers six weeks, so adjacent windows overlap by about 85 % and the
  effective sample size is nearer eight. **Differences under about 100 on the
  48-scenario mean are noise.** `--blocks` reports non-overlapping block means.
- **Never learn building identity.** v3 did and collapsed on public. A
  within-scenario *contrast* is fine; an identity is not.
- **Local is a weak predictor of public, but the direction has held.** V6 -> V7
  was −233 local and −748 public. Local under-states the gain, so far.

## 9. Reproduce the current state

```bash
python tools/train_wiener.py --stride 4 --max-iter 250   # ~8 min
python tools/fit_calibration.py --volatility-scale 1.0   # ~9 min
python tools/validate_v6.py --folds outputs/v7_folds.joblib \
    --model models/v7_wiener.joblib --volatility-scale 1.0
python -m unittest discover -s tests
```

Seeds are fixed in `bsai/wiener.py` (`random_state=20260821`) and
`batteryswap_solution/optimizer.py` (`random_seed=20260818`).

Deeper background: `docs/V7_IMPLEMENTATION.md` (what the V7 generation measured),
`docs/PLAN_V8_PRECISION.md` (the current plan and its refuted premises),
`docs/V6_IMPLEMENTATION.md` (the previous generation).
