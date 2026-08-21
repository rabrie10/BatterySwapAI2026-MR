# V7: Wiener first passage on within-day features

Written 2026-08-21 after the first real leaderboard signal put us 17th of 19 at
2915.68 against first place's 1160.67.

Every number is out-of-fold by building over all 48 train scenarios, scored with
the official `evaluate_plan()`.

---

## 1. Result

| configuration | mean | late | early | capacity | swaps | recall |
|---|---:|---:|---:|---:|---:|---:|
| all-defer | 3324.7 | 3056.3 | 0.0 | 213.4 | 0 | 0 |
| shipped v3 | 2644.9 | 1700.0 | 610.8 | — | 11.0 | 0.296 |
| V6 hazard classifier | 2526.0 | 1407.5 | 747.6 | 284.8 | 17.2 | 0.449 |
| **V7 Wiener + within-day** | **2293.2** | 1236.7 | 724.2 | 260.6 | 16.5 | **0.502** |

**−232.8 against V6**, outside the ~100 noise floor. Runtime 19.7 minutes
projected for 96 scenarios against a 30-minute limit.

## 2. The measurement that redirected the whole effort

Before changing anything, a two-line control was run: rank batteries by
`margin / -slope`, extrapolating the smoothed voltage to 2.4 V. No model.

| k | physics control | V6 (51 features, 2.8M rows, GBM) |
|---|---:|---:|
| 8 | **0.359** | 0.305 |
| 12 | 0.309 | 0.300 |
| 18 | 0.249 | 0.249 |
| best timing | 1823.4 | 1813.5 |

The gradient-boosted model was worth nothing over a straight line. That explains
why all eight V6 interventions landed inside the noise floor, and it means the
bottleneck was never the model class or the 82-event sample size — it was the
**representation**.

**This control should have been run before V6, not after.** It is the cheapest
diagnostic in the project and it invalidates a week of tuning.

## 3. What the representation was missing

`smooth_series` collapses each calendar day to one median and takes a seven-day
trailing median of those: 8,520,098 hourly readings become 360,847 numbers, and
every V6 feature was a function of that single collapsed series.

The daily cycle is where the early warning lives. Voltage responds to the daily
temperature swing through internal resistance, and resistance rises as a cell
approaches collapse. Measured on the population the level-and-slope rule cannot
rank at all — rows where the extrapolation says more than sixty days to crossing:

| signal | due median | not-due median | ratio | AUC |
|---|---:|---:|---:|---:|
| **within-day dV/dT** | 0.01266 | 0.00329 | **3.84** | **0.871** |
| daily voltage sd | 0.02682 | 0.00686 | 3.91 | 0.819 |
| daily voltage range | 0.08545 | 0.02287 | 3.74 | 0.827 |

Standalone on the whole population the within-day sensitivity (AUC 0.893) beats
the entire physics extrapolation (0.889).

This is the "knee onset" surprise made visible. Those batteries sit at a higher
voltage on a slower slope, so nothing in the smoothed series flags them, and they
cross anyway. The battery literature names the same mechanism: internal
resistance growth is a documented knee precursor, and charge-transfer resistance
dominates knee detection.

`bsai/shape.py` computes these incrementally, mirroring the smoothing cache's
watermark discipline. No temperature filter is applied — `smooth_series` keeps
only 10-30 degrees because that is how EOL is defined, but narrowing the band
would shrink the very swing this measures against.

## 4. Why a Wiener first-passage model

The margin `m(t) = smooth_v(t) - 2.4` is **non-monotonic**: it trends down but
rises and falls with the seasons. That rules out the gamma and inverse-Gaussian
*processes*, which require monotone paths, and points at the Wiener process,
whose first passage to a barrier has a closed form.

Covariates enter through the parameters, not the probability:

```
P(cross within h) = PHI((-m + drop) / s)
                  + exp(2 * drop * m / s^2) * PHI((-m - drop) / s)
```

where `drop` is the expected fall over `h` days and `s` its standard deviation,
both predicted from features. The second term is the reflection correction: it
counts paths that dip below the barrier and come back, which is what makes this
a first-passage probability rather than an end-point probability.

Three properties that reduce error against the alternatives:

1. **Sample efficiency.** Drift and volatility are fitted on every observed
   window on every device — 668,000 of them — while a classifier of "does it
   cross" only ever sees 82 events.
2. **The horizon axis is a formula, not a fit.** V6 stacked 24 horizons and
   learned the curve's shape under a bolted-on monotone constraint. Here
   monotonicity and cross-horizon consistency are automatic.
3. **The parameters transfer.** Drift and volatility are physical. A probability
   level is not, which is why V6's calibration could not be repaired across
   buildings — it over-predicted 2.21x in the closing scenarios and no calibrator
   fitted on the other four folds could see it.

Verified against known limits: at zero drift the formula reproduces the
reflection principle exactly (0.045500 against 2*PHI(-m/s) = 0.045500), and it is
monotone in margin, drift and volatility.

## 5. Ranking at the operating point that is charged

PR-AUC integrates over thresholds we never use. What the leaderboard charges is
precision at 10-25 swaps per scenario.

| k | V6 precision | V7 precision | V6 recall | V7 recall | V6 timing | V7 timing |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 0.300 | **0.373** | 0.381 | **0.474** | 1843.5 | **1561.6** |
| 15 | 0.275 | **0.350** | 0.436 | **0.555** | 1817.2 | **1456.6** |
| 21 | 0.239 | **0.300** | 0.531 | **0.665** | 1813.5 | **1442.2** |

Best analytic timing 1813.5 -> **1442.2**. The V6 curve was flat from k=12 to
k=25 — the model could not rank, so no threshold helped. The V7 curve has a real
minimum.

## 6. The level has to be calibrated on the scenario population

The volatility scale was first fitted on the training cutoffs, where it gave a
predicted-to-actual ratio of 1.025. At **scenario** cutoffs the same model
predicted 13.23 due against an actual 9.46 — ratio 1.40 — and the planner acted
on the level, servicing 26.5 batteries and giving back the entire gain
(2484.7, only −41 against V6).

Refitting the scale against the end-to-end score:

| volatility scale | mean | late | early | swaps | predicted due |
|---|---:|---:|---:|---:|---:|
| 0.85 | 2376.3 | 1434.8 | 599.8 | 13.7 | 7.42 |
| **1.00** | **2293.2** | 1236.7 | 724.2 | 16.5 | 8.83 |
| 1.15 | 2344.0 | 1085.6 | 877.9 | 20.0 | 10.38 |
| 1.40 | 2484.7 | 844.4 | 1211.6 | 26.5 | 13.23 |

A clear interior minimum. This is the same population trap V6 fell into, in a
different place: **calibrate on the population the model is used on**, always.

## 7. What is still on the table

1. **The planner's objective, worth roughly 400-500.** `_expected_score` believes
   1176.8 where the evaluator charges 2328.2 (correlation 0.613). The analytic
   ranking says the best achievable timing at 16 swaps is about 1450; the planner
   delivers 1961. The gap is day placement — it schedules swaps earlier than the
   timing cost wants, to batch trips.
2. **More training data.** The shipped model uses stride 4 and 250 iterations,
   chosen for a fast first result. Stride 2 with 400 iterations is a
   straightforward retrain.
3. **The within-day ablation was not completed** — the no-shape run crashed in
   sklearn's binning on all-NaN columns. Permutation importance on the shape
   columns is the cheaper way to attribute the gain and has not been run, so the
   split between "within-day features" and "Wiener structure" is not yet
   measured.

## 8. Also built, measured, not shipped

`bsai/margin.py` — quantile regression on the running minimum of the margin. The
target is an exact restatement of the label, verified at **1.00000000** agreement
over 271,063 rows, and it beat V6 on PR-AUC (0.4464 against 0.4052). It is kept
because the target builder is the cleanest statement of what EOL means and is
covered by tests, but the Wiener model reached a better PR-AUC (0.4725) on less
data in 87 seconds against 70 minutes, so it is what ships.

## 9. Reproduction

```bash
python tools/train_wiener.py --stride 4          # ~15 min
python tools/validate_v6.py --folds outputs/v7_folds.joblib \
    --model models/v7_wiener.joblib --volatility-scale 1.0
python tools/ranking_v7.py --folds outputs/v7_folds.joblib
python tools/physics_baseline.py                 # the control
python -m unittest discover -s tests
```

Seeds are fixed in `bsai/wiener.py` (`random_state=20260821`) and
`batteryswap_solution/optimizer.py` (`random_seed=20260818`).

`models/v6_hazard.joblib` is removed on this branch: the feature builder now
emits 64 columns and the old artifact expects 54, so loading it would fall back
to the voltage-trend forecaster. It remains in history at `bfdb0ea` on `main`.
