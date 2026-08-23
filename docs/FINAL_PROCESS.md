# Are the Wiener assumptions wrong near the threshold?

_Branch `claude/final-j2w-precision`, 2026-08-23. Code: `tools/fj_process.py`
(diagnostics 1-2), `tools/fj_calib.py` (3-4). No model fitted, no planner run._

The hypothesis: the event definition and the average drift/scatter targets are
right, but the *path law* is wrong near the threshold -- long-lived
near-threshold devices mean-revert against a bounded floor, so volatility does
not accumulate with horizon the way drifted Brownian motion requires, and V8
therefore over-states their crossing probability.

**One of the four diagnostics finds a real departure. The one that decides the
question does not: at the states where it matters, V8's passage law is
calibrated to within 0-3 %, and the zombie over-confidence turns out not to be a
property of the law at all.**

---

## 0. Two things the diagnostics had to control for

**The null has to be simulated.** `smooth_series` ends in a seven-day rolling
median, so consecutive daily values share six of seven inputs. That alone bends
variance scaling and induces increment autocorrelation. Measured against the
textbook `Var = h sigma^2`, a process that is *exactly* Brownian would look
non-Brownian. Every diagnostic below is therefore run twice -- once on the data,
once on synthetic Brownian paths pushed through the same smoother, matched per
device in length and missing-data pattern.

**Censoring fabricates mean reversion.** Requiring a window to *end* before the
crossing keeps only the near-threshold devices that did not cross, which is
exactly the population that appears to have a floor. All windows below start
before the crossing and may end anywhere observed. That is valid here: crossing
devices keep declining afterwards (median **-0.033 V** by 42 days past EOL, only
18 % rising more than 20 mV), so there is no battery-replacement contamination.

## 1. Variance scaling: one band departs, and only one

`Var(dV_h)/h`, each cell normalised to its own `h=7` so the smoothing artefact
divides out. Device-level bootstrap, 1000 draws, on the `h=42 / h=7` ratio:

| margin band | data ratio [95 % CI] | Brownian null [95 % CI] | data / null | devices |
|---|---|---|---:|---:|
| **0.00-0.03 V** | **0.67 [0.49, 0.88]** | **1.64 [1.26, 2.11]** | **0.41** | 58 |
| 0.03-0.05 V | 2.54 [0.73, 6.00] | 1.27 [0.97, 1.67] | 1.99 | 77 |
| 0.05-0.10 V | 1.09 [0.82, 1.53] | 1.19 [0.98, 1.44] | 0.91 | 118 |
| 0.10-0.20 V | 1.06 [0.79, 1.50] | 1.12 [0.99, 1.27] | 0.95 | 199 |
| 0.20 V and up | 1.11 [1.04, 1.19] | 1.29 [1.23, 1.36] | 0.85 | 459 |

* The 0.03-0.05 V "super-diffusion" is **noise** -- its interval spans 0.73 to
  6.00 and overlaps the null. Dismissed.
* Three bands above 0.05 V are **Brownian-consistent**: intervals overlap the
  null throughout.
* **The 0-0.03 V band is a real departure.** Its interval and the null's do not
  overlap: variance stops accumulating past a week, at 41 % of the rate a random
  walk through this smoother would give. That is the hypothesised behaviour, and
  it is confined to the narrowest band -- which is where three of the six repeat
  false positives sit (margins 0.0021, 0.0049, 0.0085).

## 2. Autocorrelation: mild mean reversion everywhere, weakly state-dependent

Lag-1 ACF of daily smoothed increments, device-averaged:

| group | data | Brownian null | data / null | devices |
|---|---:|---:|---:|---:|
| near-threshold survivor | 0.181 | 0.327 | 0.55 | 66 |
| near-threshold imminent | 0.237 | 0.346 | 0.69 | 55 |
| mid margin | 0.194 | 0.344 | 0.56 | 178 |
| healthy | 0.277 | 0.346 | 0.80 | 457 |

The null is flat across groups, as it must be. The data sits below it
everywhere, so there is mild mean reversion throughout -- not a near-threshold
phenomenon. The survivor-versus-imminent contrast the hypothesis predicts is
present and in the right direction (0.181 against 0.237) but small.

## 3. Is sigma over-stated? No -- it is under-stated

The statistic has to be `std((realised - drop) / sigma)`, which is 1.00 when the
scatter model is correct. Comparing an unconditional band standard deviation
against a conditional sigma inflates the ratio by Jensen -- `docs/HANDOVER.md`
trap 5, which cost a whole build cycle once already.

| margin band | n | devices | **std(z)** | mean(z) |
|---|---:|---:|---:|---:|
| 0.00-0.03 V | 84 | 47 | **1.74** | +0.38 |
| 0.03-0.05 V | 103 | 53 | **3.11** | -0.21 |
| 0.05-0.10 V | 410 | 105 | **2.35** | +0.21 |
| 0.10-0.20 V | 1464 | 189 | **2.89** | -0.06 |
| 0.20 V and up | 32840 | 461 | **1.68** | -0.13 |
| the six repeat FPs | 702 | 6 | 1.38 | +0.18 |

`std(z)` is 1.4-3.1, never below 1. **V8's sigma is too small by roughly two to
three times on the censor-free population, not too large.** HANDOVER measured
0.93-1.10 for the same quantity on V8's *training* windows, and both are right:
the scatter model is well specified on the survivor population it is fitted on
and under-specified on the true one.

That is the opposite sign from the hypothesis, which needs over-stated
dispersion to explain over-confidence.

## 4. The passage law against realised crossings -- the decisive one

Every historical cutoff with a known 42-day fate (42,140 of 44,104; 434
crossings), out of fold by building:

| margin band | n | devices | V8 mean p | realised | ratio |
|---|---:|---:|---:|---:|---:|
| **0.00-0.03 V** | 133 | 63 | **0.6820** | **0.6842** | **1.00** |
| 0.03-0.05 V | 140 | 65 | 0.3601 | 0.4929 | 0.73 |
| 0.05-0.10 V | 536 | 118 | 0.1217 | 0.2500 | 0.49 |
| 0.10-0.20 V | 1795 | 200 | 0.0145 | 0.0585 | 0.25 |
| 0.20 V and up | 39536 | 461 | 0.0000 | 0.0009 | 0.02 |

By predicted probability:

| bucket | n | devices | V8 mean p | realised | ratio |
|---|---:|---:|---:|---:|---:|
| 0.70-1.00 | 72 | 41 | 0.8459 | 0.8194 | **1.03** |
| 0.40-0.70 | 145 | 68 | 0.5288 | 0.5310 | **1.00** |
| 0.20-0.40 | 170 | 72 | 0.2854 | 0.4706 | 0.61 |
| 0.10-0.20 | 162 | 82 | 0.1464 | 0.3519 | 0.42 |
| below 0.10 | 41591 | 461 | — | — | 0.04-0.48 |

**At the top of the distribution -- the states the planner acts on, and the
0-0.03 V band specifically -- V8 is calibrated within 0 to 3 %.** It is
systematically *under*-confident everywhere else, by two to four times, which is
the knee-entry under-prediction this project has documented since V9.

So the sub-diffusion of section 1 is real and **already absorbed**. V8's scatter
is a regression with the horizon as a monotone input, not an assumption that
`sigma ~ sqrt(h)`, so a band whose variance stops accumulating is learnable --
and the resulting crossing probability there comes out at 1.00 of realised.

## 5. Then why do the zombies look over-predicted?

Because they are the same six devices counted forty-eight times.

On the 19,890-row scenario population, rows with `p >= 0.40`:

| | rows | devices | realised |
|---|---:|---:|---:|
| all | 407 | 72 | **0.369** |
| **excluding the six repeat devices** | 243 | 67 | **0.617** |

The six contribute **40 % of all high-probability rows** and 48.5 % of the
`p > 0.9` bucket, at a realised rate of 0.000. Remove them and the realised rate
is 0.617 against a predicted level near 0.66 -- calibrated, and matching the
historical population's 0.819 against 0.846 exactly.

**The apparent structural over-confidence near the threshold is six devices out
of 461, over-represented forty-eight-fold because the evaluation re-scores one
fleet on forty-eight dates.** It is not a property of the passage law, and no
alternative process -- empirical analog, OU, or otherwise -- would change it,
because the law is already right for the other 67 devices in the same states.

## 6. Verdict

```
The Brownian diagnostics are valid where the decision is made. Direction closed.
Neither alternative A (empirical analog) nor B (OU) was built.
```

The one genuine departure -- sub-diffusion below 0.03 V -- is already priced by
the fitted scatter regression, which is why calibration there is 1.00. The
sigma error that does exist runs the *wrong way* for the hypothesis and is a
training-population artefact rather than a law error. And the phenomenon that
motivated the whole line is a six-device idiosyncrasy, which is precisely the
thing V19 tried to encode as a rule and lost 403 points of late cost doing.

This is the fifth structural attempt on this branch. For the record:

| attempt | outcome |
|---|---|
| residual losses on the engineered state | 1 of 18 fits beat V8 out of fold |
| matched-volatility state | real, 5/5 folds, only orders at matched margin |
| EOL-aligned trajectory templates | larger matched edge, measured +5.54 through the planner |
| the EOL event definition | V8 already models the official event exactly |
| **the path law** | **calibrated 1.00 where it acts; the zombies are six devices** |
