# Stable low state or terminal decline? A matched near-threshold study

_Branch `claude/final-j2w-precision`, 2026-08-23. Code: `bsai/terminality.py`,
`tools/fj_terminality.py`, `tools/fj_matched.py`. Every number is out of fold by
building on the 19,890 cached scenario rows; planner numbers are the real
`CompetitionPlanner` and the official `evaluate_plan` over all 48 train
scenarios._

## 0. The correction this document starts from

`docs/FINAL_RESIDUAL_OBJECTIVES.md` closed with "Task 1 is closed". **That was
too strong.** What the residual-objective experiment closed is narrower:
*supervised reranking of V8, using the currently engineered 64-feature state and
~75 failure devices, does not transfer.* The oracle control in
`docs/FINAL_J2W_RESULTS.md` §6 -- V8's own curves, V8's own mass, handed out in
perfect order, 1984.57 -> 882.61 -- proves a far better ordering exists. The
question was never whether headroom exists. It is whether any observable
predicts it.

This document tests the strongest remaining structural argument for a *new
observable*, rather than a new loss.

## 1. The misspecification

The first-passage law asks `P(the path ever touches 2.4 V)`, and as the margin
goes to zero that probability goes to one whatever the drift regressor says. The
data contains devices that contradict it outright. `docs/V10_FINDINGS.md`
records four with per-device floors at **2.402-2.416 V** that "kiss the threshold
for years and never cross"; this branch's own profile found **six** devices
swapped in 29 to 48 of the 48 scenarios that never die, carrying half of V8's
wasted swaps.

So the quantity the model may be missing is not the distance to the barrier. It
is whether the cell has *entered* an irreversible decline or is sitting in a
stable low state it has occupied for a long time. Those are identical in
`(margin, drift, sigma)` and could differ in the trajectory.

## 2. What the repository had already falsified, and what it had not

Checked before building anything. Every dwell experiment on record is a
probability-**level** knockdown, and none is a matched study:

| prior experiment | what it was | result |
|---|---|---|
| `V10_FINDINGS` isotonic + dwell adjustment | reshaped p, both models | 2332.4 / 2248.2 |
| `V10_FINDINGS` dwell knockdown on shipped calibration | reshaped p | 2228.2, "knocked-down cells include genuine catches; their budget slots refill with worse candidates" |
| `V11_TRANSFER_FINDINGS` dwell knockdown, remaining-gated | reshaped p | 2155 (+93), "cap-slot refill effect again" |
| `PAIRED_SELECTION` persistence demotion | planner selection layer, shipped as V19 | public 2113.43, late +403 |

All four are volume or level changes, and all four failed through the budget.
**None asked whether, at matched margin, a trajectory signal ranks imminent EOL
above long-lived near-threshold survivors.** That is a different question and it
is the one below.

## 3. The matched design

`bsai/terminality.py` computes twenty trajectory signals from each device's own
smoothed daily grid up to the cutoff and nothing after it: time already spent
near the barrier, longest and current low-voltage runs, the device's own floor
and how long it has stood, new-low counts, rebound count/mean/max, recovery-day
fraction, recent minimum against recent median, volatility ratios, slope-sign
consistency, decline against the device's own early-life plateau, and the slope
and new lows that survive removing `0.00463 V/degC x (T - 20)`.

Extraction covers **all 19,890 rows**, 91-100 % per signal
(`outputs/fj_terminality.npz`).

The comparison is a matched case-control:

* population: margin in **0 to 0.10 V**;
* **cases**: an EOL record inside 42 days -- 305 rows, 71 devices;
* **controls**: *observed* to survive past 42 days -- 916 rows, 88 devices;
  and a long-survivor subgroup observed past 90 days;
* a case and a control are compared **only inside the same scenario and the same
  0.01 V margin bin**, so absolute voltage cannot solve it and neither can the
  calendar, the season or the remaining-observation window;
* **514 matched pairs**, weighted so each of the 63 case devices carries equal
  total weight.

## 4. Result: the hypothesis is refuted, and something else is not

**The stable-versus-terminal mechanism as stated is refuted.** Every signal
designed for it is at chance on matched pairs:

| designed-for-terminality signal | matched concordance | edge |
|---|---:|---:|
| `frac_below_245_180` (time spent near the barrier) | 0.471 | 0.029 |
| `longest_run_below_245_180` | 0.448 | 0.052 |
| `days_since_new_low` (is the floor old?) | 0.475 | 0.025 |
| `new_lows_90` | 0.478 | 0.022 |
| `floor_gap` | 0.478 | 0.022 |
| `floor_age` | 0.487 | 0.013 |
| `rebound_count_90` | 0.507 | 0.007 |
| `frac_up_days_90` | 0.500 | **0.000** |
| `slope_sign_consistency_90` | 0.491 | 0.009 |
| `detrended_slope_180` | 0.488 | 0.012 |
| `residual_new_low_90` | 0.520 | 0.020 |

It is **not** "how long has it been down there", **not** "is the floor old",
**not** "does it rebound", and **not** "does the decline survive temperature
removal". At matched margin none of those knows anything.

**What does separate them** (long-survivor controls, 402 pairs):

| signal | matched concordance | edge |
|---|---:|---:|
| `slope_14` | 0.312 (lower) | **0.188** |
| `slope_comp_14` | 0.313 | 0.187 |
| **`std_ratio_30_180`** | 0.672 (higher) | **0.172** |
| **`median_minus_min_90`** | 0.664 | 0.164 |
| `crossing_14` | 0.340 | 0.160 |
| `knee_trend_residual` | 0.342 | 0.158 |
| **V8's own probability** | **0.574** | **0.074** |

The picture is not persistence. It is that **a cell about to die has recently
become erratic and is falling fast over the last fortnight**, while a stable low
cell is quiet. And V8, at matched margin, retains an edge of only 0.074.

### Only one of them survives building transfer

Matched pairs grouped by the case row's building fold (the signals are unfitted,
so this is a stability check rather than a leakage one):

| signal | pooled | f0 | f1 | f2 | f3 | f4 | worst | folds > 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V8 probability | 0.574 | 0.499 | 0.542 | 0.625 | 0.509 | 0.638 | 0.499 | 4/5 |
| `slope_14` | 0.677 | **0.342** | 0.579 | 0.824 | 0.691 | 0.654 | 0.342 | 4/5 |
| **`std_ratio_30_180`** | 0.670 | 0.605 | 0.542 | 0.663 | 0.759 | 0.635 | **0.542** | **5/5** |
| `median_minus_min_90` | 0.647 | 0.283 | 0.495 | 0.657 | 0.659 | 0.731 | 0.283 | 3/5 |
| `knee_trend_residual` | 0.650 | 0.294 | 0.560 | 0.639 | 0.696 | 0.688 | 0.294 | 4/5 |
| `v_std_30` (the same quantity, *absolute*) | 0.582 | 0.466 | 0.190 | 0.372 | 0.697 | 0.717 | **0.190** | 2/5 |

`slope_14` has the best pooled edge and collapses to 0.342 on fold 0.
**`std_ratio_30_180` -- the last 30 days' trajectory volatility against the same
device's own 180-day baseline -- is above chance in five of five folds.** Its
absolute twin `v_std_30` manages two of five and a worst fold of 0.190, which is
exactly the building-fragility `docs/V11_TRANSFER_FINDINGS.md` documented for
`beta_30` against `beta_rise`: scales are building-specific, ratios against the
device's own history are not.

It is also genuinely new. V8's three shipped ratio features are at chance on the
identical pairs -- `v_std_rise` 0.449, `beta_rise` 0.535, `v_range_rise` 0.480 --
because those are *within-day* shape ratios, and this is the day-to-day
volatility of the smoothed trajectory.

## 5. Deployed, it is worth nothing -- and the reason is the interesting part

Order-only, mass preserving (`sum p / scenario = 9.4040`, exactly V8's), applied
only inside the band it was measured in.

**The first deployment was wrong, informatively.** Blending V8's rank with the
volatility rank across the whole 0-0.10 V band gives the *best matched
concordance in the whole study* -- 0.699 -- and costs **+147** end to end:

| deployment | weight | matched concordance | k=18 timing | precision | recall |
|---|---:|---:|---:|---:|---:|
| V8 | — | 0.574 | 1738.2 | 0.322 | 0.612 |
| band-wide | 0.25 | 0.641 | 1780.1 | 0.315 | 0.599 |
| band-wide | 0.50 | 0.677 | 1850.8 | 0.303 | 0.577 |
| band-wide | 1.00 | **0.699** | **1885.4** | 0.299 | 0.568 |
| per-bin | 0.50 | 0.587 | **1722.0** | 0.323 | 0.615 |

Matched concordance and cost move in **opposite directions, monotonically**. The
matched metric only ever compares within a margin bin, so it is structurally
blind to the cross-bin damage a band-wide blend does. A signal validated under a
conditioning must be *spent* under that conditioning.

Corrected -- reordering only among rows in the same 0.01 V bin, weight 0.5,
parameter-free -- through the real planner:

| planner | arm | total | Δ | early | late | swaps | misses | precision | recall | paired t | W/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | V8 | 2126.53 | — | 764.2 | 1045.8 | 17.46 | 4.00 | 0.313 | 0.577 | — | — |
| old | near-threshold | 2115.21 | **−11.32** | 763.8 | 1041.2 | 17.50 | 4.02 | 0.311 | 0.575 | −0.45 | 13/14 |
| V10 | V8 | 2073.77 | — | 757.2 | 1029.0 | 17.52 | 3.96 | 0.314 | 0.581 | — | — |
| V10 | near-threshold | 2065.83 | **−7.94** | 750.4 | 1035.4 | 17.48 | 4.02 | 0.311 | 0.575 | −0.42 | 11/15 |

The first arms of this whole branch to land on the *right* side of zero, and
both are noise (|t| < 0.5, wins and losses even).

### Why a real, transfer-stable signal converts to nothing

Measured on the same rows:

* the band holds **29.8 rows per scenario** and **3.22 rows per 0.01 V bin**;
* at the decision boundary -- V8 ranks 12 to 23, where the served/deferred line
  actually falls -- only **36.4 %** of rows have *any* same-bin peer;
* inside the band, V8's ordering already tracks margin at a median
  `|Spearman| = 0.742`.

So roughly two thirds of the marginal decisions are between batteries at
*different* margins, where this signal was never validated and where the
conditioning forbids it from acting; and in the third where it can act, it is
reordering about three rows. **The signal is real and its domain is too narrow to
move the plan.**

## 6. Verdict

```
The misspecification is real. The observable that resolves it is narrow, and it
is not a state variable of the process either (section 7).
```

`m -> 0 implies P -> 1` is a genuine defect, and at matched margin V8 really does
retain almost nothing (0.574). A new, transfer-stable signal exists -- recent
trajectory volatility against the device's own baseline, five of five building
folds -- and it does not resolve the ambiguity where the planner needs it
resolved, because the planner's marginal choice is mostly between different
margins.

What this does *not* say is that Task 1 is closed. It says the second structural
attempt has also failed, on a specific and measured mechanism, and that anything
further has to separate batteries at **different** margins, which is where V8's
remaining ordering error actually lives.

---

## 7. The dynamic-Wiener gate: does the ratio predict the *increments*?

The natural follow-up, and the right one: rather than spending the volatility
ratio as an external reranker, put it inside the process. V8's whole advantage
over every classifier this project has tried is sample efficiency -- drift and
scatter are learned from hundreds of thousands of observed voltage windows
rather than from 82 EOL events -- so if `std_ratio_30_180` is real health-state
information it should appear in the quantity those regressors actually predict.
And `docs/task1_investigation_findings.md` measures within-device 42-day
volatility at about 0.041 V against roughly 0.021 V of drift: **crossing is a
noise-driven event**, so sigma dominates the passage formula and the scatter
model is precisely where a dynamics signal would pay.

The gate, before building any variant: does adding one column to the increment
design improve out-of-building prediction of

    drop(h) = margin(t) - margin(t + h)      and      |drop - E[drop]|

`tools/fj_increments.py`, stride 8 / max_iter 150 (the `transfer_stress.py`
fidelity convention), 5 building folds, base against base-plus-one-column on
identical rows.

### It fails, on both targets

**V8's own target** (windows must end before the crossing), 273,682 windows:

| target | mean relative MAE gain | folds improved | per fold |
|---|---:|---:|---|
| drift | +0.55 % | 4/5 | −0.33 / **+2.40** / +0.19 / +0.14 / +0.35 |
| scatter | +0.57 % | 3/5 | −0.40 / **+2.50** / +0.38 / −0.10 / +0.47 |

A mean carried entirely by one fold, with the other four inside ±0.5 %.

**And that much is an artefact of survivor conditioning.** V8's target excludes
any window that ends past the crossing, so near the barrier the training
population is *the batteries that did not cross* -- exactly the censoring V10
identified ("excluding those windows censors the steepest observed drops...
biases the drift shallow at the knee"). Asking about near-barrier dynamics on
that population conditions on survival. Repeating the gate with V10's
censor-aware target (`--censor-aware`, 275,951 windows):

| target | mean relative MAE gain | folds improved | per fold |
|---|---:|---:|---|
| drift | **+0.05 %** | **1/5** | −0.50 / +1.20 / −0.34 / −0.01 / −0.08 |
| scatter | **−0.05 %** | 3/5 | −0.77 / +0.59 / +0.10 / +0.16 / −0.31 |

Once the conditioning is removed the gain evaporates. By margin band the sign is
the wrong way round for the hypothesis -- the ratio *hurts* drift prediction
exactly where the decision is:

| margin band | mean drift-MAE gain | folds improved |
|---|---:|---:|
| 0.00 - 0.05 V | **−7.5e-4** | 3/5 |
| 0.05 - 0.10 V | **−1.2e-3** | 3/5 |
| 0.10 - 0.20 V | +2.7e-4 | 3/5 |
| 0.20 V and above | +2.7e-5 | 1/5 |

### What the ratio actually is, empirically

Realised 42-day behaviour by volatility-ratio quintile, censor-free windows:

| margin band | n | mean drop, q1 -> q5 | sd of drop, q1 -> q5 |
|---|---:|---|---|
| 0.20 V and above | 32,641 | +0.0071 -> **+0.0277** | 0.039 -> 0.052 |
| 0.10 - 0.20 V | 1,382 | −0.0120 -> +0.0205 | 0.105 -> 0.103 |
| **0.00 - 0.10 V** | **396** | −0.012 / −0.034 / +0.003 / −0.015 / −0.012 | no order |

In the healthy population the relationship is strong, monotone and physically
sensible: the top volatility quintile falls about four times as fast as the
bottom. **Near the barrier there is no relationship at all.** And in the healthy
population the drift regressor already sees it -- `v_std_30`, `v_std_rise` and
seven slope windows are in the design -- which is why the explicit column buys
+0.05 %.

### Verdict

```
GATE FAILED. No dynamic-Wiener variant built, no planner run spent.
```

The pre-registered rule was "if the new feature does not improve the increment
model out of fold, stop before planner experimentation". It does not.

This resolves the apparent tension in §4-5 rather than deepening it. The ratio
separates EOL cases from matched-margin survivors in 5 of 5 building folds --
that is a statement about the *tail*, whether a path touches the barrier. It does
not improve prediction of the *central* dynamics, drift or dispersion, near the
barrier. Both can be true, and together they say the ratio is not a state
variable of the process there; it is a weak correlate of crossing that the
process representation cannot absorb.

Independently, `docs/task1_investigation_findings.md` reaches the same place from
twelve other channels: *voltage is the integral of everything that has happened
to the cell, and by the time it matters for a 42-day forecast the state is
effectively one-dimensional.*
