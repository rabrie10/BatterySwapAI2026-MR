# What actually distinguishes V8's wasted swaps from its useful ones

_Branch `claude/final-j2w-precision`, 2026-08-23. Every number here is measured
out of fold by building on the 19,890 cached scenario rows
(`tools/build_scenario_frame.py`), reproducing V8's decision probability
exactly: `max |p - cached| = 0.0`. Tools: `tools/fj_frame.py`,
`tools/fj_derived.py`, `tools/fj_populations.py`, `tools/fj_signals.py`,
`tools/fj_screen.py`._

---

## 0. The one measurement that reframes the problem

Restrict to the batteries a scenario might plausibly touch -- the top 25 by V8
probability, about 25 of 414 alive devices -- and ask how often a signal puts a
battery that really died inside 42 days above one that survived, **inside the
same scenario**. Chance is 0.500.

| signal | within-scenario concordance |
|---|---:|
| **`voltage_compensated`** (measured V minus `0.00463 x (T - 20)`) | **0.6408** |
| `margin` = raw `smooth_v - 2.4` | 0.6324 |
| `crossing_30` (margin / -slope30) | 0.6246 |
| `voltage_min` (running minimum) | 0.6227 |
| **V8's own probability** | **0.6152** |
| `rel_slope30_room` (slope against roommates) | 0.6091 |
| `p28 / p42` (CDF shape) | 0.6023 |
| `days_below_2.50` (dwell) | 0.5990 |
| `rel_margin_room` | 0.5989 |
| `season_sin` | 0.5925 |
| `slope_30` | 0.5895 |

**Inside the population where the swaps are spent, the temperature-corrected
voltage is a better within-scenario ranker than the entire first-passage model
built on top of it.** The drift regressor, the volatility regressor and the
closed-form passage law together order the candidates *worse* than one
subtraction does.

That is not an argument against the passage model -- it produces the levels, the
horizon axis and the mass, and the mass is right to 1 % (9.40 predicted against
9.46 realised). It is an argument that the model's *ordering* at the top is
noise-dominated, which is exactly the shape of the public evidence: V8 predicts
the right amount of risk and puts it on the wrong batteries.

**The mechanism.** The first-passage law takes the *measured* margin as the
distance to the barrier. `bsai/features.py` already records that within a device
residual voltage tracks residual temperature at **+0.00463 V/degC, positive in
100 % of 454 train devices**, and that an indoor annual swing of 4.87 degC is
0.023 V, "which near the knee is about two weeks of remaining life". The barrier
at 2.4 V is a chemical state; the reading is not. So the quantity the law needs
is the compensated margin, and the compensated level enters V8 only as one of 64
inputs to a gradient-boosted drift regressor -- never as the barrier distance
itself. The median absolute difference between the two margins is **0.0139 V**,
the 90th percentile **0.0365 V**, on a population whose median candidate margin
is about 0.05 V. It is a large correction exactly where it matters.

---

## 1. Population A -- what V8 actually swapped

From the real planner run (`tools/validate_v6.py --served-out`), 48 scenarios:

| | per scenario |
|---|---:|
| served | 17.46 |
| useful (TP) | 5.46 |
| wasted (FP) | 12.00 |
| precision | 0.313 |

### The waste is extraordinarily concentrated

83 distinct devices carry all 576 wasted swaps, and **the worst ten carry 49.7 %
of them**:

| device | wasted swaps | scenarios ever due | median margin | median V8 p | median dwell below 2.45 V |
|---|---:|---:|---:|---:|---:|
| d_b5b678a3f79f | 48 | **0** | 0.0049 | 0.931 | 65 d |
| d_3d26e12378f1 | 43 | **0** | 0.0291 | 0.689 | 31 d |
| d_c9a2ce794b68 | 38 | **0** | 0.0085 | 0.795 | 26 d |
| d_a85bae19463d | 35 | **0** | 0.0626 | 0.273 | never |
| d_d9d695df1683 | 34 | **0** | 0.0526 | 0.200 | never |
| d_d4b4272d5229 | 29 | **0** | 0.0021 | 0.440 | 21 d |
| d_cfdd8b69ddd4 | 18 | 6 | 0.0477 | 0.136 | 58 d |
| d_70d09dad9888 | 15 | **0** | 0.1103 | 0.007 | never |
| d_3fcf7c5e0255 | 14 | **0** | 0.1467 | 0.001 | never |
| d_9ae8a0552434 | 12 | **0** | 0.1142 | 0.008 | never |

Six devices sit within 0.03 V of the barrier, are swapped in 29 to 48 of 48
scenarios, and **never die**. `d_b5b678a3f79f` holds a margin of 0.005 V for the
whole record. The passage law cannot express this: as `m -> 0` the crossing
probability goes to one whatever the drift regressor says, so the model is not
merely wrong about these devices, it is *certain* and wrong.

### Signal-by-signal, TP against FP

AUC over the served rows (0.5 = no separation):

| signal | TP median | FP median | AUC |
|---|---:|---:|---:|
| `p_late_tail` = p126 - p42 | 0.200 | 0.014 | 0.621 |
| `beta_rise` (within-day dV/dT against own baseline) | 1.081 | 0.973 | 0.648 |
| `knee_slope_vs_history` | 2.637 | 1.809 | 0.644 |
| `beta_30` | 0.0126 | 0.0103 | 0.624 |
| `V8 p` itself | 0.501 | 0.359 | 0.607 |
| `cdf_front_mass` | 0.728 | 0.886 | 0.363 |
| `p07 / p42` | 0.560 | 0.790 | 0.365 |
| `rel_margin_room` | -0.252 | -0.128 | 0.336 |
| `slope_30` | -0.0020 | -0.0014 | 0.346 |
| `observations` | 674 | 829 | 0.274 |
| `age_days` | 786 | 859 | 0.332 |
| `temp_now` | 18.6 | 21.0 | 0.392 |

Read together: **a false positive is a saturated, front-loaded curve on an old,
long-observed, slowly-declining device that sits close to the barrier and stays
there.** A true positive is further below its roommates, declining faster
against its own history, with a rising within-day dV/dT -- the documented
internal-resistance knee precursor -- and a curve that still has mass *after*
day 42.

**Important caveat, and it is the reason most of this list is not deployable.**
Nearly all of those AUCs are *pooled across scenarios*, and the pooled number is
dominated by between-scenario variation: winter scenarios have more deaths and
different feature medians. Measured *within* a scenario, which is the only
comparison an order-only reranker can act on, `beta_rise` and
`knee_slope_vs_history` fall out of the top twenty entirely (section 0). This
is the same trap as `season_sin`, which reaches AUC 0.655 inside the candidate
band and is very nearly constant within a scenario. **Every signal in this
project has to be re-measured within scenario before it means anything.**

---

## 2. Population B -- what V9 would have added

V9's public row is the cleanest hard-negative evidence available: one more swap
per scenario for **zero** extra catches (misses 2.27 -> 2.28, recall 0.761 ->
0.760, early +111).

Locally the same exchange looks *good*. Comparing V8's top 18 with V9's top 18
per scenario, out of fold by building:

* V9 promotes **3.71 rows per scenario** that V8 excluded, realised due rate
  **0.202**, against 0.185 in V8's own top decile;
* V9 drops **3.71 rows** that V8 kept, realised due rate **0.067**;
* over all 48 scenarios the exchange gains **36 dues and gives back 12**.

A 3:1 exchange in the model's favour on train, and nothing on public. This is
not a subtle miscalibration -- the local instrument says the head reorders
correctly and the leaderboard says the reordering did not exist on fresh
buildings. It is the strongest available evidence that **a learned head's
ranking gain is a property of the 24 training buildings**, and it is why the
candidate in this branch has no fitted parameters at all.

---

## 3. Population C -- the misses

4.00 due batteries per scenario are not served. **2.29 of those carry V8
probability below 0.02** -- declared safe, not merely ranked low. Their median
predicted probability is 0.0088 against 0.5007 for the served dues.

Splitting them:

| | median margin | median `p_late_tail` | median `beta_30` | median `v_std_30` | median `slope_30` |
|---|---:|---:|---:|---:|---:|
| all misses | 0.1348 | 0.330 | — | — | -0.0014 |
| visible misses (p >= 0.05) | 0.0989 | 0.538 | 0.0137 | 0.0317 | -0.0020 |

The visible misses are mid-curve knee entries: still 0.10 V from the barrier,
declining, with an elevated within-day response and most of their probability
mass *after* day 42. The invisible ones sit at 0.13-0.20 V with no measured axis
separating them -- `docs/ROADBLOCK_REPORT.md` priced that wall at about 0.90
dues per scenario and this reproduces it.

**So the objective is not "fewer false positives".** Cutting volume is what V19
did, and it converted 1.36 dues per scenario into 403 points of late cost. The
objective is to hand V8's existing high probabilities to different batteries.

---

## 4. Hypotheses tested and rejected

Each was fitted with a pairwise ranking objective on within-scenario ranks,
out of fold by V8's own five building folds, deployed order-only, and priced at
12/15/18/21 swaps per scenario. Base is V8 at k=15 timing **1708.7**.

| hypothesis | best k=15 timing | verdict |
|---|---:|---|
| **cold room / peer contrast** (`rel_margin_room`, `rel_slope30_room`, room and building medians, robust, peer-count carried) | 1760.7 | **rejected.** The direction is right -- a TP sits 0.25 V below its roommates and an FP 0.13 V -- but within scenario it is worth 0.599 concordance against V8's 0.615, and deployed it loses. Fourth independent construction to fail; the three earlier ones are in `docs/HANDOVER.md` section 4. |
| **staleness** (`staleness`, `gap_fraction_90`, `margin x staleness`) | 1723.1 | **rejected.** Median staleness is 0 in every population profiled -- TP, FP, miss alike. The stale subpopulation the v13 notes describe is real but it is a *gating* question (does the dark channel still report?), not a ranking axis, and the gate is a volume change. |
| **dwell / persistence** (`days_below_2.45/2.50/2.60`, log dwell) | 1689.7 | **rejected, and the sign is a trap.** Conditional on V8 being *confident* (p > 0.8) longer dwell strongly predicts survival -- realised 0.450 at dwell 0-50 d against **0.163** at 50-150 d, which is the zombie effect. Pooled over the whole candidate band the sign reverses, because dwell also measures how far down the curve a device has come. A single monotone weight cannot hold both, and the interaction that could is the model `docs/PAIRED_SELECTION.md` already shipped as V19's demotion rule -- publicly falsified. |
| **CDF shape** (`p07/p42` ... `p35/p42`, front mass, late tail) | 1707.9 | **rejected as a standalone.** It separates TP from FP well (AUC 0.365 for `p07/p42`) but almost all of that is the saturation the compensated margin already re-orders; on top of the compensated blend it adds nothing (1668.1 against 1609.2 for the blend alone). |
| **within-day knee** (`beta_rise`, `beta_30`, `knee_slope_vs_history`) | 1785.0 | **rejected.** Best pooled TP-vs-FP separator in the table and worthless within scenario -- the classic between-scenario artefact. |
| **season** (`season_sin/cos`) | 1741.5 | **rejected by construction.** Nearly constant within a scenario, so it cannot reorder anything. Its pooled AUC of 0.655 inside the candidate band is entirely between-scenario incidence, which `bsai/calibrate.py` already corrects along the confounded remaining-observation axis. |
| **V9-head disagreement** as a negative warning | — | **not built.** Population B shows the head's local reordering is a training-building property; using its disagreement as a signal imports exactly the thing that failed. |

---

## 5. What survived

**Rank-average V8's own ordering with the ordering by temperature-compensated
voltage, and hand V8's own curves out in the new order.** No fitted parameter,
no learned probability, no building term, no new data. Measured on the cached
frame, out of fold by building:

| k | V8 precision | blend | V8 recall | blend | V8 timing | blend | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.363 | 0.379 | 0.460 | 0.480 | 1802.2 | 1686.6 | **-115.6** |
| 15 | 0.347 | 0.358 | 0.551 | 0.568 | 1708.7 | 1619.3 | **-89.4** |
| 18 | 0.322 | 0.341 | 0.612 | 0.650 | 1738.2 | 1609.2 | **-129.0** |
| 21 | 0.302 | 0.314 | 0.670 | 0.696 | 1770.5 | 1670.0 | **-100.5** |

Precision *and* recall improve at every operating point, which no generation
since V9 has managed and which V9 did not manage on public.

Isolating the temperature term with an otherwise identical control that ranks on
raw voltage: at k=18 the raw-voltage blend reaches 1710.9 and the compensated
one 1609.2, and per temporal block the compensated version wins 5 of 6 head to
head. **About a third of the gain is shrinking the ordering toward the physical
state; the other two thirds is the temperature correction.**

Robustness (details and the planner-level numbers in
`docs/FINAL_J2W_RESULTS.md`):

* six of six non-overlapping temporal blocks improve;
* 35 wins / 6 losses / 7 ties over the 48 scenarios, mean -129.0, sem 38.7,
  t = -3.33;
* four of the five adversarial building groups improve, `hard_hirate6`
  regresses by +42 on its own rows.
