# The 42-day decision-focused residual reranker: three objectives, two planners

_Branch `claude/final-j2w-precision`, 2026-08-23. Code: `bsai/residual.py`,
`tools/fj_residual.py`, `tests/test_residual.py` (13 tests). Every number is the
real `CompetitionPlanner` and the official `evaluate_plan` over all 48 train
scenarios, out of fold by building. Lower is better._

The last modelling question left open by `docs/FINAL_J2W_RESULTS.md`: V8 ranks
the batteries a plan might touch at within-scenario concordance 0.6152 and one
temperature-compensated subtraction reaches 0.6408, so is there a *learned*
residual, trained on the 42-day decision rather than on remaining life, that
does better -- and does the answer depend on the loss?

**No, and no.** All three objectives lose to V8 on both planners, in the same
order on each, and the two pointwise ones lose significantly.

---

## 1. Construction

**Landmarks.** One per (scenario cutoff, battery), from the cached scenario
frame that reproduces V8's decision probability exactly (`max |p - cached| =
0.0`). Restricted to the top 40 by V8 probability per scenario, which is where
the ~17.5 swaps are actually spent.

| | count |
|---|---:|
| landmark rows | **1,705** (35.5 per scenario) |
| positives -- EOL record inside 42 days | **373**, from **75 devices** |
| **excluded as censored before the horizon** | **215** of the 1,920 in the pool |
| ambiguous pairs (within scenario, V8 logit gap <= 2) | **3,902**, from 73 due devices |

A positive is an EOL record inside the window. A reliable negative is *observed*
to survive it -- a record later than 42 days, or 42 days of observation still to
come. A row whose window closes first is neither, and is dropped rather than
called safe; labelling those as negatives is the systematic noise that would
teach the model to like the closing scenarios, where precision is already worst.
`tests/test_residual.py` pins each of those cases.

**Weights come from the official cost model, on training EOL data only.** For
each landmark, using the published rates and the placement the planner actually
uses (median swap on day 1, first emergency slot day 48, both measured in
`tools/swap_ledger.py`):

```
effective = the day the evaluator prices EOL at: the record, else the substitute
served    = 0.5 * max(effective - 1, 0)  +  10 * max(1 - effective, 0)
deferred  = 10  * max(48 - effective, 0)
value     = deferred - served
```

which gives +428 for a battery dying on day 5, +270 on day 20, +50 on day 41,
−49.5 for one whose substitute end of life is 100 days out and −149.5 at 300.
Median service value: **positives +281.0, negatives −70.5**. No leaderboard
number and no hidden target is used anywhere.

**Capacity.** `f(x) = logit(p_V8) + w . rank(x)`, `w` linear over **eight**
signals, each reduced to its within-scenario percentile rank: temperature-
compensated voltage, running minimum, linear crossing estimate, compensated
30-day slope, `beta_rise`, CDF front-loading `p07/p42`, room-relative margin,
and log dwell below 2.45 V. Ranks delete anything constant inside a scenario --
season, the calendar, the remaining-observation window -- so the model can move
*order* and cannot move *volume*. Five building-disjoint folds, V8's own.

**Deployment is order-only.** The score reorders; `bsai/rerank.py` then hands
V8's own CDF curves out in the new order. Verified per scenario for all three
objectives: `np.allclose(np.sort(candidate_p), np.sort(v8_p))` and
`np.isclose(candidate_p.sum(), v8_p.sum())` hold, and the reported
`sum p / scenario` is **9.4040** for each, identical to V8's.

## 2. The three objectives

Identical landmarks, identical eight signals, identical folds, identical L2
(0.02), identical deployment. Only the loss differs.

| | mean fitted weights (out of fold) |
|---|---|
| **cost** -- 42-day binary log-loss, each landmark weighted by \|service value\| | vcomp +0.57, vmin +0.57, crossing +0.21, slope +0.28, beta_rise +0.27, p07/p42 −0.50, room +0.53, dwell −0.03 |
| **focal** -- the same log-loss with a `(1-p_t)^2` modulator and a class prior, no cost weight | vcomp +0.73, vmin +0.72, crossing +0.19, slope +0.22, beta_rise +0.17, p07/p42 −0.68, room +0.56, dwell −0.33 |
| **pair** -- weighted pairwise logistic ranking, within scenario, V8-ambiguous pairs only, each pair weighted by the service-value gap | vcomp +0.03, vmin +0.08, crossing −0.05, slope +0.21, beta_rise +0.30, p07/p42 −0.04, room +0.10, dwell +0.27 |

The two pointwise objectives agree with each other and load the compensated
barrier and the room contrast. The pairwise objective learns something different
-- `beta_rise` and dwell -- and stays much smaller.

### The capacity sweep says there is nothing to fit

`python tools/fj_residual.py --sweep`, within-scenario concordance on the
landmarks (V8 = **0.7280**):

| L2 | objective | sum \|w\| | out of fold | in sample |
|---:|---|---:|---:|---:|
| 0.20 | cost / focal / pair | 0.8 / 1.0 / 0.2 | 0.7258 / **0.7302** / 0.7266 | 0.7320 / 0.7320 / 0.7312 |
| 0.05 | cost / focal / pair | 2.0 / 2.4 / 0.7 | 0.7240 / 0.7284 / 0.7260 | 0.7340 / 0.7321 / 0.7329 |
| **0.02** | cost / focal / pair | 3.0 / 3.6 / 1.3 | 0.7198 / 0.7247 / 0.7246 | 0.7342 / 0.7289 / 0.7352 |
| 0.005 | cost / focal / pair | 5.2 / 6.1 / 2.8 | 0.7082 / 0.7136 / 0.7236 | 0.7327 / 0.7229 / 0.7386 |
| 0.001 | cost / focal / pair | 9.8 / 9.9 / 5.9 | 0.6959 / 0.7031 / 0.7256 | 0.7321 / 0.7180 / 0.7421 |
| 0.000 | cost / focal / pair | 41.7 / 29.9 / 21.4 | 0.6873 / 0.6956 / 0.6975 | 0.7294 / 0.7175 / **0.7444** |

In sample every objective improves on V8, by between 0.001 and 0.016. **Out of
fold, exactly one of eighteen fits beats V8 at all** -- focal at L2 0.20, by
0.0022, with a residual so small it is nearly the identity -- and the in-sample
minus out-of-fold gap widens monotonically as the regulariser is released. That
is the signature of a model with nothing generalizable to learn, on 373
positives from 75 devices with 85 % scenario overlap.

L2 **0.02** was fixed for the planner league before any planner run: it is the
setting at which the three objectives actually differ from each other, so the
comparison prices the loss rather than the shrinkage.

---

## 3. The two planners

* **old** -- the shipped V8 submission config: solver 0.5 s, 80/35 local search,
  4 stratified emergency samples.
* **V10** -- the planner half of the V10 submission: `robust_emergency_samples=0`
  (deterministic expected cost, one replay per evaluation instead of five) and a
  240/240 search, solver 1.0 s. **Its expected-due swap budget is deliberately
  left off**: that is a volume knob, and volume is exactly what must not move
  here. The public A/B credits the mechanics with **−111** on the operational
  components and charges the forecast +179 separately
  (`docs/V11_TRANSFER_FINDINGS.md`).

**The V10 planner reproduces its public signature locally, on V8's forecast:**

| | old planner | V10 planner | delta |
|---|---:|---:|---:|
| **mean total cost** | **2126.53** | **2070.28** | **−56.25** |
| overtime | 70.59 | 64.49 | −6.10 |
| daily limit | 83.33 | 66.67 | −16.66 |
| weekly limit | 95.83 | 85.42 | −10.41 |
| travel | 40.32 | 39.75 | −0.57 |
| early / late | 764.18 / 1045.83 | 757.10 / 1030.62 | −7.08 / −15.21 |
| served / misses | 17.46 / 4.00 | 17.52 / 3.96 | +0.06 / −0.04 |
| precision / recall | 0.313 / 0.577 | 0.314 / 0.581 | +0.001 / +0.004 |
| runtime | 9.57 s/scen | **15.11 s/scen** | +5.5 |

Selection is unchanged to three decimals; the whole gain is in the operational
components, which is what the public decomposition said. **Runtime is the
catch**: 15.11 s/scenario projects to **26.4 minutes for 96 scenarios**, inside
the 30-minute cap but past `bsai/runtime.py`'s 17-minute soft deadline and close
to its 25-minute hard one, so a large share of the private split would plan at
the degraded config or all-defer. Shipping it would need the governor's margins
re-derived, which is a Task-2 decision and outside this experiment.

---

## 4. The league

| planner | arm | total | Δ vs its own V8 | early | late | swaps | useful | FP | misses | precision | recall | paired t | W/L | runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | **V8** | **2126.53** | — | 764.2 | 1045.8 | 21.46 | 5.46 | 12.00 | 4.00 | 0.313 | 0.577 | — | — | 9.57 s |
| old | cost | 2232.53 | **+106.00** | 784.1 | 1116.9 | 21.57 | 5.31 | 12.11 | 4.15 | 0.305 | 0.562 | +2.67 | 18/30 | 9.24 s |
| old | focal | 2186.26 | **+59.73** | 782.1 | 1088.8 | 21.58 | 5.35 | 12.13 | 4.10 | 0.306 | 0.566 | +1.74 | 16/29 | 9.26 s |
| old | pair | 2155.72 | **+29.19** | 764.1 | 1074.8 | 21.58 | 5.35 | 12.13 | 4.10 | 0.306 | 0.566 | +1.27 | 19/26 | 9.12 s |
| V10 | **V8** | **2070.28** | — | 757.1 | 1030.6 | 21.48 | 5.50 | 12.02 | 3.96 | 0.314 | 0.581 | — | — | 15.11 s |
| V10 | cost | 2157.24 | **+86.96** | 780.3 | 1113.3 | 21.72 | 5.33 | 12.27 | 4.12 | 0.303 | 0.564 | +2.09 | 18/28 | 15.05 s |
| V10 | focal | 2133.21 | **+62.93** | 781.7 | 1074.0 | 21.75 | 5.42 | 12.29 | 4.04 | 0.306 | 0.573 | +1.97 | 17/26 | 15.12 s |
| V10 | pair | 2095.31 | **+25.03** | 759.5 | 1053.3 | 21.69 | 5.42 | 12.23 | 4.04 | 0.307 | 0.573 | +1.32 | 22/23 | 14.97 s |

Operational components per scenario:

| planner | arm | overtime | daily | weekly | travel |
|---|---|---:|---:|---:|---:|
| old | V8 / cost / focal / pair | 70.59 / 72.73 / 70.46 / 72.27 | 83.33 / 83.33 / 79.17 / 77.08 | 95.83 / 108.33 / 100.00 / 100.00 | 40.32 / 40.57 / 39.41 / 40.78 |
| V10 | V8 / cost / focal / pair | 64.49 / 63.88 / 64.10 / 64.17 | 66.67 / 56.25 / 64.58 / 64.58 | 85.42 / 77.08 / 83.33 / 87.50 | 39.75 / 40.13 / 39.43 / 39.72 |

### Reading it

**Every arm is worse, on both planners, in the same order.** `pair` +29.2 /
+25.0, `focal` +59.7 / +62.9, `cost` +106.0 / +87.0. The two planners are an
independent replication of the ranking, which is worth more than either run
alone: the result is a property of the model, not of a planner configuration.

**The failure signature is neither of the two the brief names.** It is not
V9-like -- swaps barely move (21.46 -> 21.57, 21.48 -> 21.72) and there is no
volume inflation. It is not V19-like -- early cost does not fall. All four
quantities move the wrong way at once: **early up (+16 to +23), late up (+23 to
+83), precision down (−0.007 to −0.011), recall down (−0.008 to −0.017)**. That
is simply a worse ordering. The success signature the brief asks for --
substantially lower early cost and higher precision at preserved recall -- does
not appear anywhere in the table, at any regularisation, under any loss.

`pair` comes closest to neutral because it barely moves: `sum |w| = 1.3` against
3.0 and 3.6, its early cost is flat to a tenth of a point (764.07 against
764.18), and its whole penalty is +29 of late cost. **Restricting to
V8-ambiguous pairs and weighting by the service-value gap is the least
destructive of the three, which is consistent with it being the objective best
matched to an order-only deployment -- but "least destructive" is not "useful".**

**The pointwise objectives lose significantly**, at t = +2.67 / +2.09 (cost) and
+1.74 / +1.97 (focal) on 48 paired scenarios. The cost weighting is the more
damaging of the two: weighting by |service value| puts most of the mass on the
few landmarks with large positive value, which are exactly the 373 positives
from 75 devices, and the resulting fit is a memorisation of those devices.

### The screen that would have predicted this

Out-of-fold within-scenario concordance on the landmarks tracks the planner
here, where `tools/rank_lab.py`'s top-k pricing did not:

| arm | OOF concordance | Δ vs V8 | old planner Δ | V10 planner Δ |
|---|---:|---:|---:|---:|
| V8 | 0.7280 | — | — | — |
| pair | 0.7246 | −0.0034 | +29.19 | +25.03 |
| focal | 0.7247 | −0.0033 | +59.73 | +62.93 |
| cost | 0.7198 | −0.0082 | +106.00 | +86.96 |

Every arm that lost concordance lost cost, and the arm that lost the most
concordance lost the most cost, on both planners. Compare with the
compensated-barrier candidate of `docs/FINAL_J2W_RESULTS.md` §6, where the top-k
screen predicted −129 and the planner delivered +23. **Concordance is the screen
to use; top-k is not.** It is still only directionally right -- `focal` and
`pair` are indistinguishable on it and differ by 30 through the planner -- so it
prunes, it does not decide.

---

## 5. Verdict

```
REJECTED -- all three objectives. V8's ordering stands, and nothing is enabled.
```

The 42-day decision framing was the strongest remaining argument for a learned
residual: correct censoring-aware labels, evaluator-derived utilities, hard
examples at the incumbent's own boundary, eight parameters, and an order-only
deployment that cannot repeat V9's or V19's volume failure. It still loses. In
sample the signal is worth 0.001-0.016 of concordance; out of fold it is worth
nothing, on 373 positives from 75 devices.

Together with the flexible-model bound already on record -- a 200-iteration
gradient-boosted classifier on 72 within-scenario ranked signals reaches
concordance **0.5567** against V8's 0.6152 -- the conclusion is not about model
class or loss function. **At this sample size the within-scenario ordering at
V8's decision boundary is not learnable, and Task 1 is closed.**

The one measurement here that is worth acting on is not a Task-1 result at all:
**the V10 planner mechanics are worth −56.25 with V8's forecast and V8's
selection**, reproducing the public −111 signature, at a runtime that would need
the governor re-derived before it could ship.
