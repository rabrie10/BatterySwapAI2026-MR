# Three hypotheses, three killing tests, three negatives

Written 2026-08-22. Baseline for everything here is `de261f5`, out-of-fold by
building over all 48 train scenarios: **2145.16**.

Nothing in this document changed the shipped pipeline. All three hypotheses were
killed by their own cheapest test and none was built. The tools that killed them
are committed so nobody has to rebuild them, and `bsai/aft.py` is kept as a
measured, rejected alternative in the same spirit as `bsai/margin.py`.

---

## 0. Reproduction first

Per `HANDOVER.md` §9, before any comparison:

| step | measured | committed |
|---|---|---|
| `train_wiener.py --stride 4 --max-iter 250` | PR-AUC 0.4303, AUC 0.9823, precision@200 0.635 | identical (only wall-clock differs) |
| `fit_calibration.py --volatility-scale 1.0` | block ratios 0.54 / 1.01 / 1.64 -> 0.78 / 1.08 / 1.31 | identical |
| calibration factors | 0.413 / 0.656 / 0.796 / 1.097 / 1.712 / 2.335 | identical |
| `validate_v6.py` out-of-fold, 48 scenarios | **2145.16** | stated 2145.1 |

The regenerated `models/v7_wiener.joblib` was reverted to the committed blob
after checking that its calibration factors matched to four decimals; the only
difference was joblib byte-ordering.

Baseline detail worth having in one place, because the local picture is *not*
the leaderboard picture:

```
early_swap  763.39   late_swap 1026.46   travel 45.69   overtime 79.52
daily_limit  87.50   weekly_limit 114.58   building_change 14.04
served 17.65   due 9.46   missed 3.94   recall 0.584   precision 0.313
block means [2042.7, 1925.5, 3038.2, 2545.6, 1970.4, 1348.6]  sd 579.9
```

Out-of-fold across buildings, `late_swap` is the larger term (1026 against 763)
and precision is 0.313 against the leaderboard's 0.511. Local is a harder
problem than public, in the opposite direction from the gap decomposition. Any
knob tuned to the local optimum will move toward *more* swaps, which is the
opposite of what the leaderboard says to do. Treat local as a ranking test, not
as an operating-point test.

---

## 1. H1 — is the planner optimising a badly wrong objective? **No.**

**Killing test.** `tools/belief_components.py` splits both the planner's
`_expected_score` and the evaluator's bill into the same buckets, over all 48
scenarios and over the whole fleet rather than the candidate subset. The early
and late halves are recovered exactly by rebuilding the cost tables twice with
one daily penalty zeroed, which works because `build_expected_cost_tables` is
linear in both. The realised side is split by walking `evaluate_plan`'s
transitions and marking everything from the first emergency swap onward; the
split was checked against a hand-computed four-battery scenario and reproduces
the official total to the last decimal.

**Result — the gap is in no single term.**

| | believed | realised | gap | share |
|---|---:|---:|---:|---:|
| early | 400.5 | 763.4 | +362.9 | 30.0% |
| late | 315.8 | 1026.5 | +710.7 | 58.7% |
| operational | 218.5 | 355.3 | +136.8 | 11.3% |
| **total** | **934.8** | **2145.1** | **+1210.4** | |

Correlation between believed and realised total is **0.164** across 48
scenarios. (`tools/belief_v6.py` reports 0.613 on the first 12; the opening
scenarios are the well-behaved ones.)

**The cost formulas are right. The probabilities are on the wrong batteries.**
Two measurements settle it.

First, the branch the planner is *correct* about is priced correctly:

| | believed | realised |
|---|---:|---:|
| early cost of swaps on genuinely due batteries | 32.5 | **36.1** |
| early cost of swaps on not-due batteries | 367.9 | **727.2** |
| late cost from missed batteries | 296.3 | **966.9** |

Where the planner has identified a battery correctly, its expected timing cost
matches the bill. Where it has not, both error terms blow up — and they blow up
because the *count* is wrong, not because the per-battery formula is.

Second, the aggregate probability is calibrated almost exactly, and the split is
not:

| | believed | realised |
|---|---:|---:|
| due batteries among the 17.6 planned | 8.3 | **5.7** |
| due batteries among those deferred | 1.1 | **3.9** |
| **total believed due** | **9.4** | actual due **9.46** |

The model gets the right amount of probability mass and puts it on the wrong
devices. The planner then believes it is catching 8.3 of 9.4 (recall 0.88) and
catches 5.7 (recall 0.60). Every one of the three gap terms follows from that
one fact.

**Verdict: killed.** This is not a bug in the objective, so the 400-500 points
attributed to it in `HANDOVER.md` §6 item 1 are not there to be collected by
fixing the planner. They are the same precision problem, seen from the
optimizer's side. **Do not reopen this as a planner workstream.**

### 1b. One genuine structural defect, recorded but not built

`build_expected_cost_tables` computes each battery's expected position in the
emergency queue as the sum of horizon probability over the **whole fleet**
alphabetically before it. The evaluator's queue only ever contains the batteries
the plan *misses*. So deferring is over-priced by
`late_rate x P_i x (fleet rank - missed rank)`, which biases every decision
toward servicing — the wrong direction for precision.

It is also closed-form to fix, which is worth writing down. Because
`emergency_offset - t >= 0` always holds (the offset is 42-48 days and the event
day is 0-42), the `max(., 0)` never binds and the defer cost is exactly linear
in the rank:

```
defer_i(rank) = late * (P_i * offset - M_i)  +  late * scale * P_i * rank_i
                \_________ base_i _________/
```

with `M_i = sum_t pmf[i,t] * t`. Summing the self-consistent version over the
deferred set `D`, where `rank_i = sum_{j in D, j < i} P_j`, collapses to

```
sum_{i in D} base_i  +  late * scale * (S_D^2 - sum_{i in D} P_i^2) / 2,
S_D = sum_{i in D} P_i
```

which is O(|D|) per search evaluation.

**Its magnitude has not been measured and it is not implemented.** H1's killing
test killed the hypothesis, and the rule is to stop rather than build. Judging by
`believed_late_deferred = 296.3` against a realised 966.9, the correction moves
the belief in the direction that already has the larger error, so it is unlikely
to be worth the noise floor on its own. Anyone picking it up should measure it
before assuming otherwise.

---

## 2. H2 — do binary labels waste the trajectories? **They do, and it does not help.**

The premise is real and the size of it is worth stating: on the same 88,013
cutoffs, the binary 42-day label yields **862 positives**, while a censored
time-to-event target yields **15,581 uncensored event rows** — 18x more labelled
events, median event time 385 days.

**What was fitted.** `bsai/aft.py`: a log-normal accelerated failure time model
whose mean is a `HistGradientBoostingRegressor`, fitted by EM on the Tobit
likelihood (censored rows get the conditional expectation of the latent log-time
given that it exceeds the censoring point, then the regressor is refitted; four
passes). The inverse Mills ratio is evaluated as `exp(logpdf - logsf)` so it
stays finite in the far tail where the long-lived censored rows sit. On synthetic
data at 55% censoring it recovers sigma to 0.427 against a true 0.45 and cuts the
mean bias from -0.556 (censoring ignored) to -0.094.

Everything else was held fixed: the same 64 features, the same folds — read back
out of the Wiener bundle by object identity rather than recomputed, because
`GroupKFold` assigns buildings by row count and this model is fitted on cutoffs
where that one is fitted on increment windows — and the same
`min(horizon, remaining)` censoring clip. Neither model was calibrated.

**Result — strictly dominated at every operating point the leaderboard charges.**

| swaps/scenario | precision Wiener | precision AFT | recall Wiener | recall AFT |
|---:|---:|---:|---:|---:|
| 12 | **0.370** | 0.304 | **0.469** | 0.386 |
| 15 | **0.349** | 0.271 | **0.553** | 0.429 |
| 18 | **0.325** | 0.251 | **0.619** | 0.478 |
| 21 | **0.302** | 0.231 | **0.670** | 0.513 |

Best analytic timing 1420.5 (Wiener, k=21) against 1760.4 (AFT, p>0.2). On the
stride population the same story: PR-AUC 0.2824 against 0.4303, precision@200
0.535 against 0.635.

This is not a precision-for-recall trade; the AFT is worse on both axes by about
0.07 of precision throughout the charged band. It lands almost exactly on the
two-line physics control (precision@12 0.304 against the control's 0.309), which
is the level `HANDOVER.md` trap 1 says means "no signal beyond arithmetic".

**Not a plumbing artefact.** For any row with more than 42 days of remaining
observation the effective horizon is the same 42 days for every device, so
`P(T <= h) = PHI((log h - mu)/sigma)` is a monotone function of `mu` alone and
sigma cannot affect the ranking. The deficit is in the fitted mean, not in the
fitted scale.

**Why it loses, despite 18x the labelled events.** Squared error on log-time
spends the model's capacity across the whole distribution, whose median event
sits 385 days out. The decision needs the shape of the first 42 days. The
first-passage construction targets exactly that, and gets its timing information
from a third source that is larger still: **546,042 observed increment windows**,
which measure the rate at which the margin falls rather than when it will hit
zero. More labelled events, weighted in the wrong place, is worth less.

**Verdict: killed at the ranking stage. No end-to-end run was made**, because
the killing test's stated condition — precision must move at 12-21 swaps — was
met in the wrong direction by a wide margin.

If anyone retries this, the thing to change is the loss, not the label: a
tail-weighted or censored-quantile objective aimed at the 42-day region rather
than at `E[log T]`. That is a different experiment and it has not been run.

---

## 3. H3 — is internal resistance the real state variable? **No. `beta` has no barrier.**

**The stated killing test passes.** On the 82 devices that reached EOL, with
`beta` smoothed over a 14-day trailing window:

* **Monotone:** median Spearman against time **0.811** (p10 0.592, p25 0.729),
  93.9% above 0.5, 1.2% negative.
* **Consistent at failure:** `beta` at crossing has median 0.01242, p10 0.00792,
  p90 0.01761 — **p90/p10 = 2.22**. Not "scattered across orders of magnitude".

Both criteria are met. Both are also met by something that is not a state
variable, so two controls were run before anything was built on them.

**Control 1 — the threshold is vacuous.** End of life is *defined* as the
smoothed voltage crossing 2.4 V. If `beta` largely tracks that voltage, its value
at crossing is tight by construction. Taking each crossed device's `beta` on the
first day its margin falls below a fixed level:

| margin at which beta is read | median beta | p90/p10 |
|---:|---:|---:|
| 0.40 V | 0.00736 | 2.43 |
| 0.30 V | 0.00957 | 3.04 |
| 0.20 V | 0.01242 | 2.36 |
| 0.10 V | 0.01305 | 2.22 |
| 0.05 V | 0.01300 | 2.30 |
| 0.02 V | 0.01282 | 2.27 |
| **at crossing** | **0.01242** | **2.22** |

The spread is flat from 0.40 V of headroom all the way to the barrier. `beta` is
exactly as tightly clustered where nothing is failing as it is at failure — so
the tight value at crossing says nothing about end of life. Worse for the
hypothesis, the median **saturates**: `beta` stops rising once the margin is
inside about 0.1 V. A state variable approaching an absorbing barrier does the
opposite.

**Redundancy.** Spearman between `beta` and the smoothed margin is **-0.7905**
pooled and **-0.7900** as the within-device median, over 335,525 device-days.
`beta` is largely a re-encoding of the quantity the Wiener model already tracks.

**Control 2 — the rise is not seasonal, but it is not specific either.** The
fleet's monthly mean `beta` does vary, 0.00315 in August to 0.00552 in February
(1.75x), so a calendar confound is plausible a priori. Subtracting the surviving
fleet's monthly mean barely moves anything: crossed devices 0.811 -> 0.816,
survivors 0.628 -> 0.646. So the rise is real and not calendar. But **survivors
rise too**, at a median 0.646 — which is what -0.79 correlation with a falling
margin predicts. "beta rises over a device's life" is "voltage falls over a
device's life", restated.

Supporting control: 52.2% of surviving devices reach the median crossing level at
some point, 12.1% end above it, and 5.24% of 293,846 survivor device-days sit
above it.

**Verdict: killed.** `beta` is a correlate with a monotone envelope, not a state
with a barrier. The Wiener process was chosen because the margin is
non-monotonic; `beta` does not supply a monotone alternative, because `beta` is
mostly a function of that same margin. **The rejection of the gamma and
inverse-Gaussian processes stands.**

**Useful residue.** `beta`'s discrimination is *early*: it climbs between 0.40 V
and 0.10 V of margin and then flattens. That is consistent with `bsai/shape.py`'s
AUC 0.871 on exactly the population more than sixty days from crossing, and it
says `beta` is an early-warning feature rather than a proximity-to-failure one.
Which is how it is already used. No change indicated.

---

## 4. What this leaves

All three hypotheses are closed. What survives them is a single, sharpened
statement of the problem:

> The forecast's total probability mass is right to within 1% (9.40 predicted
> against 9.46 realised). Its **ranking** is wrong: out-of-fold across buildings
> the model catches 5.7 of the 9.46 due batteries inside its 17.65 swaps.
> Everything else — the early cost, the late cost, the emergency operations, the
> capacity penalties, the planner's belief gap — is downstream of that one
> number.

So the remaining levers are the ones that change *which* devices rank highest,
and the two cheapest untried ones are unchanged from `HANDOVER.md` §6:

1. **Full-strength retrain** (stride 2, 400 iterations). Never run. It is the
   only lever here that is pure compute.
2. **Peer contrast done properly** — weighted by peer count, pooled to building,
   or used as a post-hoc veto. §4 of `HANDOVER.md` explains why the first attempt
   failed and what to fix first. It remains the only measured source of
   *between-device* information in the project; everything else is within-device.

And one that this work adds:

3. **A tail-weighted survival loss.** H2 shows the extra labelled events exist
   (15,581 against 862) and that `E[log T]` is the wrong thing to do with them.
   Whether a loss aimed at the 42-day region can extract them is untested.

## 5. Tools added

| tool | what it answers |
|---|---|
| `tools/belief_components.py` | Where the planner's belief and the evaluator's bill diverge, per cost component and per probability branch. Start here before blaming the planner. |
| `tools/beta_state.py` | Is `beta` monotone per device, and does it reach a consistent value at failure? |
| `tools/beta_controls.py` | The two controls that make the previous answer mean something: is the threshold vacuous, and is the rise seasonal? |
| `tools/train_aft.py`, `bsai/aft.py` | The censored days-to-EOL model. Measured and rejected; kept so the measurement is reproducible. |

`tools/ranking_v7.py` gains `--no-calibration` and `--volatility-scale`, so two
models can be compared raw.
