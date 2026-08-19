# Task 1 Model Investigation Log

Status: closed 2026-08-19. Companion to `docs/TASK1_IMPLEMENTATION.md`
(which describes *what the system is*); this document records *what we
learned building it*, with the evidence for each claim. Written to be usable
as source material for the final technical report.

Everything below was measured on the train split with the official evaluator
(`batteryswap_public.evaluate`), not estimated. Commands to reproduce each
measurement are given inline or in `docs/BENCHMARK_LOG.md`.

---

## 1. The headline result

| Model | Mean `total_cost` (48 train scenarios) | Swaps/scenario | vs all-defer (3324.68) |
|---|---:|---:|---|
| v1/v2 — Platt calibration, blend after calibration | 6023.61 | 71.1 | 1.81x worse |
| v3 — pooled isotonic, blend before calibration | 4005.33 | 43.1 | 1.20x worse |
| **v4 — horizon-conditional isotonic (SHIPPED)** | **3128.88** | 28.0 | **0.94x — beats it** |
| v5 — v4 + tie-break + negative crossing-days | 3168.27 | 28.3 | 0.95x (rejected) |
| *(reference)* all-defer baseline | 3324.68 | 0 | — |
| *(reference)* oracle EOL ceiling | ~78 | ~13.5 | — |

**v4 is the shipped artifact** (`models/risk_forecaster.pkl`,
`model_version = task1-aft-horizon-isotonic/v4`). v5 was a genuine attempt to
fix Defect 4 (Sec 6) and its reasoning still looks correct, but it measured
1.3% *worse* end-to-end, so it was rejected and both of its changes were
reverted to defaults — `MIN_CROSSING_DAYS = 0.0` and
`tie_break_weight = 0.0` — so the shipped code reproduces v4's benchmark
exactly. Both mechanisms remain in the codebase as documented, opt-in knobs.

That v5 outcome is itself a finding worth reporting: the isotonic-plateau
problem is real and observable (Sec 6), but removing the ties did not by
itself improve cost. The plateau is a *symptom* of the model having genuinely
run out of discriminating signal at the top of its ranking, not an
independent defect — with only 82 events, the top ~30 batteries really are
close to indistinguishable, and forcing an ordering on them adds no
information.

The submitted leaderboard entry (2026-08-18 22:44:25, score **7389.39**,
rank 11/14) used the v1/v2 model. The v4 artifact is roughly **half** that
cost on our local benchmark.

**The single most important lesson:** every one of the four defects below
passed the model's own statistical metrics. Out-of-fold concordance was
0.90-0.91 and out-of-fold Brier looked excellent (0.0167 at 42 days) through
*all* of them. Only end-to-end evaluation against the official cost function
exposed the problem. Prediction metrics were necessary but nowhere near
sufficient.

---

## 2. How the problem first surfaced

The leaderboard component breakdown was the first hard signal. Comparing
Dynamic duo against the mean of the top 10 entries:

| Component | Top-10 mean | Dynamic duo | Ratio |
|---|---:|---:|---:|
| early_swap | 501.4 | 3074.6 | **6.1x** |
| daily_limit | 161.5 | 733.3 | 4.5x |
| overtime | 111.6 | 488.1 | 4.4x |
| travel | 58.8 | 226.2 | 3.9x |
| building_change | 12.1 | 43.4 | 3.6x |
| room_change | 7.2 | 25.6 | 3.5x |
| weekly_limit | 100.0 | 347.9 | 3.5x |
| battery_swap | 4.4 | 14.0 | 3.2x |
| late_swap | 1097.7 | 2436.3 | 2.2x |

Every component was worse, which initially looked like nine separate
problems. It was not. `battery_swap` is a pure headcount (0.25 h/battery), so
14.03/0.25 = **56 swaps per scenario against the top-10's ~17.6**. Because
the evaluator services a deferred-but-due battery as an *individual emergency
visit*, each unnecessary swap also drags travel, building/room changes,
overtime and the 100-point daily/weekly threshold penalties along with it.
One upstream cause — massive over-swapping — cascaded into every column.

The asymmetry that makes this so expensive: swapping a battery that never
reaches EOL in the data costs `0.5 x days_to_proxy_EOL`, which on early train
scenarios is **~182 points per battery** (proxy EOL = `location.end_time +
30 days`, ~364 days out). Deferring that same battery costs **nothing**.
Over-swapping is punished far harder than under-swapping.

---

## 3. Defect 1 — Platt scaling cannot calibrate a saturated model

**Symptom.** Predicted in-horizon failure probability averaged 7-9% against
a true rate of 2.2% (~3.5-4x over-prediction), and predicted
`prob_unobserved_eol` was 0.69 against a true 0.91.

**First hypothesis (wrong).** That the physical-prior blend was inflating
risk. Disproved directly: a blend-disabled variant over-predicted *slightly
worse* (ratio 3.67 vs 3.23), so the blend was not the driver.

**Actual cause.** The fitted Platt calibrator had slope **0.7107**,
intercept 0.0293. A slope below 1 flattens the logit, pulling probabilities
toward 0.5 — which in a rare-event regime *multiplies small probabilities
upward*:

| raw | calibrated | factor |
|---:|---:|---:|
| 0.001 | 0.008 | 7.5x |
| 0.010 | 0.038 | 3.8x |
| 0.050 | 0.113 | 2.3x |
| 0.300 | 0.361 | 1.2x |

**Why Platt could not be repaired by refitting.** Sweeping the slope from
0.711 to 1.5 moved the mean prediction only from 0.0827 to 0.0727 — the
ratio stayed in the 3.1-3.6x band. The over-prediction was not a uniform
shift of small probabilities; it was ~60 batteries/scenario pushed to a
*high* probability against ~9 real events. Platt is a two-parameter
shift/scale of the logit and is structurally incapable of pulling a
saturated cluster down.

**Fix.** Fit isotonic regression as well and select between calibrators by
out-of-fold weighted log loss. Isotonic maps each predicted level onto its
observed frequency, so it *can* squash a saturated high end — verified on a
synthetic case where a cluster predicted at 0.95 with a true rate of 0.15
was correctly mapped to 0.17.

**Result:** OOF log loss identity 0.2039 → Platt 0.0942 → isotonic 0.0851.

---

## 4. Defect 2 — the blend was applied after calibration

The physical crossing-day prior was combined as
`max(calibrated_AFT, physical)` — i.e. *after* the calibration step. Any
value the physical term raised was pushed straight off the calibrated scale
the calibrator had just been fit to produce, silently invalidating it.

**Fix.** Blend on the raw survival curve *before* calibration, in both
training (`run_cross_validation`) and inference (`Task1Forecaster.predict`),
so the calibrator is fit on, and applied to, the same blended distribution.

---

## 5. Defect 3 — one pooled calibration map across a 50x event-rate spread

**Cause.** Calibration pooled all horizons into a single isotonic map. But
the event rate is a strong function of horizon:

| horizon (days) | 7 | 14 | 28 | 42 | 90 | 180 | 365 |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted event rate | 0.32% | 0.64% | 1.30% | 1.92% | 4.01% | 7.66% | 16.45% |

A pooled map is calibrated to the pooled average (**4.20%**). Applied at the
42-day planning horizon (true rate **1.92%**) it over-predicts by ~2x — the
residual 1.53x bias that survived the isotonic fix.

A related sub-defect: calibration originally covered only horizons ≤42 days,
yet `predict()` must evaluate out to `evaluation_observation_end` (up to
~334 days past origin) to split observed-tail from unobserved-EOL mass. That
was pure extrapolation, and explains why `prob_unobserved_eol` read 0.69
against a true 0.91.

**Fix.** `HorizonIsotonicCalibrator` — a separate isotonic map per horizon,
linearly interpolated between them, with calibration horizons extended to
`(7, 14, 21, 28, 35, 42, 60, 90, 120, 180, 240, 300, 365)`.

**Result:** OOF log loss 0.0851 → **0.0783**, best of all four candidates
(identity 0.2039, Platt 0.0942, pooled isotonic 0.0851).

---

## 6. Defect 4 — isotonic plateaus destroyed the model's ranking

This one only became visible by instrumenting *which* batteries the planner
chose to swap.

**Evidence.** For scenario `s_0`, all three swapped batteries had an
identical predicted probability of **0.3104**. In `s_24`, 47 batteries were
swapped (40 of them wasted) all clustered in 0.166-0.359, with the top 12
numerically identical.

**Cause.** Isotonic regression is a *step function*. It maps whole groups of
batteries onto exactly the same probability, collapsing a model with genuine
AUC-0.93 ranking into a handful of discrete levels. Faced with 40 tied
batteries just above its decision threshold, the planner has no basis to
prefer the riskiest few — so it takes the entire block.

**Attempted fixes (v5) — tested and rejected.**
1. A tie-break inside the calibrator: mix a small weight (0.02) of the raw
   score back into the calibrated value. Both terms are monotone in the raw
   score, so this preserves monotonicity and holds the calibrated level to
   within 0.01, while restoring strict ordering inside each plateau. Verified
   to work mechanically: on `s_24` the top-50 went from ~5 distinct values to
   29.
2. `crossing_days_extrapolated` was clipped at a lower bound of **0**, making
   every battery *already below* the 2.4 V threshold numerically identical.
   v5 bounded it at −90 days instead, preserving the ordering between "just
   dipped below" and "far below and still falling". This feature feeds only
   the physical blend (`crossing_days_log` is computed but is not in
   `CURATED_FEATURES`), so the change does not disturb the AFT fit.

**Outcome: 3168.27 vs v4's 3128.88 — 1.3% worse.** Both changes were
reverted to defaults. Even after change 1, the top 12 batteries in `s_24`
remained exactly tied, because their raw blended value was saturated at the
same physical-CDF ceiling; the tie-break can only separate what the
underlying score already separates. The honest conclusion is that the
plateau reflects a genuine absence of discriminating signal at the top of
the ranking rather than a calibration artefact that can be engineered away.
Improving this needs a better-discriminating model (more events, or richer
features such as time-below-threshold and temperature-residualised voltage),
not a better post-processing step.

---

## 7. What was *not* wrong — and why that matters

**The ranking was always good.** Measured over 8 scenarios:

| Metric | Value | Random baseline |
|---|---:|---:|
| AUC | 0.932 | 0.500 |
| hits@10 | 2.9 of ~9.9 | 0.21 |
| hits@20 | 5.0 of ~9.9 | 0.43 |
| hits@40 | 7.4 of ~9.9 | 0.86 |

This is why the fixes worked: the model's *ordering* of batteries by risk
carried real signal throughout, and every defect above was in converting that
ordering into calibrated probabilities. Had the AUC been near 0.5, no amount
of recalibration would have helped and the model would have needed rebuilding.

**The Task 2 planner was not at fault.** A sweep of the sanctioned
`late_risk_multiplier` decision-risk knob showed it is a cliff, not a dial:

| multiplier | 0.2 | 0.4 | 0.6 | 1.0 |
|---|---:|---:|---:|---:|
| swaps/scenario | 0.00 | 0.06 | 0.56 | 22.75 |
| total_cost (16 scen) | 4578.8 | 4585.6 | 4594.8 | **3079.0** |

Anything below ~1.0 collapses the plan to all-defer. The planner was
faithfully optimising expected cost; it was being fed bad probabilities. The
default of 1.0 was kept.

---

## 8. Known limitations (carry into the report)

1. **The local benchmark is in-sample.** The final artifact is fit on all 461
   train devices and then benchmarked on train scenarios. Cross-validation
   (used for model and calibrator selection) is honest — grouped by building,
   so no fold sees its own validation buildings — but the end-to-end
   `total_cost` figures in Sec 1 are *not* out-of-fold. The design spec
   (Sec 8.1) calls for generating OOF forecasts per scenario and evaluating
   Task 2 only against those; that harness was not built. Expect the true
   generalisation gap to be unfavourable relative to these numbers.
   *This is the largest outstanding methodological gap.*

2. **Residual over-prediction remains.** Even at v4, predicted/true in-horizon
   ratio is ~1.48 and the model swaps ~28 against an oracle ~13.5.

3. **Brier alone was actively misleading.** With a 2.2% base rate, predicting
   a constant zero scores 0.0224. The v1/v2 model's "good" 0.0167 was barely
   better than that, and its *calibrated* Brier was 1.97x **worse** than the
   trivial baseline. Any future metric reporting should include the
   trivial-baseline ratio, not the raw score.

4. **`physical_uncertainty_days = 20.0` is a fixed constant**, matched to
   `VoltageTrendForecaster`'s scale (18) and never independently tuned.

5. **Runtime headroom is thin.** A full 48-scenario `script.py` run takes
   **19.91 min** (precisely timed). The official run evaluates public *and*
   private together; if their combined scenario count approaches 2x train's
   48, the 30-minute hard limit is at risk. Per-scenario time is now dominated
   by Task 2's CP-SAT and local search, not Task 1's forecast.

6. **Only 82 unique observed EOL events** across 461 devices (82.2% censored)
   fundamentally limits identification: every AFT covariate had p > 0.25 even
   when fit alone, under a robust sandwich variance estimator that correctly
   accounts for ~461 independent devices behind 48,059 person-period rows.

---

## 9. Method note for the report

The productive debugging loop was:

1. **Measure against the real objective, not a proxy.** The official
   evaluator on all 48 train scenarios, not Brier or concordance.
2. **Instrument the decision, not just the prediction.** Asking *which*
   batteries the planner swapped, and at what predicted probability, is what
   exposed the isotonic plateaus (Defect 4). Aggregate metrics cannot show
   that a model has collapsed into ties.
3. **Compare against trivial baselines.** all-defer (3324.68) and the oracle
   ceiling (~78) bracket the achievable range and immediately revealed that a
   6023.61 model was worse than doing nothing.
4. **Falsify the first hypothesis.** The initial blend-inflation theory was
   wrong and was discarded on evidence within one experiment.

`docs/local_benchmark_log.csv` records every end-to-end run
(commit, model version, all official cost components, scenario count) so any
claim here can be re-derived. Reproduce with
`python tools/benchmark_task2.py --dataset-path data/raw/train --mode real --limit 0 --record docs/local_benchmark_log.csv --label "..."`.
