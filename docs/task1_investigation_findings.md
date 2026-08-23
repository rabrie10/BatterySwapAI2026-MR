
Cloud
/

















































































Task1 investigation findings · MD
# Task 1 (EOL prediction) — independent investigation findings
 
Rebuilt from primary sources (raw dataset + `batteryswap_public` source), not from prior
V6–V10 code. Prior work was used only as a source of hypotheses to test. All numbers below
were measured in this investigation unless marked *(from v10_validation_report.json)*.
 
Evaluation protocol used throughout: features strictly truncated at the scenario cutoff;
within-scenario AUC with the direction fixed globally; recall at each scenario's actual
swap count; chronological block stability over 6 blocks; **42 fully observable scenarios only**.
 
**Headline conclusion: margin (smoothed voltage above 2.4 V) is close to a *sufficient
statistic* for this problem. Twelve independent signal channels were tested against it; none
produced a transferable ranking gain above the noise floor. See §6.**
 
---
 
## 1. Verified facts (safe to build on)
 
**Pipeline.** `smooth_series` reproduced to `maxdiff = 0.00e+00` on 8 devices with matching row
counts. The exact rule: daily grid spans the first→last day with ≥1 reading in 10 < T < 30;
values are the daily median of those readings, NaN unless ≥5 of them; then
`rolling(7, min_periods=3).median()`. The grid boundary is set by the *temperature-filtered*
subset — getting this wrong fabricates values past the last usable observation.
 
**Label.** "Due" = recorded EOL within (cutoff, cutoff + 42d]. Reconstruction gives
**9.458 due / scenario** against the prior pipeline's 9.46. Mean active fleet **414.4**,
due rate 2.28%. Active set = not (`eol <= cutoff`); the `start_time` filter is a no-op.
 
**The label is clean.** Zero censored devices end at or below 2.40 V. Recorded-EOL devices
end at median 2.308 V, censored at 2.697 V. There is no meaningful pool of unrecorded deaths —
false positives are genuine model errors, not label artefacts.
 
**Censoring.** 82 / 461 devices ever cross (17.8%). That is the hard information ceiling for
any label-supervised approach; synthetic cutoffs multiply rows, not events.
 
**EOL is a service threshold, not failure.** Devices keep reporting normally for months below
2.4 V (one observed 5 months down to 2.25 V). The target is *"when does this specific smoothed
transform cross an administrative line"*, which is why temperature is part of the target rather
than noise, and why battery-physics state estimation is largely beside the point.
 
**Observability.** Only 42 of 48 scenarios have a fully observable 42-day window; s_42–s_47 are
truncated. Including them flatters the mean cost by ~190 points (1986.7 over 48 vs **2176.5**
over the 42) and inflates recall (V10: 0.601 over 48, **0.582** over the 42).
 
**Prediction-time visibility.** `iterate_scenarios` removes already-dead devices from *both*
`locations` and the timeseries. The planner therefore cannot observe that a neighbour crossed;
only living devices are visible. This constrains any peer/cohort feature (§5).
 
---
 
## 2. The cost structure — this is the most important section
 
*(from v10_validation_report.json, out-of-fold, 48 scenarios, mean 1986.7)*
 
| component | cost | share |
|---|---:|---:|
| **late_swap** | 986.04 | **49.6%** |
| **early_swap** | 740.46 | **37.3%** |
| weekly_limit / overtime / daily_limit | 190.93 | 9.6% |
| travel + building + room + swap | 69.27 | 3.5% |
 
**Timing penalties are 86.9% of score; operational work is 3.5%.**
 
**`late_swap` is 100% missed batteries.** Every scenario with `missed = 0` has
`late_swap = 0.0` exactly (s_44–s_47). There is essentially no cost from mistiming a correctly
identified battery — the planner already places days safely early. A distribution-valued
Task 1 → Task 2 contract therefore cannot help through the *timing* channel.
 
Derived economics:
- cost per missed battery ≈ **261**
- cost per wasted swap ≈ **71**
- break-even: swap if **P(due) > 71/(261+71) ≈ 21%**
- value of catching one more due battery ≈ **245 points**
**The planner is already at the volume optimum.** At K=16 the marginal candidate sits near
margin 0.06 where the empirical due rate is 0.174:
`0.174 × 261 − 0.826 × 71 = −13.3`. Slightly negative. So better *count* estimates cannot
help much — which independently explains why prior count-model work (MAE 2.85 → 2.36) had
little end-to-end effect. **Volume is not a lever. Ranking is the only first-order lever.**
 
Cost is also highly concentrated: top 5 scenarios = 22% of total, top 10 = 38%.
Worst are s_16 (4974, 11 missed of 15 due), s_23 (4554), s_15 (4200).
 
---
 
## 3. Process structure
 
- **The knee is at ~2.95 V, not near the threshold.** Drift is −0.22 mV/day on the plateau
  (3.00–3.05 V), jumps to −0.88 at 2.90–2.95, then is roughly constant at −0.5 to −0.7 mV/day
  all the way down to 2.4. *Caveat: binning by device-day is length-biased toward slow movers.*
- **Volatility beats drift ~2:1 over the window.** Within-device 42-day σ ≈ **0.041 V** against
  ~0.021 V of drift. Crossing is a noise-driven event, which is why margin dominates slope.
  (An earlier 5:1 figure was wrong — it used pooled σ, which includes between-device spread.)
- **Within-device variance ≫ between-device** (0.0348 vs 0.0088 for 30-day drift), so stable
  per-battery drift estimates are not obtainable.
- **Temperature:** median dV/dT = **+0.0065 V/°C**, positive in 97.4% of devices. Sensitivity
  **quadruples with depletion**: 0.0039 (3.0–3.1 V) → 0.0165 (2.4–2.5 V), monotone over 300k
  device-days. Removing temperature cuts 30-day volatility by only 12%.
- **Season is large and knowable.** Danger-zone due rate ranges **0.032 (July) to 0.335
  (September)** — 10×. Mechanism: September has the steepest forward temperature fall
  (−1.69 °C/30d) and voltage fall (−29.7 mV/30d), of which ~⅓–½ is direct temperature.
  Arithmetic checks: −1.69 × 0.0065 ≈ −11 mV temperature + ~−18 mV depletion ≈ −29 mV observed.
  **But it acts on calibration/volume, which the cost function has already flattened.**
- **Naive Brownian first-passage over-predicts 2× at low margin** (0.772 predicted vs 0.348
  observed at margin 0.02). Cause: dwelling batteries. This is the "zombie" phenomenon.
---
 
## 4. Ranking: what works and what does not
 
**Baselines (recall at each scenario's actual swap count, 42 scenarios):**
 
| rule | recall |
|---|---:|
| V10 full stack | 0.582 |
| margin + 0.00035·dwell (hand rule) | 0.574 |
| margin alone | 0.547 |
| small GBM, 5 features, OOF by building | 0.506 |
| logistic, 2 or 5 features | ~0.465 |
 
Noise floor ≈ 100 points ≈ 0.41 batteries ≈ **0.041 recall**. So V10, the hand rule, and plain
margin are **all within noise of each other**. The entire V6→V10 ranking gain is not
distinguishable from a one-line rule at matched swap counts. (V10's own ledger attributes
−146 of its −149 improvement to planner-side changes, not Task 1.)
 
**Candidate generation (margin threshold → candidates/scenario → recall):**
0.05→10→0.352 | 0.10→29→0.685 | 0.15→53→0.836 | 0.20→80→0.930 | 0.30→142→0.986
 
A Stage-A cut at "top 30–50" therefore discards 16–26% of positives before any reranker sees them.
 
**Headroom:** perfect reranking of the top 40 by margin = recall 0.805 (≈ −473 points);
of margin < 0.2 (80 candidates) = 0.930 (≈ −760 points).
 
### Channels tested and rejected
 
| channel | within-scenario AUC (danger zone) | verdict |
|---|---:|---|
| **margin** | **0.781** | the dominant signal |
| days_below_2.50 | 0.750 | largely margin re-encoded |
| dwell below 2.50 | 0.723 | real conditionally; does not convert |
| room_low_frac / vs_room | 0.635 / 0.621 | inflated by self-inclusion; clean version 0.537 |
| slope_30 / slope_60 | 0.588 / 0.594 | positive AUC, **poisons the top of the ranking** |
| within-day dV/dT | 0.557 | 0.60–0.69 in some strata, no conversion |
| peer distress (self-excluded) | 0.537 | zero conversion (§5) |
| temperature anomaly | 0.528–0.571 | mostly calendar, not battery |
| transmission completeness | 0.516–0.524 | mechanism false; is a *building* property |
| age | 0.553, **inverted** (0.411) at low margin | younger is *riskier* at matched voltage |
| gap_days / staleness | 0.505 | nothing |
 
**Three structural lessons:**
 
1. **Positive AUC ≠ ranking gain.** `margin + 42·slope`, `margin/−slope`, and a temperature+
   seasonal formula all scored *worse* than margin alone (0.488, 0.351, 0.436 vs 0.552 at K=16).
   Recall@16 selects the extreme tail of the score, where a noisy additive term selects noise.
2. **Hand rules beat fitted models here.** ~82 independent positives behind 415 correlated
   rows. A logistic in log-space combines multiplicatively and lets dwell swamp margin; the
   additive hand rule keeps margin in charge and wins by 0.11 recall.
3. **Beware self-inclusion in group statistics.** Room/building aggregates that include the
   battery itself leak its own margin back in and manufacture apparent signal (0.635 → 0.537
   once excluded). Always exclude self before believing a peer feature.
### Why dwell fails despite being real
 
At matched margin, fresh dips cross far more often than long dwellers:
 
| margin band | fresh ≤14d | dwell >43d | ratio |
|---|---:|---:|---:|
| (0, 0.025] | 0.692 | 0.239 | 2.89 |
| (0.025, 0.05] | 0.647 | 0.252 | 2.57 |
| (0.05, 0.075] | 0.375 | 0.129 | 2.90 |
| (0.075, 0.1] | 0.239 | 0.112 | 2.14 |
 
Solid across four bands and thousands of rows. But **demoting a zombie at margin 0.02
(due rate 0.24) frees a slot filled by a margin 0.10 candidate (due rate 0.17). You trade
0.24 for 0.17 — a loss, every time.** Validated gain: **+0.027 recall, positive in 4 of 6
chronological blocks** — below the noise floor. Dwell also fails as an early-cost predictor
(corr with days-to-effective-EOL = 0.172, AUC 0.562).
 
**Suppressing false positives cannot help while the replacement pool is worse than the false
positives. Only *raising* knee-entry cases helps, and no tested channel does that reliably.**
 
### Dead ends worth not retrying
 
- **Past-EOL veto.** 15 devices have an `end_time` earlier than the dataset boundary, so from
  ~s_21 their imputed EOL (`end_time + 30d`) is already in the past and swapping them would
  charge `late_swap` from months back. Correct in principle, worth **zero**: all 322 such rows
  sit at margin 0.231–0.591, far outside any candidate set.
- **Early-cost-aware selection.** Not-due candidates' days-to-effective-EOL has median 154 and
  sd 74, but **79.5% sit on the single global boundary date**, so within a scenario there is
  little per-battery variation to exploit, and nothing tested predicts the residual.
---
 
## 5. The building effect and cohort clustering
 
Conditional on margin, building due rates differ up to **3×** (per-building excess ranges
−69.5 to +23.6 events; sd 22.0 over 16 buildings, against ~370 due in the danger zone).
 
**Crossings cluster within buildings — this is real.** Median within-building gap between
consecutive crossings is **22 days** against a permutation null of 31 d [23, 40], one-sided
**p = 0.014**. 14 of 24 buildings have ≥2 crossings (max 15). Deployment cohorts deplete
together, which is also why the per-scenario due count oscillates rather than drifting smoothly:
it is a bursty arrival process convolved with the 42-day window.
 
**But the clustering is not separately exploitable.** A self-excluded peer-distress feature
(fraction of living building-mates below margin 0.10) scores AUC 0.537 and converts to
**exactly zero** — the optimal coefficient is 0, and all six chronological blocks are +0.000.
The reason is the key insight of the whole investigation: *if cohort-mates deplete together,
their voltages decline together, so each battery's own margin already encodes the cohort state.*
 
**Not explained by any transferable covariate tested:** correlation of building excess with
mean temperature 0.234, with temperature variability −0.072, with dV/dT −0.033 (n=16).
Independently confirmed by the 365-day climate features scoring AUC 0.496 and 0.511.
 
Deploy cohort and building are inseparable (19 of 24 buildings have exactly one deployment
cohort, covering 45.8% of devices). Only 24 buildings exist in total.
 
**Consequences:** building-grouped CV is mandatory, not precautionary; building/room identity,
deploy cohort, and anything correlated with them must never enter a model; and this effect is
an irreducible floor on achievable precision.
 
---
 
## 6. Why margin is hard to beat
 
Twelve channels — slope, curvature, ETA ratio, temperature level, temperature anomaly, building
climate, within-day dV/dT, age, dwell, transmission completeness, peer distress, and fitted
models at three complexities — each either failed outright or turned out to be margin in disguise.
 
The pattern has a physical explanation. **Voltage is the integral of everything that has
happened to the cell**: duty cycle, temperature history, internal resistance, manufacturing
batch, cohort membership. Every one of those causes is real and several were measured here —
but they act *through* voltage, and by the time they matter for a 42-day forecast, voltage has
already recorded them. In the regime that matters, the state is effectively one-dimensional.
 
That explains why margin matches a 51-feature GBM, why it matches V10 within noise, and why
each attempted second axis collapses. It is not a failure to find the right feature.
 
---
 
## 7. Recommendation
 
1. **Stop adding Task 1 features.** Twelve channels, one protocol, no transferable gain.
2. **Keep margin as the ranking score.** Optionally add dwell at a ≈ 3e-4 — directionally right
   and cost-neutral, but do not expect or claim a gain.
3. **Put remaining effort into Task 2**, where V10's own ledger says −146 of −149 came from.
   Specifically: **capacity penalties are 191 points**, 2.7× all operational work combined, and
   are highly concentrated (s_23 alone pays 435 overtime + 800 daily + 300 weekly = 1535 of its
   4554). V10 notes record 6–8 over-limit days left unfixed by the repair loop in exactly those
   scenarios. That is a scheduling problem, not a prediction problem.
4. **Exploit cost concentration.** Top 10 scenarios are 38% of the mean. A fix that only works
   on catastrophic scenarios beats a uniform improvement — s_16 alone (4974) outweighs a
   0.03 recall gain across all 48.
5. **Fix the local metric before optimising against it.** Report on the 42 fully observable
   scenarios; the truncated tail is worth ~190 points of free-looking improvement.
6. **If Task 1 is revisited,** the only target worth attacking is *raising* knee-entry cases
   (margin 0.05–0.15, failing a median 25 days later). Nothing tested here does that, and §6
   suggests the information may not exist in this dataset.
## 8. Cached artifacts
 
`data/processed/daily_raw.parquet` (360,847 rows, float64, per device-day),
`data/processed/smoothed.parquet` (385,153 rows, exact `smooth_series` reproduction),
`data/processed/within_day.parquet` (within-day dV/dT, ranges),
`data/processed/cutoff_features.parquet` (19,890 rows, one per active battery per cutoff,
strictly causal).
 

