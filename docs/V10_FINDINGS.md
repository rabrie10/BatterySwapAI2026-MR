# V10: what was measured, what shipped, what died

Written 2026-08-22 after a full-day push from the de261f5 baseline (2145.1 local
out-of-fold; 2078.28 public). Everything below is out-of-fold by building over
all 48 train scenarios unless stated. Noise floor ~100 on the 48-scenario mean.

## What shipped (local 1997.5, −149 vs baseline; rerun for stability below)

| piece | local effect | why it earns its place |
|---|---:|---|
| **Censor-aware drift targets** (`bsai/wiener.py::build_increment_targets`) | −84 alone | Training windows previously had to end before the crossing, censoring exactly the steepest drops out of the drift fit — the model was systematically shallow at the knee. Windows now may end past the crossing (observation continues a median 204 d past EOL); a window containing the crossing counts at least the full margin, which guards against post-replacement recovery. Stride-4 PR-AUC 0.4303 → 0.4706; misses 3.92 → 3.46/scenario; late −112. |
| **Deterministic expected-cost objective** (`robust_emergency_samples=0`) | −30 | The 4-sample emergency average cost 5 replays per search evaluation and triggered the 35-evaluation "uncertain" budget cut. The deterministic path scores one replay per evaluation and measured better outright. |
| **Search budget 240/240** | −60 (with samples=0, net faster than baseline) | The repair loop was exhausting its budget with limit-hit days unfixed (s_4/s_21/s_23 had 6–8 planned >24 h days). Capacity components all improved. |
| **Expected-due swap budget** ceil(1.6·E[due]+1) (`scenario_planned_swap_limit`) | −56 vs same stack unbudgeted | The audit shows the local score surface is flat in volume (marginal defer 39.9 vs service 41.4) — local validation cannot police swap count. The leaderboard can: every team ahead of us plans <17 swaps/scenario, and our early cost per planned swap ran 48.7 vs the leader's 23.8 at identical recall (7.2 dues caught each). The budget imposes the volume discipline the local metric is indifferent about; local served 18.9 → 15.9 at −56. |

Planner runtime with the shipped settings: ~6.2 s/scenario, ~12 min projected
for 96 scenarios (soft deadline 17 min, hard 25 min unchanged).

## The diagnosis chain that got here

1. **Reliability at scenario cutoffs was anti-monotone at the top**: raw rows
   predicted ≥0.7 realised 0.256 (3.41×), while everything below 0.35
   under-predicted 1.3–5×. The remaining-observation calibration balanced
   aggregate counts (451 vs 454) by inflating opening-scenario probabilities
   2.3×, which corrupted the top of the distribution where wasted swaps are
   most expensive.
2. **Four zombie batteries** (d_b5b678a3f79f, d_c9a2ce794b68, d_3d26e12378f1,
   d_d4b4272d5229) accounted for 141 of the 207 top-bucket rows: volatile
   (59–99 mV std) series with per-device floors at 2.402–2.416 V that kiss the
   threshold for years and never cross. Brownian passage arithmetic reads
   volatility + tiny margin as certainty; survival evidence says the opposite
   (fresh dip below 2.45 → 80% cross in 42 d; 43–90 d of dwell → 18%).
3. **The missed dues are knee-entry cases**: median margin 0.12 V at cutoff,
   failing a median 25 d later; 85% already had within-day dV/dT above 2× the
   fleet median (the IR channel fires), but their increment distribution is
   "probably flat, small chance the plunge starts" — a mixture a
   location-scale Gaussian cannot hold. The censor-aware targets move the mean
   the right way and bought half a due per scenario.

## Measured and rejected this session (do not retry without new evidence)

| tried | result |
|---|---|
| Isotonic reliability map (per remaining band), alone | 2196.7 (+50). Fold-heterogeneous poison defeats p-only maps. |
| Isotonic + dwell adjustment (both models) | 2332.4 / 2248.2. Better probabilities ≠ better plans: served stayed ~18, late +100. The planner's operating point is insensitive to p-values (wide decision margins); reshaping p reshuffles marginal picks and costs catch-luck. |
| Dwell knockdown alone on shipped calibration | 2228.2. The knocked-down small-margin cells include genuine catches; their budget slots refill with worse candidates (late 1354). Honest probabilities and co-adapted economics do not compose freely. |
| Reflection weight 0 (endpoint-only passage) | Does not kill zombies (their −1.5 mV/d trend × 42 d exceeds their margin) and wrecks mid-range calibration (0.27 predicted realising 0.51). Knob kept at default 1.0. |
| Candidate margin −15 (standalone-economics gate) | 2134.7 (−12, noise). The PLAN_V8 "batching leak" is already dead at this baseline: 0.08 swaps/scenario fail the standalone test, total 1.4 points. |
| Calibration boost clamp ≤1.0 | 2273.4 (+126). The boost buys real recall in opening scenarios. |
| Stride 2 / 400 iterations full retrain | 2103.1 (+106 vs ship) and OOF PR-AUC 0.4504 < stride-4's 0.4706. Denser windows do not help; closes HANDOVER §6.2. |

## Honest read of the local↔public map

Local misses (3.6–3.9/scenario OOF) far exceed public misses (~2.3 inferred),
because OOF-by-building on train is harder than production-on-public; local
LATE overstates. Public EARLY is the binding constraint (12.2 wasted swaps ×
~76 pts at identical recall to first place). The budget and volume discipline
are aimed at the public structure; locally they price at −56 but the
leaderboard arithmetic prices the same shape change at several hundred. The
prior transfers held direction (V6→V7 −233 local → −748 public), so direction,
not magnitude, is the claim.

## Largest remaining measured-but-unexploited items

1. **Knee-onset mixture pricing.** Empirical: margin 0.05–0.10 & beta30 ≥ 0.01
   realises 0.17–0.27 vs model mean p 0.07–0.14 (median 0.03–0.08). A
   probability floor keyed on (margin, beta30) — same machinery as
   `DwellAdjust`, opposite sign — is built-adjacent but unvalidated; with the
   budget active it changes budget composition rather than volume.
2. **Block-3 miss cluster** (s_16–23): late ~3000/scenario block mean;
   partially forecast-visibility, partially the s_16 cluster (10 misses of 15).
3. **Per-scenario capacity shaping** beyond the search budget: the top-7 teams
   hold capacity flat at 193–252 across 12–19 swaps; we ship at ~194 local.
