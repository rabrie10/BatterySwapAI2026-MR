# Two-phase Wiener with random changepoint (P3-1): build + gate report

_Model engineer, 2026-08-22. Code: `bsai/twophase.py`, `tools/twophase_fit.py`.
Artifacts: `outputs/twophase_series.joblib` (per-device daily margin series),
`outputs/twophase_gates.json` (all numbers below), `outputs/twophase_model_oof.joblib`
(fold-dispatching model, written with `--force` for future work — gates did NOT
all pass). Fit + full gate battery runs in ~40 s._

## VERDICT: NO-GO for the planner validate

Gate ladder (in order, per spec): **g1 PASS, g2 FAIL, g3 FAIL, g4 PASS.**
Stopped at the g2 failure per protocol; `tools/validate_v6.py` was NOT run, so
`outputs/twophase_validation.json` does not exist. Reasoning at the bottom.

## Final model (what was actually built)

Absorbing 2-state filter over each device's smoothed margin, exactly-EM
(changepoint enumeration = closed-form forward-backward for an absorbing
chain), 5-fold grouped by building with GroupKFold like `tools/train_wiener.py`;
per-device causal onset posterior pi_t; prediction = the P3-1 tau-mixture over
`bsai.wiener.first_passage_probability` segments (conditional two-segment
decomposition, NOT the pooled N(mu-mix, var-mix): the pooled form counts paths
absorbed in the plateau as alive at onset and misprices the reflection term;
degradation check mu1=mu2, sigma1=sigma2 -> single-phase law agrees to
max|diff| 0.0026).

Four measured pathologies forced structural additions (each is documented with
its measurement in the module docstring):

1. **Heavy tails** (daily-increment std 5.96 mV vs MAD 0.51 mV): plain
   two-Gaussian EM split on variance, not drift (sigma2/sigma1=3.7, mu2~mu1).
   Fix: jump component in the PLATEAU emission only — keeping the plunge
   pure-core encodes irreversibility (recoveries count against an ongoing
   knee); a shared jump neutralized recoveries and ratcheted pi to 1.0.
2. **Median-hold atom** (35% exact-zero increments): Gaussian collapse
   (sigma->floor, lambda->cap). Fix: hold runs merged into (dm != 0 over dt).
3. **Per-device wiggle heterogeneity** (robust daily scale 0.3–6 mV): second
   state degenerates into "the volatile devices". Fix: causal per-device
   observation scale s_d(t) in the emission (CMVN precedent); drifts stay
   physical.
4. **Knee identity**: unconstrained EM makes state 2 the fleet seasonal
   decline (mu2 -0.6 mV/d). Fix: mu2 ceiling -1.5 mV/d (measured knee drift
   q90 -1.27, median -2.7) + sigma2 <= 1.5*sigma1 (a knee in a 7-day median
   is a steep SMOOTH fall). Both constraints bind.

Passage law (decision side): per-device trailing-42d drift, EB-shrunk (random-
drift Wiener, the literature's unit-to-unit variability); dwell-conditioned
onset hazard (the allowed covariate iteration — margin/dwell instead of beta30,
whose foresight measured 0.151): floor-fresh 2.9e-3/d vs floor-chronic
(>365 d below 2.45) 3.7e-4/d vs mid-margin 2.0e-3 vs high 3.1e-4 — the dwell
table as exposure statistics; measured no-onset floor rebound +0.37 mV/d;
BGK daily-monitoring barrier shift 0.5826*sigma; floor diffusion frame-fitted
per fold (sigma 2.5 mV/sqrt-d, reflection weight 0 — MLE rode the grid edge,
see risks). RemainingCalibration (fit_calibration pattern, per-fold OOF)
bolted on for level (g4).

### Parameter estimates (production fit; fold ranges tight)

| param | value | folds |
|---|---|---|
| mu1 (plateau drift) | -0.197 mV/d | -0.188..-0.210 |
| sigma1 (rel; passage 1.81 mV/sqrt-d) | 1.52 | ~±3% |
| mu2 (plunge drift) | -1.50 mV/d (ceiling binds) | all at ceiling |
| sigma2 (passage) | 2.72 mV/sqrt-d (cap binds) | 2.67–2.82 |
| lambda / sigma_J (plateau jump) | 0.165 / 15.7 (rel) | 0.157–0.181 |
| rho (homogeneous) | 3.7e-4/d | 3.2–3.9e-4 |
| rho bins (floor-fresh/chronic/mid/high) | 2.9e-3 / 3.7e-4 / 2.0e-3 / 3.1e-4 | stable |
| pi0 | ~0 | all ~0 |
| floor rebound | +0.37 mV/d | +0.04..+0.46 (fold 4 unstable) |
| LR 2-phase vs 1-phase | 2214 (same emission family) | — |

## Gate table (scenario frame, 19,890 rows, fold-appropriate params + per-device filtering)

| gate | criterion | result | verdict |
|---|---|---|---|
| g1 dwell (margin<0.02) | p42 falls like 0.80/0.29/0.18 | **0.811 / 0.629 / 0.347** (realized 0.783/0.310/0.175; pi 0.47/0.51/0.15) | **PASS** (monotone, band2<=0.35) |
| g2 zombies | median p42 < 0.3 | **0.394** (incumbent 0.48–0.93; per-device 0.20–0.51; two of five — c9a2ce, d4b427 mid-descent lookalikes — stay >0.45) | **FAIL** |
| g2 genuine | open-block top-12 due rows keep p>0.5 | **1.000** | pass |
| g2 ordering view | zombie top-15 slots/scen | **3.46 vs incumbent 3.75** (audit: 4.0 planned) | flip did NOT materialize |
| g3 PR-AUC | >= 0.45 | **0.384** (raw 0.344) | **FAIL** vs stated bar; NOTE: the stated incumbent reference 0.43–0.47 does not reproduce on this frame — measured incumbent frame AP is 0.267 raw / 0.308 calibrated (0.4706 is the stride-cutoff training population, base rate 4.4% vs frame 2.3%). Relative to the measured incumbent: +0.076 AP |
| g3 mid-block top-12 | report vs 0.214 | **0.292** (incumbent-on-frame 0.214) | +36% ordering gain |
| g3 open-block top-12 | report vs 0.589 | 0.562 (incumbent 0.589) | slightly short |
| g4 level | sum-p/realized in 0.8–1.25 per block | **0.832 / 1.127 / 0.956** (raw 0.775/1.759/1.742; calibration factors 0.35–2.2) | **PASS** |

## Sum-p / budget guard (the audit trap — it FIRED)

| | open 0-15 | mid 16-31 | late 32-47 | mean |
|---|---|---|---|---|
| sum-p mine (calibrated) | 11.02 | 9.65 | 6.28 | 8.98 |
| sum-p incumbent (p_cal) | 10.33 | 9.27 | 8.62 | 9.40 |
| budget mine = min(15, ceil(1.6*sum+1)) | ~15 | ~15 | **~11–12** | 13.42 |
| budget incumbent | ~15 | ~15 | ~15 | 14.31 |

Mean budget shrink 0.90 slots/scen, 12/48 scenarios smaller; concentrated in
the LATE block (~3 slots below the incumbent) where my censor-honest level is
lower. Every prior volume-cut experiment (cap 13, x-banded caps) measured
worse; `--max-planned 15` cannot floor the due-budget side. If this model ever
goes to the planner, the budget's sum-p must come from the incumbent (the
litreview's stage-1 guard) — as a new RANKING column only.

## Why NO-GO despite the ordering gains

The +150..+350/scen estimate rides on one mechanism: the five floor-zombies
vacating ~4 slots for dues sitting 1–5 ranks out. Measured here: their p
halved (0.48–0.93 -> 0.39 median) but their slot occupancy moved only
3.75 -> 3.46 — a fractional flip, because two of the five spend most cutoffs
in genuinely knee-shaped descents this model class cannot distinguish (their
floors sit 1–2 mV below their approach path; only device-level frailty
memory would know). The audit's attribution record says fractional demotions
do not cash through the substitution-saturated planner (DEMOTION-ONLY 2028.6,
worse), and the budget guard fired on top. A 15-min planner run against the
1924–1972 band with ±50–60 reroll noise cannot rescue that combination.

## What the next iteration should take (ranked)

1. **pi and pi-weighted drift as GBDT features** (litreview stage 2): the
   filter's ordering signal is real (mid-12 +36%, AP +0.08 over incumbent on
   the frame) and the GBDT keeps the level machinery + budget.
2. **Per-device onset frailty** (beta-geometric dwell hazard) for the two
   stubborn zombies; the dwell-binned hazard is its 2-bin skeleton.
3. Ranking-column-only deployment with incumbent sum-p for the budget.

## Deviations / honesty notes

- Planned `validate_v6.py --production --model outputs/twophase_model_oof.joblib`
  (the artifact is internally leave-building-out via per-device fold dispatch,
  since `OofHazardModel` does not forward device ids) — NOT run, per gates.
- Floor-law MLE preferred the grid edge (sigma 2.5 mV, w=0): the mixed stalled
  population trades hedging against demotion; the frame-fit is per-fold OOF
  but this instability is a transfer risk flag.
- mu2 ceiling and sigma2 cap both bind; the EM would otherwise re-absorb the
  seasonal-decline regime. LR vs 1-phase remains decisive (2214).
- Fold 4's rebound (+0.04 vs +0.35..0.46) shows the floor population is thin
  under building-holdout — the same fragility class P2 documents.
