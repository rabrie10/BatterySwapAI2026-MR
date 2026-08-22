# V11: the transfer problem, measured to the bottom

Written 2026-08-22 after the V10 submission scored 2179.06 (prior 2078.28) and a
parallel research push (three agents + integration) dug out why, and what does
and does not survive the local-to-public boundary. Full reports:
`docs/TRANSFER_STRESS.md`, `docs/XAWARE_RULES.md`, `docs/KNEE_FINDINGS.md`.

## The public A/B, decomposed (new − old submission, per-scenario means)

| component | delta | verdict |
|---|---:|---|
| travel −7.8, overtime −18.8, daily −41.7, weekly −8.3 | **−111** | planner mechanics (deterministic objective + 240-eval search) transfer 1:1 with local measurement |
| early +103.7, late +74.8 | **+179** | the censored-drift forecast, deployed uncapped, planned 19.2 swaps/scenario (deduced from battery_swap = 0.25 × swaps and the late quadratic) |

## Why the budget never bound — quantified by true leave-one-building-out

5-fold grouping leaves near-twin buildings in every training fold. Under true
LOO (24 folds, `tools/transfer_stress.py`):

| model | Σp/scen 5-fold | Σp/scen LOO | inflation | deployed estimate | public deduced |
|---|---:|---:|---:|---:|---:|
| v7 | 8.45 | 9.95 | ×1.05 | ~9.4 | — |
| cens | 10.01 | 12.27 | **×1.30** | ~11.6 | ≥11.4 ✓ |

The LOO harness reproduces the public probability level to within 2%. Per-building
Σp error spans ×0.07–×3.15. **No absolute-probability mechanism (threshold or
Σp-scaled budget) survives this; volume must be capped by construction.**

## But the censored model RANKS better out-of-distribution

PR-AUC on 5/5 hard building holdouts: cens 0.428 vs v7 0.391 (and pooled LOO
0.316 vs 0.280). The public +179 was level/volume, not ordering. Hence the ship:
**cens ranking behind a hard cap** (`BATTERYSWAP_MAX_PLANNED=15`), local 2056.5,
14.0 planned swaps/scenario, runtime 5.9 s/scenario.

## Feature fragility (the next model generation's spec)

Per-building dispersion of medians: `beta_30` ×5.8 (CV 0.47), `v_std_30` ×14.6 —
HVAC/duty scale, not health — while `beta_rise` ×1.32 (CV 0.054) and
`v_std_rise` ×1.42 are building-invariant. The worst-transferring holdouts shift
exactly the scale features (KS 0.63–0.66). The drift model's top-10 importances
include four fragile features (temp_now/lifetime/std, age_days). Spec: replace
absolute shape features with rise ratios, estimate the V/°C compensation per
device instead of the global 0.00463, keep absolute margin (real signal).

## Measured and rejected in this generation (planner-validated, 48 scenarios OOF)

| idea | analytic | through the planner |
|---|---:|---:|
| Rank→realized-rate recalibration (flat) | counts sane | 2295 (+233): calendar-blind, opening misses explode |
| Dwell knockdown, remaining-gated (≥200 d) | top-5 rate 0.51→0.69 | 2155 (+93): cap-slot refill effect again |
| Knee-onset banded floor (margin×beta×remaining≥220) | **+126.8 ± 45.2/scen** | 2076 (+14): noise — fifth probability-layer lever to die in the planner |
| X-aware EV selection rule (`R5_flatg_tail`) | −102 ± 35 vs top-k | its planner twin (candidate margin) already measured noise; edge is count-flex the public data contradicts |

The consistent law of this codebase: **the planner responds to order and volume,
never to probability-level reshaping.** Every calibration-layer intervention
(isotonic, dwell ×4 variants, rank map, knee floor) measured worse or noise
through the planner while looking good analytically.

## Where the ranking actually stands (rate-at-rank, OOF)

Opening block: top-12 realized 0.59–0.60 (first-place class). Mid block
(s_16–31): **0.214** — the model cannot order the knee-entry population, and the
knee miner's floor bought no catches there. Late block: 0.26–0.29, partly
mechanical censoring. The pooled top-12 rate 0.36 vs the leader's implied 0.57
is the remaining forecast gap, and it is concentrated in one third of the year.

## Honest ceiling arithmetic for the next submission

Banked/high-confidence: planner mechanics −111 (measured on public), volume cap
(19.2→≤15 planned) −250 to −400 on early plus −40 to −90 on capacity at a late
give-back of +0 to +100. Central estimate: **public ~1550–1750** from 2078.
Matching first place (1160) additionally requires their mid-block-equivalent
precision — the two documented candidate paths are the rise-ratio invariant
retrain (spec above) and a mid-block ranker; both are cross-building bets that
should ride the transfer-stress harness, not 5-fold OOF, before any submission.
