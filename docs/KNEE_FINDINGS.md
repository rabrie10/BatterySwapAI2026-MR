# KneeBoost: knee-onset probability floor — findings (T3 knee miner, 2026-08-22)

Target: the late pool (~3.6-3.9 missed dues/scenario, ~1000 local points). Raw OOF frame
`outputs/frame_oof_raw_beta.parquet` (19,890 rows, 454 dues, 48 scenarios). All fits
out-of-fold by building (fold map from `outputs/v7_folds.joblib`); label is the
censor-capped due (within min(42, remaining)); floors gated at remaining >= 30 and
scaled by min(1, remaining/42).

## 1. Mining checks reproduce
| population (all remaining) | n | realized | mean p | median p |
|---|---|---|---|---|
| margin 0.05-0.10 x beta30 0.010-0.014 | 415 | 0.169 | 0.073 | 0.028 |
| margin 0.05-0.10 x beta30 > 0.014 | 153 | 0.275 | 0.141 | 0.081 |
| margin 0.10-0.15 x beta30 > 0.010 | 722 | 0.072 | 0.028 | 0.004 |

## 2. Flat floor table (margin x beta30) — fitted, but economically dead
Production floors (shrink k=25 toward pooled 0.086, clip 0.35, >=50 rows & >=10 events/cell):

| margin \ beta30 | [0.008,0.012) | [0.012,0.016) | 0.016+ |
|---|---|---|---|
| [0.05,0.10) | 0.120 (391/48) | 0.220 (248/58) | 0.292 (61/23) |
| [0.10,0.15) | 0.045 (506/22) | 0.094 (328/31) | — (70/8) |
| [0.15,0.20) | — (465/8) | 0.039 (413/15) | — (111/6) |

Analytic EV rule (swap iff p*280 > (1-p)*0.5*(remaining+30) + 30) at evaluator prices
(miss = 10/day lateness vs the padded-Sunday emergency queue + 6 op; wasted = 0.5*(X_eff - 7)
+ 1.5, X_eff = observed d_eol else remaining+30; caught = 0.5*early + 1.5):
**net −12.3 ± 6.5/scenario, 0.00 catches gained, +0.34 wasted/scen** (raw arm −11.6).
Reason: EV break-even p* runs 0.19 (remaining 30) → 0.44 (remaining 300); every flat floor
sits below p* wherever its dues live. Flips only occur in closing scenarios and are all waste.
**Flat (margin x beta) floor: NO-GO.**

## 3. Why: the cell rate is savagely non-stationary along remaining
Hot cells (margin [0.05,0.10) x beta30 >= 0.012), realized by remaining band:
0.09/0.05/0.13/0.10 at 30-220d — **0.46 at 220-290, 0.88 at 290-400**. The pooled 0.22-0.29
averages a 0.05 regime and a 0.87 regime. Mechanism: the opening scenarios harvest the ripe
stock the dataset starts with (same axis and same season/calendar confound documented in
bsai/calibrate.py). Winter-vs-summer inside the pool shows nothing (0.077 vs 0.093), and in
the stock regime winterness points AWAY from due (AUC 0.397) — on train this is stock
harvest, not season.

## 4. Banded floor (margin x beta30 x remaining) — the live candidate
Bands: remaining (30,150,220,+inf); beta merged to 2 bands (0.008,0.012,+inf) because the
top sliver (20 events) cannot fund a per-fold 25-event gate; min 40 rows & **25 events**/cell,
shrink 25 toward the band pool, clip 0.65. Production: exactly two active cells, both rem>=220:

| cell (rem >= 220) | rows/events | rate | floor |
|---|---|---|---|
| margin [0.05,0.10) x beta30 >= 0.012 | 113/62 | 0.549 | **0.488** |
| margin [0.10,0.15) x beta30 >= 0.012 | 136/34 | 0.250 | **0.244** |

Evidence quality: 62 events = 30 distinct batteries across 9 buildings; leave-fold-out
complements realize 0.48-0.69 (per-fold floors 0.41-0.51) — stable across building groups.
min_rows=40 not 50 because fold 0's complement is 42 rows / 29 events at 0.69.

Analytic result (calibrated arm = per-fold RemainingCalibration applied, floor on top —
algebraically identical to the compensated pre-calibration hook):
**net +126.8 ± 45.2/scenario** (raw arm +196.1 ± 57.1); catches 3.69 → 4.31/scen
(+0.62), missed 5.77 → 5.15, wasted 7.12 → 7.94 (+0.82); 3/48 scenarios worse
(s9 −402, s10 −330, s11 −299 vs s16 +1122, s15 +872). Gain concentrated: opening block
+310/scen, mid +70, closing 0. Flip set: 69 rows, 30 dues (precision 0.435), 16 distinct
due batteries / 5 buildings, scenarios 1-16 only. Insensitive to pricing assumptions
(swap day 7/14/21, op 1.5/4, emergency op 6/30 → +118 to +142).

Fragility (honest): floors x0.75 → nothing fires (floor 0.49 vs p* 0.37-0.44 is a cliff in
the EV proxy; the real planner's pricing is smoother but the warning stands);
min_rows=50 → +38.6; min_events=40 → +12.7; shrink=60 → +46.2. The whole effect is one
bet: **opening-scenario knee cells realize ~0.5, not ~0.2.** On train that is measured;
transfer rides on public/private sharing the chronological scenario structure — the same
bet the shipped RemainingCalibration (x1.71-2.34 at rem>=220) already makes.

## 5. Deeper mining in the pool (margin 0.05-0.15 x beta30>=0.008, rem>=30; n=1604, 190 dues)
AUC (effective, direction noted), pooled / stock regime (rem>=220) / flow regime (<220):
- **knee_worst_14d_drop: 0.68 / 0.71 / 0.57** (more negative → due) — best of the named four
- knee_trend_residual: 0.68 / 0.63 / 0.55 (negative residual → due)
- **slope_30: 0.66 / 0.65 / 0.53** (steeper decline → due)
- knee_slope_vs_history: 0.64 / 0.65 / 0.51; range_90: 0.70 pooled (unrequested, noted)
- v_range_rise: **0.51 — dead**; season/winterness: **0.49 pooled — dead** (anti-winter in stock)
The smoothed-series plunge features DO rank within the beta-elevated pool; the IR channel
finds the population, the plunge features time it. Third axis for the table? Median slope_30
split inside the hot cell: 0.67 vs 0.43 (38 vs 24 events) — the 24-event half fails the
25-event rule, so **not added**. Option for a stricter variant: floor only the
slope_30<=median half at ~0.55 (38 events, survives a x0.75 haircut in the EV proxy).

## 6. Integration (for the integrator — no code change needed in bsai/wiener.py)
`WienerModel.predict_grid` already carries the hook (dwell → **knee_boost** → calibration),
signature `boost.apply(grid, margin, beta30, remaining)`. Wiring is per fold model:

    from bsai.knee import KneeBoostBanded
    boost = KneeBoostBanded.fit(others.margin, others.beta30, others.due, others.remaining)
    boost.compensation = model.calibration   # RemainingCalibration runs after the hook;
    model.knee_boost = boost                 # floor is pre-divided so it lands at the rate

Verified: floored 42d column lands exactly at the empirical rate after calibration,
out-of-domain rows bit-identical, horizon monotonicity preserved. Whole-grid rescale also
lifts the post-window tail, which converts unobserved-EOL mass into observed-tail mass in
the cost tables — the planner will price these swaps cheaper than the proxy did (watch the
wasted count). With an isotonic ReliabilityCalibration in the slot instead, do NOT set
compensation; refit floors against iso-calibrated OOF predictions.

Caveats for the planner test: (1) EV-proxy baseline misses 5.77/scen vs real planner ~3.5-3.9,
so part of the +127 may already be banked — but the missed-dues autopsy profile (margin
0.12, fails 25d later, beta-elevated) is exactly this pool; (2) gain must show in the
opening block without the closing block degrading; (3) dwell adjustment (margin<0.05) is
disjoint from the floor domain (margin>=0.05) — no interaction.

Artifacts: `outputs/knee_floors.json` (flat + banded tables, per fold + production),
`outputs/knee_analytic.json` (all arms, sensitivities, per-scenario), `bsai/knee.py`
(KneeBoost, KneeBoostBanded), `tools/fit_knee.py`, `tools/knee_analytic.py`,
`tools/knee_mine.py`.
