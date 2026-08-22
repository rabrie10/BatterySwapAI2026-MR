# Pi-feature hybrid (P3-1 stage 2): filter state as GBDT features — gate report

_Model engineer, 2026-08-22. Code: `tools/twophase_pihybrid.py` (+ `PiHybridModel`
appended to `bsai/twophase.py`; no shared module was modified). Artifacts:
`outputs/twophase_pi.parquet` (212,977 per-(device,day) rows: pi_posterior,
pi_drift = pi*mu2 + (1-pi)*trailing_drift, OOF fold id),
`outputs/pihybrid_gates.json`, `outputs/twophase_pihybrid_model.joblib`,
`outputs/pihybrid_validation.json` (+ `_dm125.json` attribution control)._

## VERDICT: NO-GO as commissioned (dm1.6: 2037.9 vs bar 1860) — but the
## one-knob budget-anchored control scored **1861.45** (bar tie, -63 vs band):
## conditional GO candidate pending a replicate

As commissioned (the operating flags, due-multiplier 1.6), gate (d) FAILED at
2037.9 against the 1924-1972 band. The decomposition attributes the whole
failure to the budget mechanism, not the ordering — the exact inverse of the
audit's sum-p-collapse trap: honest calibration inflated expected_due and made
the due-budget inert. Re-anchoring that single knob to the incumbent's
effective budget (dm 1.25) with the SAME model produced the best planner mean
this repo has recorded (1861.45; late/early/misses/recall/precision all better
than the incumbent simultaneously). Details and caveats below.

## Build

Exact train_wiener recipe (stride 4, max_iter 250, censored increment targets,
5-fold GroupKFold by building; increment rows/targets extracted through
`build_increment_targets` itself via an index-carrying frame, so base and +pi
variants share byte-identical rows, folds and targets). Parity proof: the base
control reproduces the recorded pipeline exactly — 88,013 cutoffs, 550,560
windows, PR-AUC 0.4706, predicted/actual 0.845; the stride-8 transfer bank
reproduces the harness's 275,951 windows. Pi features are OOF at every stage
(a device's pi comes from the fold filter that excluded its building; hard
folds re-fit the EM per fold). Deployment inside `PiHybridModel.predict_grid`:
pi columns appended before dispatch to the fold GBDT, so `feature_row`,
`bsai/features.py` and the forecaster stay untouched. Volatility scale 1.2
(same as v8); RemainingCalibration per fold OOF.

## Gate table

| gate | criterion | result | verdict |
|---|---|---|---|
| (a) stride OOF PR-AUC | vs like-for-like 0.4706 | **+pi 0.4530 vs base 0.4706** (-0.018; AUC 0.9834 vs 0.9836) | FAIL (this clause) |
| (a) frame-level AP | vs cens-cal 0.308 | **0.4779 cal / 0.4899 raw** (+55%) | pass (this clause) |
| (b) mid-block top-12 | >= 0.27 | **0.339** (incumbent 0.214, +58%) | **PASS** |
| (b) open-block top-12 | >= 0.55 | **0.615** (incumbent 0.589) | **PASS** |
| (c) hard-holdout mean AP | >= 0.428 | **0.547 cal / 0.572 raw** — beats cens on ALL 5 folds: large5 0.527/0.497, small10 0.568/0.603, mosteol5 0.482/0.288, hirate6 0.603/0.339, betashift5 0.555/0.413 | **PASS** |
| (c) sum-p inflation | <= 1.15 | hard mean 0.874 (worst fold 1.21); 5-fold OOF 1.039 | **PASS** |
| (d) planner, operating config | <= 1860 | **2037.86** (reference same-flags incumbent 1924.5; band 1924-1972; reroll noise +-50-60) | **FAIL** |
| (d') attribution control, dm1.25 | <= 1860 | **1861.45** (bar tie; -63 vs band best; late 980 / early 623 / misses 3.73 / recall 0.606) | tie, single reroll |

Sum-p / budget: blocks cal 12.88 / 9.15 / 7.46 vs incumbent 10.33 / 9.27 / 8.62
(realized 13.25 / 8.56 / 6.56 — mine is the honest level, ratios 0.97/1.07/1.14);
mean budget 13.94 vs 14.31, 14/48 scenarios smaller.

## Why (d) failed while (a-c) flew: the budget rewards miscalibration

Decomposition vs the incumbent's same-flags run (per scenario): late +80.6,
early +25.1, weekly +10.4, ops ~+2 — with recall UP (0.570 vs 0.564), misses
DOWN (4.06 vs 4.12). The damage is all in the closing third (6-block means:
-10 / -53 / +17 / **+210 / +214 / +302**). Mechanism: the incumbent
under-predicts the opening block (ratio 0.78) and its expected_due of 10.28
lets the due-budget ceil(1.6*sum_p+1) bind below 15 in the s_26-s_39 region
(11-14 slots — part of the measured operating point). My calibrated levels are
honest (expected_due 13.12, opening ratio 0.97), so ceil(1.6*13.12+1) >= 15
everywhere: the due-budget went inert, served rose 14.19 -> 14.90, and the
extra marginal swaps landed exactly where the audit prices them worst. The
incumbent's miscalibration was functioning as a hidden volume damper. This is
the mirror image of the audit's "sum-p collapse shrinks the budget" trap: a
BETTER-calibrated sum-p inflates it.

Attribution control (one knob): same model, --due-multiplier 1.25 (=1.6 x
10.28/13.12, restoring the incumbent's effective budget profile under my
honest levels — `outputs/pihybrid_validation_dm125.json`):

| run | mean | late | early | served | missed | recall | precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent lm1.8 (same flags, dm1.6) | 1924.5 | 1034.4 | 650.7 | 14.19 | 4.12 | 0.564 | 0.376 |
| hybrid dm1.6 (as commissioned) | 2037.9 | 1115.0 | 675.8 | 14.90 | 4.06 | 0.570 | 0.362 |
| **hybrid dm1.25 (budget re-anchored)** | **1861.45** | **980.4** | **622.9** | 14.65 | **3.73** | **0.606** | **0.391** |

The budget knob alone moves the hybrid -176. At the incumbent-equivalent
budget the ordering prize arrives on every axis at once: late -54, early -28,
misses -0.39/scen, recall +4.2 pts, precision up — more catches with fewer
wasted swaps, the definition of the ORDERING pool (ops+capacity gives back
~+19, weekly-heavy). Six-block means 1749/2168/2595/1692/1917/1047 vs
1548/2186/3024/1880/2073/836: the mid blocks (the documented wall) carry
-429/-187/-156 while blocks 1 and 6 give back +201/+210 (single-reroll
numbers; block 6's closing scenarios have tiny realized counts). 1861.45 vs
the 1860 bar is a statistical tie (reroll noise +-50-60) and sits 63 below
the band's best member (1924.5).

Caveats before anyone ships this: (i) one reroll — repo protocol demands a
replicate and the paired harness; (ii) dm1.25 was chosen AFTER observing the
dm1.6 failure (one degree of adaptivity — derived from expected_due ratios,
not tuned on score, but still post-hoc; the honest deployment re-derives the
multiplier per split as 1.6 x incumbent-E[due]/hybrid-E[due], or drives the
budget from the incumbent sum-p per the audit guard); (iii) the opening-block
+201 needs one look at its swap audit before submission talk.

## What survives regardless (for the next wave)

1. The pi features are the largest transfer-stable ranking gain recorded on
   this repo's own harnesses: hard-holdout mean 0.547 vs cens 0.428 vs v7
   0.391; mid-block top-12 0.214 -> 0.339 (the documented wall); frame AP
   0.308 -> 0.478. `outputs/twophase_pi.parquet` is reusable as-is.
2. The stride-pooled AP dip (0.4706 -> 0.4530) with simultaneous scenario-frame
   gains says the pi features re-allocate capacity from device-day-weighted
   discrimination to fleet-at-cutoff discrimination — the population the
   planner actually sees.
3. Mechanism lesson for the economist/planner roles: the due-budget
   (1.6*sum_p+1, cap 15) is only calibration-honest for models that
   under-predict like the incumbent. Any better-calibrated model needs the
   budget re-anchored (multiplier ~1.25 at honest levels, or rank-quota /
   absolute), or it silently converts calibration honesty into volume.
4. With the budget re-anchored there is no residual ordering deficit: the
   hybrid beats the incumbent on late, early, misses, recall and precision
   simultaneously. The open items are the replicate (reroll noise), the
   pre-registered budget rule, and the deployment plumbing (incremental
   filter cache + 'pi' variant registration) for a real submission.

## Deviations / notes

- Gate (a) was coded as "+pi must beat base on stride AND frame > 0.308"; the
  stride clause failed, (b)+(c) passed, and (d) was run as commissioned (it
  was the decisive arbiter either way). Ladder-literal reading would have
  stopped at (a); the coordinator's (d) commission and the transfer gate's
  record-margin pass justified the run — and (d)'s answer is what settles it.
- The variant-registry integration ('pi' in bsai/features.py FEATURE_VARIANTS)
  was deliberately NOT wired: the pi lookup lives in PiHybridModel, so no
  shared file changed. Cost: `train_wiener.py --feature-variant pi` does not
  exist yet; the registry route needs a ShapeCache-like incremental filter
  cache plus a feature_row pi-callable for a real submission (the forward
  filter is already incremental; the work is plumbing, est. ~150 lines in
  features/forecaster following the raw-daily pattern).
- Validation used `validate_v6.py --production --model` with the
  self-dispatching OOF artifact (OofHazardModel does not forward device ids);
  leave-building-out honesty is inside the artifact, verified by construction.
- Hard-fold fidelity mirrors the harness (stride 8 / iter 150 / per-fold
  calibration on training-building rows); pi filters were re-fit per hard fold.
