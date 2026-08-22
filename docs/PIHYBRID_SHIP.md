# Pi-hybrid: conditional-GO completion (replicates, budget rule, deployment)

_Model engineer, 2026-08-22. Completes the three ship conditions on the
pi-feature hybrid (see `outputs/pihybrid_findings.md` for the gate record:
frame AP 0.478 vs 0.308, mid-block top-12 0.339 vs 0.214, hard-holdout mean
0.547 vs cens 0.428 with all five folds won)._

## 1. Replicates at the re-anchored budget (dm 1.25, operating flags otherwise)

| run | mean | late | early | served | missed | recall | precision | report |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| replicate 1 | 1861.45 | 980.4 | 622.9 | 14.65 | 3.73 | 0.606 | 0.391 | `outputs/pihybrid_validation_dm125.json` |
| replicate 2 | **1839.02** | 952.1 | 630.1 | 14.65 | 3.67 | 0.612 | 0.395 | `outputs/pihybrid_validation_dm125_r2.json` |
| **pair mean** | **1850.2** | 966 | 627 | | 3.70 | 0.609 | | |
| incumbent operating band | 1924.5-1972.5 | 1034-1078 | 651-668 | 14.2 | 4.1-4.25 | 0.55-0.56 | 0.376 | audit + val_lm18* |

Pair mean 1850.2 clears the 1860 win bar; both replicates sit 63-134 below the
band (reroll noise +-50-60; replicate spread 22). Decision metrics improve on
every axis simultaneously in both replicates — the ordering prize, not a
volume trade. Six-block means move together (r1/r2 mid blocks -429/-497,
-187/-202, -156/-180 vs the band's audit replicate; blocks 1 and 6 give back
+160-210 each — opening-block swap audit is the one open diagnostic).
(As-commissioned dm1.6 run, for the record: 2037.9 — the budget went inert at
honest levels; see findings.)

## 2. Pre-registered budget rule (deployable formula)

    planned-swap budget = min(BATTERYSWAP_MAX_PLANNED,
                              ceil( m_pi * sum_p_cal + 1 ))
    m_pi = 1.6 * E_incumbent / E_hybrid = 1.6 * 10.28 / 13.12 = 1.2536 -> 1.25

where sum_p_cal is the hybrid's RemainingCalibration-corrected expected-due
mass over the alive fleet, computed inside plan() from the split's own data
(the same quantity the incumbent budget consumes), and E_* are the two models'
train-time 5-fold-OOF expected-due levels — model constants frozen at training
time exactly like the 1.6 itself, the volatility scale and the climatology. No
train-realized counts enter at plan time. Rationale: the 1.6 multiplier was
tuned against the incumbent's expected-due level (10.28/scen OOF); the hybrid
is better calibrated (13.12 vs realized 13.25 in the opening block), so the
SAME effective budget requires rescaling by the level ratio, else calibration
honesty silently converts into +0.7 swaps/scenario of marginal volume
(measured: 2037.9 vs 1850.2). Deployment: `BATTERYSWAP_DUE_MULTIPLIER=1.25`
(env already wired in `script.py::build_planner_config`); cap stays
`BATTERYSWAP_MAX_PLANNED=15`.

## 3. Deployment plumbing

* **Incremental filter**: `bsai.twophase.PiFilterCache` — rides the
  forecaster's own `SmoothingCache` (no re-smoothing), advances each device's
  changepoint filter per `predict()` call: hold-merge on exact repeats, causal
  expanding-MAD observation scale (refresh 30/+28, fleet constant 1.160
  mV/sqrt-day frozen from train before 30 increments), plateau jump mixture /
  pure-core plunge, transition-after-emission. The provisional smoothing
  boundary day (mid-day scenario cuts) is deferred until finalized, so the
  filter never consumes a value that could be revised (pi lags <= 1
  observation day).
* **Attachment**: `ProductionPiHybrid` wrapper carries the production GBDT +
  the cache; `HazardForecaster.predict` feeds any model-attached `pi_cache`
  (5-line additive hook in `bsai/forecaster.py`, same pattern as the
  resurrection gate; the variant registry was deliberately left untouched —
  the model computes its own features, so `script.py`'s load path needs no
  variant selection).
* **Equivalence proven** (`tools/twophase_ship.py`): t1 single-pass
  incremental vs the batch pipeline (make_tracks/causal_scales/forward_pi):
  461/461 devices, day sequences identical, max |pi diff| 0.0 -> PASS.
  t2 cumulative-chunked (three growing slices, the make_submissions feeding
  pattern): 461/461, max |pi diff| 0.0 -> PASS. (A first t2 draft fed
  disjoint slices and failed — no deployment path feeds disjoint data.)
* **Tests**: full suite 66/66 OK (102 s) with the forecaster hook in place.
* **Smokes**: (i) `validate_v6.py --production --model models/pihybrid.joblib
  --limit 4` + operating flags: runs the live filter per scenario, 0 cold
  starts, 8.7 s/scenario mean (13.3 max), projected 16.2 min for 96 scenarios
  — inside the 30-minute budget (`outputs/pihybrid_smoke.json`; in-fold, so
  its score is diagnostic only). (ii) `script.py` loader path with
  `BATTERYSWAP_MODEL_PATH=models/pihybrid.joblib`,
  `BATTERYSWAP_DUE_MULTIPLIER=1.25`: BudgetedPlanner loads the artifact and
  plans s_0/s_1 in 11.8 s / 7.4 s (cache warm on the second call).

## Artifacts

| path | content |
|---|---|
| `models/pihybrid.joblib` | production: all-buildings 66-feature Wiener GBDT (stride 4, iter 250, volatility 1.2), production RemainingCalibration (factors 0.443/0.735/0.990/1.104/1.197/1.265), live `PiFilterCache` (production filter params, fleet scale 1.160 mV) |
| `outputs/twophase_pihybrid_model.joblib` | fold artifact: leave-building-out dispatching hybrid (per-fold GBDTs + OOF filters + per-fold calibration) — the object behind every validation number |
| `outputs/twophase_pi.parquet` | 212,977 per-(device,day) OOF pi features |
| `outputs/pihybrid_gates.json`, `outputs/pihybrid_findings.md` | gate record |

## Residual risks before a real submission

1. The dm 1.25 constant was derived after observing the dm 1.6 failure (from
   expected-due ratios, not score-tuned) — pre-registered NOW for any future
   split, but it has been validated on train only.
2. Public transfer of the LEVEL still rides RemainingCalibration; the
   hard-holdout inflation was 0.87 mean (worst fold 1.21) — better than the
   incumbent's x1.30 record, and the cap 15 still backstops it.
3. Opening/closing-block give-back (+160-210/block, single-reroll scale)
   unexplained; one `swap_audit` pass on blocks 1 and 6 recommended.
4. Emergency-replaced devices keep filter state through the replacement jump
   (self-correcting via recovery evidence; not equivalence-tested post-EOL).
