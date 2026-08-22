# V12 building-invariant feature generation: measured to a NO-GO

Written 2026-08-22. Model engineer's full record of the V12 feature-variant
generation: design, two measured iterations (the second including the
coordinator's knee re-weighting and any-temperature raw channel), judged on
the transfer-stress harness (hard building holdouts + 24-fold LOO), and the
honest verdict. Measurement basis: docs/TRANSFER_STRESS.md,
docs/V11_TRANSFER_FINDINGS.md, outputs/roadblock_report.md.

## Verdict: NO-GO

No V12 arm beats the cens incumbent's mean hard-holdout PR-AUC of 0.428 (best
arm v12b_k1: **0.4195**, and it loses to cens on 4/5 hard folds), no arm beats
cens's pooled-LOO PR-AUC of 0.3155 (best 0.2975), and every arm inflates the
raw out-of-building level MORE than cens (LOO raw x1.42-1.48 vs x1.297).
Calibrated inflation is fine everywhere (<= x1.075 vs gate 1.10) -- the level
problem is already absorbed by the RemainingCalibration, not by feature
choices. Per the mission gate, the planner validation was NOT run and no
production artifact was shipped to models/.

**The central finding inverts the spec's premise: the building-bound absolute
features (within-day scales beta_30/v_std_30/etc. and the temperature levels)
are fragile by distribution shift yet load-bearing for out-of-building
RANKING. Removing or replacing any subset of them costs 0.01-0.03 hard-holdout
PR-AUC, and nothing added back (rise ratios, per-device z, recovery residuals,
raw channels, knee re-weighting) recovers the loss.**

## What was built (all additive, base path proven byte-identical)

Feature variants in `bsai/features.py` (registry `set_feature_variant` /
env `BSAI_FEATURE_VARIANT`; `feature_row(..., variant=, raw=, raw_any=)`):

| variant | n | definition |
|---|---:|---|
| base | 64 | unchanged, pickle-compatible with every shipped model |
| extended | 84 | base + 11 new + 7 raw-daily + 2 any-temp raw |
| invariant (round 1, frozen) | 63 | drops observations, 4 temp levels, 6 absolute shape scales, t_range_30; no raw |
| invariant2 (iteration) | 79 | drops only observations + 4 temp levels; keeps scales, adds everything |

New features: `temp_now_delta`/`temp_recent_delta` (within-device temperature
contrasts), `beta/v_std/v_range_rise_7` (7d vs 180d), `_rise_mid` (30d vs
90d), `beta_z_30` (causal prefix median/IQR z), `recovery_residual_30/60`
(voltage move minus the device's own causal early-life dV/dT x temperature
move; per-device beta from a V~1+T+day regression over the first 180
pair-valid days, frozen, fleet fallback 0.00463, clipped to [0, 0.02]),
`raw_last/min3/mean3/min7/slope7/days7_below_2.42/2.45` (integrator's
bsai/rawdaily.py, 10-30 degC), `raw_any_last/min3` (bsai/v12_rawany.py, no
temperature filter -- roadblock vii: 85% of dark dues are raw-fresh there).

Knee re-weighting (roadblock ii-iv): `tools/v12_fit.py` reproduces
bsai/wiener.py's exact fit recipe with `sample_weight = 1 + w_knee` on
increment windows whose end lies within 21 d of the device's crossing
(measured share 0.90%); swept w_knee in {1, 3}.

Files: `bsai/features.py`, `bsai/shape.py` (prefix_median_iqr), `bsai/v12_rawany.py`
(new), `bsai/forecaster.py` (raw caches, gated on variant; base path inert),
`tools/train_wiener.py` (`--feature-variant`, `--knee-weight`),
`tools/v12_frame.py`, `tools/v12_fit.py`, `tools/v12_transfer.py`,
`tools/v12_loo_rate.py`, `tools/v12_rate_at_rank.py`, `tools/v12_check.py`
(35 checks). All 66 unit tests pass; the harness reconstruction check
reproduces the shipped sums exactly (v7 8.717 / cens 12.37 raw per scenario)
proving the base columns are untouched fleet-wide.

## Harness results (stride 8 / max_iter 150, cens targets, identical to the incumbent measurements)

Means over the 5 hard grouped holdouts and pooled 24-fold LOO
(outputs/v12_transfer.md has every per-fold row):

| metric | v12b | v12b_k1 | v12b_k3 | v12b_noany | v12b_noraw | v12 | v7 | cens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard PR-AUC lvl mean (gate > 0.428) | 0.4058 | **0.4195** | 0.4161 | 0.4001 | 0.4070 | 0.3975 | 0.3914 | **0.4278** |
| hard PR-AUC rank mean | 0.3573 | 0.3709 | 0.3576 | 0.3558 | 0.3628 | 0.3530 | 0.3700 | **0.3748** |
| LOO PR-AUC lvl | 0.2898 | 0.2722 | 0.2975 | 0.2743 | 0.2802 | 0.2671 | 0.2796 | **0.3155** |
| LOO inflation raw (gate <= 1.10) | 1.417 | 1.482 | 1.471 | 1.422 | 1.427 | 1.447 | **1.052** | 1.297 |
| LOO inflation calibrated | 1.067 | 1.054 | 1.028 | 1.069 | 1.075 | 1.072 | 1.041 | **1.007** |
| hard inflation cal worst fold | 1.199 | 1.194 | 1.136 | 1.265 | 1.152 | 1.096 | 1.023 | 1.056 |

Best arm (v12b_k1) vs cens per hard fold, PR-AUC lvl: large5 0.472 vs 0.497,
small10 0.608 vs 0.603, mosteol5 0.283 vs 0.288, hirate6 0.326 vs 0.339,
betashift5 0.409 vs 0.413 -- one coin-flip win, four losses.

## Rate-at-rank, pooled LOO (out-of-building; the 0.214 reference was 5-fold OOF)

Top-12 realized rate per scenario third (outputs/v12_loo_rate_at_rank.json):

| variant | early | mid | late | pooled |
|---|---:|---:|---:|---:|
| v12b | 0.547 | 0.203 | 0.229 | 0.326 |
| v12b_k1 | 0.562 | 0.214 | 0.214 | 0.330 |
| v12b_k3 | 0.573 | 0.188 | 0.224 | 0.328 |
| v12 | 0.552 | 0.198 | 0.219 | 0.323 |
| v7 | 0.573 | **0.245** | 0.234 | 0.351 |
| cens | **0.594** | 0.219 | **0.260** | **0.358** |

The mid-block ordering failure is untouched: no arm beats even the incumbents
under LOO (v7 0.245 / cens 0.219), let alone the ~0.4 the leaderboard gap
implies. The knee re-weighting moved hard-fold PR-AUC +0.010..+0.014 but
mid-block rate not at all (k3 is worst there at 0.188).

## Component attribution (ablations, LOO PR-AUC / hard PR-AUC)

* Temperature levels -> deltas (cens vs v12b_noraw, no other change):
  **-0.035 LOO / -0.021 hard.** The single most costly substitution. The temp
  levels' KS shift on hard folds is real, but so is their signal.
* Absolute shape scales dropped (v12 vs v12b_noraw): -0.013 LOO / -0.010 hard.
  Confirms round 1: fragile AND load-bearing.
* Filtered raw channel (v12b_noany vs v12b_noraw): -0.007 hard / -0.006 LOO --
  nothing, despite ranking #5-8 in drift importance. Any-temp raw (v12b vs
  v12b_noany): +0.006 hard / +0.016 LOO -- the only consistently positive
  increment of anything added this generation, consistent with roadblock vii,
  but an order of magnitude too small to close the gap.
* Knee weighting: +0.014/-0.018 (k1), +0.010/+0.008 (k3) hard/LOO -- inside
  noise, and it worsens raw inflation (1.48/1.47 vs 1.42).
* Drift permutation importance (v12b_k1, in-sample): temp_now_delta #3,
  raw_slope7 #5, raw_last/raw_min7 #7-8 -- the new channels are USED by the
  model; they just do not add out-of-building ordering beyond what the
  absolute features already carry.

## Why the level gate reads differently than the spec assumed

Raw sum-p inflation is target-bound, not feature-bound: censor-aware targets
(the production choice, kept per spec) put even the shipped cens at LOO raw
x1.297; v7's x1.052 comes from its uncensored target, not its features. Under
the production procedure (RemainingCalibration fitted on training buildings),
every arm and both incumbents sit at x1.007-1.075 -- inside the x1.10 gate.
The V11 ship decision (cens ranking behind BATTERYSWAP_MAX_PLANNED=15 +
calibration) already contains the level; V12's features were only worth
shipping if they beat cens's RANKING out-of-building, and they do not.

## What was deliberately not run

* Planner validation (`tools/validate_v6.py`) -- gated on the harness, which
  failed. Per docs/V11_TRANSFER_FINDINGS.md the planner responds to order and
  volume; the harness shows V12 orders worse, so a planner run could only
  produce a noise-band number that invites a wrong KEEP.
* Production artifacts `models/v12_invariant.joblib` / `outputs/v12_folds.joblib`
  -- not written, to keep a refuted variant out of the shippable set. The full
  path is proven end-to-end by a smoke run (stride 64, 2 folds, scratch
  output; model_version `bsai-wiener/v1+invariant2+k1`, 79 stored
  feature_names, predicts). If anyone wants the artifact:

      OMP_NUM_THREADS=3 ./.venv/Scripts/python.exe tools/train_wiener.py \
        --feature-variant invariant2 --knee-weight 1 --stride 4 --max-iter 250 \
        --out models/v12_invariant.joblib --folds-out outputs/v12_folds.joblib \
        --report outputs/v12_training_report.json
      # then fit_calibration.py / validate_v6.py with BSAI_FEATURE_VARIANT=invariant2

## What survives this generation

1. **Keep shipping cens-behind-cap.** Nothing here beats it across buildings.
2. The variant machinery (registry, extended rows, raw adapters, weighted
   fits, the harness with arms/ablations) is tested and reusable at ~25 min
   per full 6-arm judgment; the next feature idea costs one arm, not a rebuild.
3. The any-temp raw channel is the only additive with a consistent positive
   sign (+0.016 LOO PR-AUC). Its natural next use is NOT as a tree feature but
   in the gate/forced-include layer the roadblock report ranks #3 (~95 net
   pts/scen), where its 85%-of-dark-dues coverage acts directly on selection.
4. The knee problem (drift 0.036 V vs required 0.12-0.20 V) is real but
   sample-weighting does not fix it out-of-building; the shrinkage is
   information-limited (105 crossing windows at h<=14 fleet-wide), not
   loss-limited. Candidate next steps: pool crossing evidence across horizons
   (per-device time-to-cross target) or a dedicated short-horizon head, judged
   on this same harness.
5. For the integrator: `bsai/forecaster.py` now carries both raw caches, gated
   on `feature_lib.variant_needs_raw(active_feature_variant())` -- inert under
   "base". `bsai/rawdaily.py` was not modified; the any-temp twin lives in
   `bsai/v12_rawany.py`. Wiring pattern for any future variant model:
   `raw=partial(raw_cache.features_at, device_id)`, same for `raw_any`.

## Artifacts

* outputs/v12_transfer.json / outputs/v12_transfer.md -- full harness tables,
  gate ladder, permutation importance (raw-channel + all-new-feature slices).
* outputs/v12_loo_rate_at_rank.json -- rate-at-rank per block, all arms.
* Work dir (scratchpad/v12_transfer): v12_prep.joblib (84-col frames, window
  bank + knee sidecar), fits/ (174 fold fits across 6 arms).
* Reproduction: `python tools/v12_check.py` (35 checks);
  `python tools/v12_transfer.py --phase all --arms v12b v12b_k1 v12b_k3`;
  `python tools/v12_loo_rate.py`.
