# Ensemble study: OOF matrix, combination league, winner, GO/NO-GO

_Ensemble engineer, 2026-08-22. Code: `tools/ensemble_matrix.py` (builds
`outputs/ensemble_matrix.parquet` + `outputs/ensemble_matrix_meta.json`),
`tools/ensemble_eval.py` (all numbers -> `outputs/ensemble_results.json`).
Matrix build 32 s, full league 9 s; no planner runs spent._

## The matrix (19,890 rows x 20 cols, keyed scenario/battery)

All components verified against their recorded anchors before use:

| component | source | pooled AP | mid top-12 | hard mean |
|---|---|---:|---:|---:|
| p_cens (cal, 5-fold OOF) | frame_oof_cal / research_rowfeat p_cal | 0.3083 | 0.213 | 0.4278 (refit protocol = the bar) / 0.4298 prod-OOF |
| p_tp (two-phase p42) | twophase_model_oof.joblib predict_rows | 0.3843 | 0.292 | **0.4513** |
| p_qh (quantile head) | qhead_folds.joblib by-building dispatch on transfer prep | 0.2830 | 0.213 | 0.4201 |
| raw composite (min3+slope7 ranks) | research_rowfeat | 0.1106 | 0.167 | 0.1399 |
| p_censhard_* (5 cols) | transfer-stress hard refits (stride8/it150) | — | — | reproduces 0.4278 exactly |

Seq head: not shipped (findings file still TBD) — excluded. Spearman decorrelation:
cens~tp 0.92, **cens~qh 0.25, tp~qh 0.25**, qh is the orthogonal voice; raw ~0.63/0.66.

## Combination x gate league (fuller table in ensemble_results.json)

Gate (a): pooled AP vs best single 0.3843; mid-block top-12 vs 0.292 (frontier).
Gate (b): mean PR-AUC over the 5 transfer-stress hard groups (score built with the
hard-refit cens component, whole-fleet ranks, AP on held rows) vs bar 0.428.

| combination | AP | top12 open/mid/late | hard mean | verdict |
|---|---:|---|---:|---|
| rank-avg ct (=Borda) | 0.3105 | .536/.250/.255 | 0.4070 | rank flattening kills level info; all rank avgs 0.24-0.31 AP |
| rank-mix c.25/t.75 (best of 0.25 sweep) | 0.3130 | .557/**.297**/.260 | 0.3907 | only mid-12 win anywhere (+1 catch); fails (a) pooled + (b) |
| min-rank ct / ctq | 0.280 | .568/.260/.260 | 0.372 | union inflates weak-component FPs; dead |
| noisy-OR ct / ctq | 0.349/0.354 | .552-.562/.260-.276 | 0.460/0.463 | passes (b), fails (a) |
| logit-mean family (17 weightings c/t/q) | 0.350-0.386 | mid .240-.286 | **0.468-0.482 every member** | (b) passes family-wide, selection-free; 5/5 folds for c.125/t.75/q.125 |
| LOBO logistic stack ct/ctq/(+remaining) | 0.318-0.354 | mid .234-.255 | 0.429-0.452 | **stacking trap confirmed again**: never beats both inputs nor the rank average |
| **WINNER: remkeyed_puremid** — logit mix keyed on remaining (open>225d: .25c+.5t+.25q; mid 115-225d: pure tp; late<=115d: .125c+.75t+.125q) | **0.3873** | .573/**.292**/.255 | **0.4742** | passes ladder: beats every single |

## Winner vs gate ladder (remkeyed_puremid)

- (a) pooled AP **0.3873 > 0.3843** (tp, best single); LOO-by-scenario delta positive
  in **46/48** leave-outs (range −0.002..+0.009). Per block AP .558/.225/.246
  (mid AP dips at the remaining/block seam rows; mid top-12 is exactly tp's).
  Mid-block top-12 **ties** the best single at 0.292 by construction (mid regime IS tp
  — every mixing weight tried there loses catches; cens+qh are blind to knee-entries).
  Open-block ordering improves (top-12 .573 vs tp .562; open AP .555 vs .531).
- (b) hard mean **0.4742 >= 0.428 bar**, beats tp 0.4513 (4/5 folds), beats cens
  0.4278 (worst fold 0.4147 vs cens's 0.2879 mosteol5 collapse — the transfer-fragility
  that produced the +179 public surprise is exactly what the mix repairs).
- Honesty: the pooled-AP margin (+0.003) is small; the robust, family-wide gain is
  TRANSFER (+0.02-0.03 hard over best single, +0.05 over the shipped cens). Mid-block
  learnability is NOT improved — that wall stands for this generation.

## The budget trap (checked, it fires)

Deployed as a p-level, the mix shrinks the due-budget min(15,ceil(1.6Σp+1)) in 13/48
scenarios (mean 13.52 vs cens 14.31) — the measured-dead volume cut. **Deployment must
be Σp-preserving: keep cens's per-scenario p multiset, reassign p to batteries in
ensemble-rank order** (RankCalibration-style within-scenario remap). Measured as
`perm_censLevels_remkeyedPuremidOrder`: identical per-scenario order/top-12 as the
winner, identical Σp/budget/slot economics to ship.

## Deployable scorer spec (integration = model swap, zero forecaster edits)

`HazardForecaster.predict` hands the WHOLE scenario's rows to `model.predict_grid`
(bsai/forecaster.py:211; the rank_calibration hook at :216 is the precedent that
within-scenario reordering lives at this level). Ship a `RankMixModel` in bsai/ that
wraps the three models and presents the WienerModel interface:

    p_c grid = cens.predict_grid(features, remaining, devices)   # OOF or prod artifact
    z = w(remaining) . [logit p_c42, logit tp.predict_rows(margin, remaining, devices)[:,0],
                        logit qh.probabilities(features, min(42, remaining))]
    out = p_c grid with column 11 (and order) remapped: sort p_c42 desc, assign by z-order

3-line integration in tools/validate_v6.py / script.py: load the three artifacts,
`model = RankMixModel(cens, tp, qh)`, pass to HazardForecaster (calibration already
inside cens). Weights: open(>225d rem) .25/.5/.25, mid(115-225) 0/1/0, late(<=115)
.125/.75/.125 on logits.

Fold artifacts for local OOF validation: outputs/v8_folds_cens.joblib (via
OofHazardModel), outputs/twophase_model_oof.joblib (fold-dispatching, calibration
inside), outputs/qhead_folds.joblib (by_building). Production: models/v8_cens.joblib,
same twophase artifact (production_params fallback), outputs/qhead_model.joblib.

## GO/NO-GO: **GO — as an order-only, budget-safe reorder, validated on the paired harness**

Hand to the integrator: replay the reorder as a selection arm on
`outputs/paired_incumbents.joblib` (tools/paired_selection.py --reuse, ~2 min, exact
deltas, no reroll noise) before any full validate_v6 run. Expected realized value is
bounded: mid-block order is unchanged vs tp, so the local paired delta should come
from open/late ordering and the hard-transfer insurance is the real prize (public
Task-1 direction). If the paired arm reads >= 0 locally, the transfer argument alone
still justifies A/B'ing cens -> RankMix on the next submission slot; that call is the
integrator's. NO-GO for: any direct p-level deployment of the mix (budget shrink),
rank-average/min-rank/noisy-OR/stack variants (fail gate a), and any further weight
tuning (grid is at selection-noise scale).
