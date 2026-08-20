# Submission Decision Record — 2026-08-20

Branch: `combined`. Shipped artifact: `models/risk_forecaster.pkl`
(`task1-mixture-cure-shrunk/v4`).

This records what is being submitted, why, and what the evidence does and does
not support. It is deliberately conservative about claims because the only
out-of-sample datapoint we have is a single leaderboard result.

## What is being submitted

The partner's mixture-cure v3 model with **one field changed**:
`incidence_scale = 0.7`. Nothing else differs — same AFT, same incidence
classifier, same physical timing weights, same Task 2 planner and runtime
budget.

## Why

The 2026-08-19 submission exposed a train-to-public generalization gap:

| | local (in-sample) | public leaderboard |
| --- | ---: | ---: |
| `total_cost` | 2648.61 | **4252.33** |
| batteries serviced / scenario | 16.9 | **41.1** |
| `early_swap` | 615 | **2435** |
| `late_swap` | 1700 | **535** |

`early_swap` exploding while `late_swap` collapses is the signature of
**over-swapping**: too many batteries predicted to fail, most serviced
needlessly (early cost), leaving almost nothing to miss (late cost falls).

Public and private contain entirely different buildings, so the cause is
incidence over-prediction on unfamiliar devices. `docs/OOF_GENERALIZATION.md`
documents the harness built to measure this locally; it confirms the mechanism
— planned swaps rise monotonically with unfamiliarity, **10.98 in-sample →
15.06 out-of-fold → ~41 on public**.

`incidence_scale=0.7` was selected on that harness (2793.93 vs 2880.46 at
1.0, with 0.5 clearly worse at 3151.97, bracketing a real optimum).

## Evidence quality — read before trusting the change

- The OOF harness reproduced only **~14%** of the observed public gap. It is a
  directionally valid signal, not a calibrated predictor. Expect improvement
  over 4252.33, not a jump to the leaders' ~1500.
- The change costs **3.3% in-sample** (2648.61 → 2735.73) and gains **3.0%
  out-of-fold** (2880.46 → 2793.93). Whether that trade pays depends entirely
  on public being more like the OOF condition than the in-sample one — which
  the leaderboard result strongly suggests, but does not prove.
- `incidence_scale` was tuned **only** against the OOF harness. It was not
  fitted to leaderboard feedback. Using the published component breakdown to
  diagnose the failure mode is legitimate; fitting a parameter to a hidden
  split would be leaderboard probing and is prohibited.
- One leaderboard datapoint cannot distinguish "incidence over-predicts on new
  buildings" from other split differences (device age mix, event density). The
  mechanism is consistent with all evidence but is not proven.

## The defensible alternative

Submitting the **unmodified v3** is a reasonable choice. It is the in-sample
optimum, it is the partner's validated and documented work, and the hedge
rests on a proxy that understates the very effect it is correcting. Switch
with no code change:

```powershell
$env:BATTERYSWAP_FORECASTER_PATH = "models/risk_forecaster_v3_baseline.pkl"
```

If two submissions are available, submitting both and comparing is strictly
more informative than choosing between them on this evidence — and it would
give the first real measurement of how OOF gains translate to public.

## Pre-submission checklist

| Item | Status |
| --- | --- |
| `script.py` at root, produces `submission.csv` | yes |
| Honours `BATTERYSWAP_DATASET_PATH` / `BATTERYSWAP_SPLITS` | yes |
| Loadable artifact committed at `models/risk_forecaster.pkl` | yes |
| No network access at inference | yes |
| Only competition-available packages | yes (pandas, numpy, sklearn, lifelines, scipy, ortools) |
| Root `LICENSE` (MIT) | yes |
| `Dockerfile` + `.dockerignore` | yes |
| Unit tests | 32 passing |
| No secrets committed | verified by scan |
| Full train run produces valid `submission.csv` | yes — 19,890 rows, 48 scenarios, 381–458 batteries each, 0 duplicates, 0 nulls |

## Runtime — measured, and no longer the binding risk

Task 2 submission profile is `1.0` CP-SAT seconds / `80` local-search / `35`
uncertain-case evaluations (`script.py` defaults).

A precisely timed full run of `script.py` over all 48 train scenarios with the
shipped artifact took **6.6 minutes** (`Measure-Command`, 2026-08-20),
projecting to roughly **13 minutes for a 96-scenario** public+private run
against the 30-minute limit.

That is a large improvement on the partner's 827.8 s / 25.75–27.6 min
projection for the same planner settings, and it is a **side effect of the
conservatism change**: `incidence_scale=0.7` services ~6.2 batteries per
scenario instead of ~11, so there is materially less routing, local search and
replay work per scenario. Runtime was previously the tightest constraint on
this submission; it now has roughly 2x headroom.

If a future change reverses this, the levers remain
`BATTERYSWAP_SOLVER_SECONDS` and `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS`, both
environment-overridable. Note that the rollback artifact
(`risk_forecaster_v3_baseline.pkl`) swaps more and will therefore run closer to
the partner's original ~26-minute projection.

## Provenance

- Mixture-cure model, incidence classifier, cutoff-balanced weighting,
  tactical experiment protocol, runtime budget: partner
  (`docs/TACTICAL_EXPERIMENTS.md`).
- Calibration defect investigation on the shared v1/v2 baseline:
  `docs/TASK1_MODEL_INVESTIGATION.md` (branch `ronsdag`).
- OOF harness, `incidence_scale`, this record:
  `docs/OOF_GENERALIZATION.md`.
