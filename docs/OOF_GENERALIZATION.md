# Out-of-Fold Generalization Harness

Status: new 2026-08-19. Addresses the gap flagged in
`docs/TASK1_IMPLEMENTATION.md` Sec 6 and required by
`docs/SOLUTION_DESIGN_SPEC.md` Sec 8.1.

## Why this exists

Every local `total_cost` number produced before this harness — in
`docs/TACTICAL_EXPERIMENTS.md`, `docs/local_benchmark_log.csv`, and
`tools/benchmark_task2.py --mode real` — is **in-sample**. Task 1 is fit on
all 461 train devices and then scored on train scenarios, so every battery
being forecast was in the training set. Model *selection* inside those fits is
building-grouped and honest; the end-to-end cost is not.

The 2026-08-19 submission made the cost of that gap concrete:

| | local (in-sample) | public leaderboard |
| --- | ---: | ---: |
| `total_cost` | 2648.61 | **4252.33** |
| batteries serviced / scenario | 16.9 | **41.1** |
| `early_swap` | 615 | **2435** |
| `late_swap` | 1700 | **535** |

Public and private contain entirely different buildings. The inversion —
`early_swap` exploding while `late_swap` collapses — is the signature of
**over-swapping**: the model predicts too many batteries will fail, the planner
services them, most never needed it (early cost), and almost nothing is left
to miss (late cost falls).

## What the harness does

`tools/fit_oof_forecasters.py` fits one model per building fold, each trained
with ~6 of the 24 buildings held out, and wraps them in
`src.risk.oof.OutOfFoldForecaster`. That wrapper implements the ordinary
`RiskForecaster.predict()` contract but routes **each battery to the fold
model that never saw that battery's building**. Running the normal benchmark
against it therefore measures unseen-building behaviour — the same condition
the public split presents — while still using train labels for scoring.

```powershell
python tools/fit_oof_forecasters.py --dataset-path data/raw/train --template models/risk_forecaster.pkl
python tools/benchmark_task2.py --dataset-path data/raw/train --mode real --forecaster-path models/risk_forecaster_oof.pkl --limit 0
```

Fitting takes ~11 minutes (4 folds x AFT + incidence); the benchmark ~15.

## Result: the harness reproduces the direction, not the magnitude

Mixture-cure v3, 48 train scenarios:

| Metric | in-sample | **out-of-fold** | public (actual) |
| --- | ---: | ---: | ---: |
| `total_cost` | 2648.61 | **2880.46** | 4252.33 |
| planned swaps | 10.98 | **15.06** | ~41 |
| `early_swap` | 614.60 | **901.55** | 2434.99 |
| `late_swap` | 1700.00 | **1597.71** | 535.00 |
| all-defer reference | 3324.68 | 3324.68 | unknown |

Swap count rises monotonically with unfamiliarity: **10.98 → 15.06 → ~41**.
That is the mechanism, confirmed. But the harness captures only ~14% of the
observed cost gap (+8.8% vs +61%), so train's own held-out buildings are
*less* out-of-distribution than the public split's buildings.

**Use it as a directional selection signal, never as a calibrated predictor of
leaderboard score.** When choosing a parameter on this harness, prefer the
conservative side of the OOF optimum, because the harness systematically
understates the shift.

## The conservatism knob and its selected value

`Task1Forecaster.incidence_scale` (default `1.0`, a no-op) uniformly shrinks
predicted incidence at inference. It is the mildest intervention that reduces
swap count: the ranking of batteries by risk and the timing shape are both
untouched, only the total predicted event mass moves.

Swept on the OOF harness, 48 train scenarios:

| `incidence_scale` | OOF `total_cost` | planned swaps | `early_swap` | `late_swap` |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 (unmodified v3) | 2880.46 | 15.06 | 901.55 | 1597.71 |
| **0.7 — selected** | **2793.93** | 7.98 | 435.39 | 2048.13 |
| 0.5 | 3151.97 | 3.50 | 192.18 | 2682.29 |

0.5 being clearly worse is the important half of this result: it brackets a
real optimum rather than showing a monotone "more conservative is better"
trend. Below ~0.7 the missed in-horizon failures outweigh the early-swap
saving.

### In-sample cost of the hedge

| Model | in-sample | OOF |
| --- | ---: | ---: |
| mixture-cure v3 (`scale=1.0`) | **2648.61** | 2880.46 |
| shrunk v4 (`scale=0.7`) | 2735.73 | **2793.93** |

An almost exactly symmetric trade: 3.3% worse in-sample, 3.0% better
out-of-fold. `scale=0.7` is shipped because the competition scores unseen
buildings, and because the observed public failure mode was over-swapping
(~41 batteries serviced per scenario against ~11 locally), which this
directly reduces. Anyone preferring the in-sample optimum can switch back in
one environment variable — see "Rolling back" below.

Expectation management: the OOF harness reproduced only ~14% of the real
public gap, so this should be read as a hedge with a validated *direction*,
not a tuned constant. It is expected to improve on 4252.33 without
approaching the leaders' ~1500.

The asymmetry that justifies shrinking rather than inflating: servicing a
battery that never reaches EOL costs up to ~182 (early penalty across the days
to its proxy EOL, `location.end_time + 30`), while deferring that same battery
costs **nothing**. Over-prediction is therefore far more expensive than
under-prediction, and the public result shows the model errs toward
over-prediction on unfamiliar devices.

Select it on the OOF harness. Do **not** tune it against leaderboard scores —
using the displayed component breakdown to *diagnose* is legitimate, but
fitting a parameter to a hidden split is leaderboard probing, produces a value
overfit to one observation, and is prohibited by the competition rules.

## Shipped artifacts and rolling back

| Path | Model | Purpose |
| --- | --- | --- |
| `models/risk_forecaster.pkl` | `task1-mixture-cure-shrunk/v4` | **What `script.py` loads.** v3 with `incidence_scale=0.7`. |
| `models/risk_forecaster_v3_baseline.pkl` | `task1-mixture-cure-cutoff-balanced/v3` | Unmodified baseline, kept for one-step rollback. |

The two artifacts differ in exactly one field. To submit the unmodified
baseline instead, no code change is needed:

```powershell
$env:BATTERYSWAP_FORECASTER_PATH = "models/risk_forecaster_v3_baseline.pkl"
```

The OOF artifacts used for the sweep are deliberately **not** committed — they
are ~2.4 MB each and fully regenerable from
`tools/fit_oof_forecasters.py` in ~11 minutes. Only what `script.py` needs at
submission time is committed.

## Honest limitations

- 24 buildings / 4 folds means each fold model trains on only 18 buildings, so
  fold models are weaker than the production model. Some of the OOF penalty is
  reduced training data rather than pure distribution shift; the harness is a
  slightly pessimistic estimate of a full-data model on familiar buildings and
  a clearly optimistic one for public.
- Scoring still uses train labels and train scenario structure. Only the
  *forecasting* side is held out.
- The harness inherits whatever the template artifact fixes (family,
  penalizer, physical weights). It re-fits the AFT and incidence per fold but
  does not re-select hyperparameters, so hyperparameter overfitting to train
  is not captured.
