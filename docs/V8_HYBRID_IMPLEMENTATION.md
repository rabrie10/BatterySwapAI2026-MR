# V8 hybrid failure model

V8 combines three complementary survival signals:

- a histogram gradient-boosted discrete-hazard model trained at official
  scenario landmarks;
- a direct 42-day gradient-boosted ranking head;
- a censored Weibull AFT tail.

The hybrid is blended in log-odds space with the V7 Wiener first-passage model.
The Wiener component supplies the conditional tail beyond day 42.  A smooth
scenario-phase correction changes only total probability mass, preserving the
battery ranking and conditional curve shape.

Training calibration is grouped by building.  The full 48-scenario train
backtest uses out-of-fold building models and therefore never scores a building
with a model trained on that building.  The selected ensemble uses 25% hybrid,
75% Wiener, and Wiener volatility scale 1.4.

## Results

| Validation | Mean total cost |
|---|---:|
| Shipped V7 local baseline | 2,293 |
| V8 ensemble, before phase correction | 2,240 |
| V8 ensemble, selected phase correction | 1,999 |
| **V8 + expected-incidence service budget** | **1,962** |

This is a 12.8% reduction against the shipped local V7 baseline.  It is not a
claim of beating the 1,160.67 public leader: only a public submission can
establish that, and train-to-public shift may be material.

### Early-swap precision repair

The final planner limits proactive service to
`ceil(2 * expected_due + 5)` batteries per scenario.  Unlike the old
percentage cap, this is computed from the calibrated full-fleet 42-day event
mass and is therefore stable when candidate filtering changes size.  The
headroom protects dense scenarios; the constraint activates mainly in quiet
closing scenarios where the uncapped planner serviced 29--43 batteries for
only 2--6 observed failures.

On all 48 building-grouped OOF scenarios this changed mean total cost
1999.33 -> **1962.22**, early cost 778.85 -> **749.03**, and planned service
17.81 -> **16.15** batteries per scenario.  Recall moved only 0.634 -> 0.626.
Heavier hazard blends from 30% through 50% were rejected: 50% reduced early
cost to 709.10 but increased late cost to 1010.62 and total cost to 2043.95.

## Reproduce

```powershell
python tools/train_hybrid.py
python tools/train_wiener.py
python tools/build_v8_ensemble.py
python tools/validate_v6.py --dataset data/raw/train `
  --folds outputs/v8_hybrid_folds.joblib `
  --blend-folds outputs/v7_wiener_folds.joblib `
  --blend-weight .25 --blend-volatility-scale 1.4 `
  --report outputs/v8_ensemble_phase_full.json --audit
```

The submission entry point now defaults to `models/v8_ensemble.joblib`.
