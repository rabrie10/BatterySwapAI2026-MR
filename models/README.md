# Shipped model card

| Field | Description |
|---|---|
| Artifact | `v7_wiener.joblib` (Git LFS) |
| Class | `bsai.wiener.WienerModel` |
| Model version | `bsai-wiener/v1` |
| Task | Multi-horizon probability of crossing the battery failure threshold |
| Inputs | 64 causal features from voltage, temperature, seasonality, and within-day statistics |
| Architecture | Gradient-boosted drift/volatility regressors with a Wiener first-passage distribution |
| Training | Official competition training buildings; 546,042 increment windows in the saved report |
| Evaluation | Building-grouped out-of-fold predictions; 88,013 cutoff examples, ROC-AUC 0.9823 at scale 1.0 |
| Runtime wrapper | `bsai.forecaster.HazardForecaster` |
| Shipped postprocessing | Volatility 1.0, probability scale 1.5, observation-horizon handling, transmission veto |

The training metrics describe out-of-fold model predictions before the latest submission postprocessing, not the in-fold production artifact or the synthetic demo. See [results and evidence](../docs/RESULTS.md).

## Download and inspect

```bash
git lfs install
git lfs pull --include="models/v7_wiener.joblib"
python tools/demo.py --model
```

The model contains participant-trained weights, not a third-party pretrained model. It is intended for the competition sensor population and evaluator. New device types, chemistry, seasons, and observation policies may change both calibration and ranking. Recorded failure is not equivalent to an independently observed physical failure under communication loss.

The latest rank-5 result supplied by the team is not yet tied to an exact artifact/configuration. It should not be treated as a verified score for every setting in the current code.

## Historical artifacts

- `cox_baseline.pkl`: earlier Cox proportional hazards baseline.
- `risk_forecaster.pkl`: earlier risk forecaster.
- `v9_blend.joblib`: research blend, retained for comparison; not the default in `script.py`.

Historical artifacts may import classes in `src/` or other `bsai/` modules. Those packages are retained for compatibility. Joblib/pickle can execute code when loaded; use the trusted repository artifact, not arbitrary downloaded files.
