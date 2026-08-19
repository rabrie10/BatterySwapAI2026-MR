---
license: mit
---

# BatterySwapAI 2026

Competition solution for RUL forecasting and cost-aware battery-swap planning.
The submitted entry point is `script.py`; the Task 2 implementation lives in
`batteryswap_solution/`.

## Task 2 architecture

- `forecast.py`: versioned Task 1 -> Task 2 probability contract, validation,
  and a submission-safe voltage-trend fallback.
- `costs.py`: evaluator-aligned expected early, late, deferred emergency, and
  evaluator-unobserved timing costs.
- `optimizer.py`: joint CP-SAT service/defer and day assignment with room,
  building, overtime, daily, and weekly constraints.
- `routing.py`: exact Held-Karp building routes for small days and deterministic
  insertion + 2-opt for larger routes.
- `replay.py`: fast operational replay that is checked against the official
  evaluator before every returned plan.
- `planner.py`: official `Planner.plan()` adapter, robust emergency sampling,
  building/day LNS, compound threshold repair, and all-defer safety fallback.

## Task 1 integration

Task 1 should serialize an object implementing `RiskForecaster.predict()` to:

```text
models/risk_forecaster.pkl
```

The exact v1 schema is defined in `batteryswap_solution/forecast.py` and
`docs/SOLUTION_DESIGN_SPEC.md`. If the artifact is absent or fails validation,
the planner uses `VoltageTrendForecaster`; that fallback is for operational
safety and parallel development, not the intended final leaderboard model.

The production artifact is trained in two explicit stages. Stage 1 fits the
censoring-aware AFT timing model; stage 2 attaches the grouped-CV incidence
model and writes the final mixture-cure artifact in place:

```powershell
python -m src.risk.train
python tools/fit_incidence_model.py
```

The stage-2 defaults reproduce the validated configuration: one-day physical
timing uncertainty, `0.25` physical timing weight, and a 210-day survivor gate.
Training diagnostics are written to `docs/incidence_training_report.json`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover forecast-contract validation, asymmetric timing cost, route
optimization, plan completeness, emergency queue behavior, and equality
between the fast operational replay and `batteryswap_public.evaluate_plan()`.

## Train benchmark

Oracle mode isolates the attainable Task 2 ceiling using train labels only:

```powershell
python tools/benchmark_task2.py --mode oracle --limit 12
```

The oracle forecaster is defined only in the benchmark tool and is never loaded
by `script.py`. Fallback mode exercises the complete no-artifact submission path:

```powershell
python tools/benchmark_task2.py --mode fallback --limit 3
```

## Local submission generation

```powershell
$env:BATTERYSWAP_DATASET_PATH = (Resolve-Path dataset).Path
$env:BATTERYSWAP_SPLITS = "train"
python script.py
```

The official run uses `public,private`. Optional environment controls are:

- `BATTERYSWAP_FORECASTER_PATH`
- `BATTERYSWAP_PLANNER_PATH`
- `BATTERYSWAP_LATE_RISK_MULTIPLIER`
- `BATTERYSWAP_MINIMUM_EXPECTED_IMPROVEMENT`
- `BATTERYSWAP_SOLVER_SECONDS`
- `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS`
- `BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS`
- `BATTERYSWAP_ROBUST_SAMPLES`

Keep the default runtime profile until a full public/private-equivalent local
run confirms that a larger search budget remains comfortably below 30 minutes.
