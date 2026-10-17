# Synthetic planning demo

Run from the repository root after installing `requirements.portfolio.txt`:

```bash
python tools/demo.py
```

Open `outputs/demo/index.html`. The self-contained report works offline and shows six synthetic sensors across three buildings, daily failure curves over 14 days, and service/defer decisions from the real planner. Linked CSV files expose the exact forecasts and schedule; a JSON summary records the forecast source and operational cost.

## Two modes

| Command | Forecast source | Requirements |
|---|---|---|
| `python tools/demo.py` | Explicit illustrative failure distributions | Python dependencies only |
| `python tools/demo.py --model` | Shipped Wiener model on synthetic hourly voltage/temperature | Dependencies and `models/v7_wiener.joblib` downloaded through Git LFS |

The default separates the forecasting/planning interface cleanly: high-risk and low-risk examples make service-versus-deferral behavior easy to inspect. It uses an independent fixture, not imports from test code. The optional model mode exercises the actual artifact and inference adapter; its forecasts should not be interpreted as realistic field estimates because the telemetry is synthetic.

## What is checked

Both modes run `CompetitionPlanner`, generate daily assignments and route order, and call the official `check_plan_valid` and `evaluate_plan`. Errors that trigger the planner's fallback fail the demo rather than reporting a seemingly successful result.

No true end-of-life labels are supplied. The displayed score therefore covers operational costs only; it does not measure failure prediction, total competition performance, or savings. Demo search budgets are smaller than submission budgets to keep the example quick.

## Outputs

- `index.html`: forecast curves, decision table, and explanation.
- `forecast.csv`: per-battery, per-day cumulative failure probabilities.
- `schedule.csv`: the actual planner output; rows also encode visit order.
- `summary.json`: synthetic-data label, forecast mode, decision counts, and operational score.

Use `--output path/to/folder` to choose another destination. Generated files are ignored by Git; GitHub Actions uploads its demo output as a run artifact.
