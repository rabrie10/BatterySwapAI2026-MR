# Tactical Task 2 Experiments

This protocol keeps leaderboard work attributable: one behavioral variable is
changed per experiment, every candidate is compared with the same baseline,
and public submissions are reserved for candidates that improve locked local
validation.

## Error buckets

`tools/tactical_task2.py` records both official cost components and the action
classification behind them:

- `planned_due`: a battery due inside the horizon that the planner selected;
- `planned_not_due`: a selected battery without an observed in-horizon EOL;
- `emergency`: an in-horizon EOL missed by the plan and replaced by the
  evaluator after the horizon;
- `due_recall`: planned due batteries divided by all in-horizon failures;
- `planned_precision`: planned due batteries divided by planned swaps.

The battery report also contains forecast CDF values by horizon, planned timing
cost, location, EOL state, and the forecast tail probabilities.

## Locked protocol

1. Freeze the current model and planner settings as `E0`.
2. Run `E0` on all 48 scenarios for the permanent baseline report.
3. Screen one-variable changes on even scenarios from the first 24 weeks:
   `0,2,4,...,22`.
4. Choose one candidate using paired scenario deltas, not aggregate score alone.
5. Confirm that candidate once on the locked later period `24..47`.
6. Reject candidates that improve mean while materially worsening p90,
   worst-case cost, capacity penalties, or runtime.
7. Only then run the candidate on all 48 scenarios and consider an official
   submission.

The first controlled variable is `physical_uncertainty_days`. It changes the
sharpness of the physical crossing-time forecast without retraining Task 1 or
changing Task 2.

## Commands

Full E0 baseline:

```powershell
python tools/tactical_task2.py --run-name e0-full --physical-uncertainty-days 20
```

Initial screen on the tuning subset:

```powershell
python tools/tactical_task2.py `
  --run-name uncertainty-screen `
  --scenario-indices 0,2,4,6,8,10,12,14,16,18,20,22 `
  --physical-uncertainty-days 12 16 20 24 28
```

Outputs are written under `outputs/tactical/<experiment>/`:

- `summary.json`: aggregate mean, median, p90, maximum, and exact config;
- `scenarios.csv`: paired scenario-level costs and error counts;
- `batteries.csv`: forecast and decision details for every active battery.

Generated reports are intentionally ignored by Git. The tool and this protocol
are versioned; experiment outputs remain local evidence.

## 2026-08-19 E0 and uncertainty result

The complete E0 run reproduced the leaderboard-style failure locally. The
planner selected many batteries with no observed in-horizon EOL, while still
missing a smaller set of real failures:

| Metric | E0 (`u=20`) | Candidate (`u=1`) |
| --- | ---: | ---: |
| Mean total cost, 48 scenarios | 6023.61 | 5024.83 |
| All-defer mean | 3324.68 | 3324.68 |
| P90 total cost | 11095.99 | 9418.98 |
| Maximum total cost | 14248.23 | 13604.33 |
| Early swap | 4562.57 | 3562.65 |
| Late swap | 278.13 | 429.38 |
| Operational cost | 1182.91 | 1032.81 |
| Planned swaps | 71.08 | 61.73 |
| Emergency swaps | 1.21 | 1.42 |
| Due recall | 0.865 | 0.847 |
| Planned precision | 0.148 | 0.173 |

`u=1` was selected on the predefined 12-scenario tuning set. On the untouched
later 24-scenario period it improved mean cost from 4990.04 to 4412.11, won 21
of 24 paired comparisons, reduced P90 from 8370.37 to 7263.65, and limited the
worst regression to 169.42. Across all 48 scenarios it won 43 comparisons and
improved mean by 998.78 (16.6%).

This is a validated component improvement, not a release candidate. It remains
worse than all-defer because the physical forecast floor is still overconfident:
among 2577 planned batteries without an observed in-horizon EOL, the median
predicted day-42 failure CDF is 0.841. The next one-variable experiment should
therefore reduce or gate the physical term in `max(AFT CDF, physical CDF)` while
holding `physical_uncertainty_days=1` and every planner setting fixed.
