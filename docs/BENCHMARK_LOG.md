# Local Benchmark Log

Status: working log, last updated 2026-08-19.

The official leaderboard allows only 5 submissions/day. `docs/local_benchmark_log.csv`
is a local substitute: it runs the real evaluator (`batteryswap_public.evaluate`)
against **train** scenarios through the same `CompetitionPlanner`, and records
one row per run with the exact same cost-component columns the leaderboard
shows (`battery_swap`, `building_change`, `room_change`, `travel`, `overtime`,
`daily_limit`, `weekly_limit`, `late_swap`, `early_swap`, `total_cost`). This
lets us tell whether a change helped *before* spending a submission on it.

This is a directional signal, not a leaderboard prediction: train, public, and
private are different scenario sets, so absolute numbers will differ from what
the leaderboard shows. What's comparable across runs of this log is the
*relative* change — did `total_cost` go down, did `late_swap`/`early_swap`
shrink, etc. — run over run, on the same fixed scenario set (train, all 48
scenarios by default).

## How to record a new entry

After making a change (retraining, a code fix, a config tweak):

```powershell
python tools/benchmark_task2.py --dataset-path data/raw/train --mode real --limit 0 --record docs/local_benchmark_log.csv --label "short description of what changed"
```

- `--limit 0` runs all 48 train scenarios (takes ~15-20 minutes; use a smaller
  `--limit` for a quick sanity check, but only compare `--limit 0` runs to
  each other — different scenario counts aren't apples-to-apples).
- `--label` is a free-text note stored with the row (e.g. `"baseline"`,
  `"after physical-prior blend fix"`) — make it specific, it's the only thing
  that tells you what a row actually represents later.
- `--mode fallback` or `--mode oracle` record comparison baselines (the
  deterministic fallback, or the train-label oracle ceiling) using the same
  mechanism — useful context rows, not competitors to the real model.

The log is append-only CSV (`docs/local_benchmark_log.csv`) — never overwritten,
so history is preserved. Open it in any spreadsheet tool, or diff two rows
directly, to see what a change actually did.

## Reading a row

| Column | Meaning |
|---|---|
| `commit` | short git SHA the run was made at |
| `mode` | `real` (the actual Task 1 artifact), `fallback` (`VoltageTrendForecaster`), or `oracle` (train-label ceiling, dev-only) |
| `model_version` | `Task1Forecaster.model_version` for `real` runs |
| `n_scenarios` | how many scenarios the mean is over — only compare rows with matching counts |
| `total_cost` … `early_swap` | mean of the official evaluator's cost components, same names as the leaderboard |
| `all_defer` | the trivial "defer everything" baseline's cost on the *same* scenarios, for context |
| `label` | your free-text note — always fill this in |

## Recorded history (all 48 train scenarios)

| Row label | `total_cost` | swaps | Note |
|---|---:|---:|---|
| `current-pushed-model (blend+speed fix)` | 6023.61 | 71.1 | v1/v2 — the model behind the 7389.39 leaderboard entry |
| `v3: isotonic cal + blend-before-cal + long-horizon cal` | 4005.33 | 43.1 | |
| `v4: horizon-conditional isotonic calibration` | **3128.88** | 28.0 | **shipped** — first to beat all-defer |
| `v5: tie-break + negative crossing-days` | 3168.27 | 28.3 | rejected, 1.3% worse |

Reference points on the same 48 scenarios: **all-defer = 3324.68**, and the
train-label **oracle ceiling ≈ 78**. Those bracket the achievable range — a
model above 3324.68 is worse than doing nothing, which is exactly how the
v1/v2 problem was caught.

Compare any new row against **3128.88** first, then against all-defer.
`docs/TASK1_MODEL_INVESTIGATION.md` explains what changed between each row.

## Caveat: these numbers are in-sample

The artifact is trained on all 461 train devices and benchmarked on train
scenarios, so these figures are optimistic. Model/calibrator selection is
honest (building-grouped OOF), but the end-to-end costs are not. Treat the
log as a *relative* instrument for comparing changes, not as a leaderboard
prediction. See `docs/TASK1_IMPLEMENTATION.md` Sec 6.
