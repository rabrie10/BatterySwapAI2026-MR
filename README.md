---
license: mit
---

# BatterySwapAI 2026

Competition solution for RUL forecasting and cost-aware battery-swap planning.
The submitted entry point is `script.py`.

- **Task 1** is `bsai/`: a multi-horizon first-passage model for the 2.4 V
  crossing.
- **Task 2** is `batteryswap_solution/`: the scheduling and routing optimizer.

They meet at the versioned forecast contract in
`batteryswap_solution/forecast.py`, so the model can be replaced without
touching any scheduling code.

The reasoning behind this design is in
[`docs/PLAN_V6_MAXIMUM.md`](docs/PLAN_V6_MAXIMUM.md); what it actually measures,
including the approaches that failed, is in
[`docs/V6_IMPLEMENTATION.md`](docs/V6_IMPLEMENTATION.md).

## Where it stands

Out-of-fold by building, official `evaluate_plan()` over all 48 train scenarios:

| configuration | mean total cost |
|---|---:|
| all-defer (service nothing) | 3324.7 |
| shipped v3 | 2644.9 |
| **this branch** | **2526.0** |
| perfect knowledge, this planner (scenarios 0-11) | 77.8 |

Runtime is 5.65 s per scenario, projecting to 11.3 minutes for the 96 public and
private scenarios including harness overhead — against a 30-minute limit and the
previous solution's 25.8-27.6 minute projection.

The local mean is a weak predictor of the leaderboard: v3 scored 2644.9 locally
and 4252.3 on public. What that public score actually reflected was
over-servicing in scenarios where the observation window is closing — 41 swaps
per scenario against roughly 9.5 due. That defect is measured and fixed here
(predicted-to-actual due ratio 2.21 -> ~1.0 in the closing scenarios), which is a
difference local validation cannot score.

## What the problem actually is

Lower score is better. Per scenario the evaluator charges work time, travel,
overtime, flat 100-point daily and weekly capacity penalties, 0.5 per day for
swapping early, and 10 per day for swapping late. A battery that reaches EOL
inside the 42-day window and is *not* serviced gets a dedicated emergency visit
after the window, costing between 60 and 480 hours.

On the train split roughly **9.5 batteries per scenario** genuinely reach EOL
inside the window, out of about 420 alive. Servicing nothing at all scores
3324.7. Perfect knowledge with this planner scores 77.8 on scenarios 0-11. The
whole competition is the question of *which* batteries to touch.

## Task 1: `bsai/`

| module | role |
|---|---|
| `smoothing.py` | Incremental, exact reimplementation of `smooth_series`. Pinned to the official function to 1e-12, including the partial-day boundary. |
| `features.py` | 51 causal features from one device's smoothed grid up to one cutoff. |
| `hazard.py` | Multi-horizon classifier: P(EOL recorded within h days), monotone in h. |
| `forecaster.py` | Adapter to the Task 2 forecast contract, including the censoring branch. |
| `validation.py` | Out-of-fold dispatch, so no device is ever scored by a model that saw its building. |
| `runtime.py` | Wall-clock governor for the 30-minute evaluation budget. |

Three design decisions that came from measurement rather than habit:

**The target is a distribution, not a number.** A mean RUL regressor measured on
train has MAE 25 days but is optimistic by +17 days exactly where it matters
(true RUL under 14 days) and pessimistic by 21 days at 70-120 days. That is
regression to the mean, in the direction that makes you late, and late costs
twenty times what early costs.

**Censoring is known, not estimated.** `locations.end_time` is handed to
`plan()`, so for each battery we know the last day an EOL could be recorded, and
the evaluator's substitute EOL for the unrecorded case is exactly
`normalize(end_time + unobserved_eol_days)`. The predicted CDF is capped at its
own value at that horizon. This is what makes the closing scenarios stop
demanding service on their own, with no hand-tuned survivor gate.

**Temperature is a first-order driver.** Within-device, residual voltage tracks
residual temperature at +0.00463 V/degC, positive in 100% of 454 train devices.
The 4.87 degC indoor annual swing is 0.023 V, which near the knee is about two
weeks of remaining life, and EOL incidence is 1.76x higher in Nov-Mar than in
May-Sep. Features include temperature-compensated levels and slopes plus the
expected seasonal temperature change across the planning window.

## Task 2: `batteryswap_solution/`

- `forecast.py`: the versioned Task 1 -> Task 2 contract and its validation.
- `costs.py`: expected early, late, deferred-emergency and evaluator-unobserved
  timing costs, in the unconditional-pmf convention.
- `optimizer.py`: joint CP-SAT service/defer and day assignment under room,
  building, overtime, daily and weekly constraints.
- `routing.py`: exact Held-Karp routes for small days, insertion + 2-opt above.
- `replay.py`: evaluator-exact fast replay, checked against
  `batteryswap_public.evaluate_plan()` before every returned plan.
- `planner.py`: the official `Planner.plan()` adapter, local search, and an
  all-defer safety fallback.

## Reproducing the submission

Both commands are deterministic given the seeds in the source.

```bash
python tools/train_v6.py
```

Writes `models/v6_hazard.joblib` (the shipped artifact, fitted on every
building, carrying isotonic calibrators fitted on out-of-fold predictions only),
`outputs/v6_folds.joblib` (the five fold models, used for validation and not
needed at submission time), and `docs/v6_training_report.json`.

```bash
python tools/validate_v6.py
```

Scores the production planner over all 48 train scenarios using predictions from
models that never saw the device's own building, and prints the anchors so a
result is never read in isolation.

Local submission generation:

```bash
python script.py
```

with `BATTERYSWAP_DATASET_PATH` pointing at the dataset and
`BATTERYSWAP_SPLITS=train`. The official run uses `public,private`.

## Validation protocol

Folds are grouped by **building**, because the public and private splits contain
different buildings and the observed EOL rate per training building spans 0.043
to 0.833. A random split leaks that structure and flatters every number.

The 48 train scenarios are not 48 independent samples: they start a week apart
and each covers six weeks, so adjacent windows overlap by roughly 85% and the
effective sample size is nearer eight. `tools/validate_v6.py --blocks` reports
non-overlapping block means for that reason, and differences under about 100 on
the 48-scenario mean should be treated as noise.

## Tests

```bash
python -m unittest discover -s tests -v
```

Coverage includes exact equality of the incremental smoothing cache against the
official `smooth_series` (single-pass, incremental, and truncated-prefix),
forecast-contract validity on an early and a late scenario, the censoring
branches, the runtime governor, route optimization, plan completeness, and
equality between the fast operational replay and
`batteryswap_public.evaluate_plan()`.

## Environment overrides

- `BATTERYSWAP_MODEL_PATH` -- Task 1 artifact, default `models/v6_hazard.joblib`
- `BATTERYSWAP_PLANNER_PATH` -- load a pickled planner instead
- `BATTERYSWAP_SOLVER_SECONDS`, `BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS`,
  `BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS`, `BATTERYSWAP_ROBUST_SAMPLES`
- `BATTERYSWAP_LATE_RISK_MULTIPLIER`, `BATTERYSWAP_MINIMUM_EXPECTED_IMPROVEMENT`
- `BATTERYSWAP_SOFT_DEADLINE`, `BATTERYSWAP_HARD_DEADLINE` -- governor thresholds

## Third-party components

All runtime dependencies are supplied by the competition image and listed in
`requirements.txt`. The ones this solution actually uses are `numpy`, `pandas`,
`scikit-learn`, `scipy`, `joblib`, `fastparquet`, `ortools` and
`batteryswap_public`, under BSD-3-Clause, BSD-3-Clause, BSD-3-Clause,
BSD-3-Clause, BSD-3-Clause, Apache-2.0, Apache-2.0 and the competition's own
terms respectively. No external datasets and no third-party pretrained model
weights are used. Participant-authored code in this repository is MIT licensed;
see `LICENSE`.
