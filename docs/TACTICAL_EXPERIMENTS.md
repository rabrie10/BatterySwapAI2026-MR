# Tactical Task 2 Experiments

> Historical experiment protocol. Configuration names and baselines below are
> not the current submission defaults; see `V8_HYBRID_IMPLEMENTATION.md` and
> `script.py` for the active configuration.

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

## 2026-08-19 mixture-cure correction

The next experiments confirmed that reducing the physical floor alone did not
solve the late-scenario error. AFT-only predictions still overserviced devices
that had survived deep into the observation window. The root cause was model
structure: one survival curve was being asked both whether EOL would be
observed and when it would occur.

The v2 model separates those decisions. A building-grouped logistic incidence
model owns `P(EOL by observation end)`; AFT and physical extrapolation only
shape event timing conditional on EOL. Each battery has total training weight
one across synthetic cutoffs. Grouped CV selected `C=1.0`, with weighted Brier
`0.110209`, log loss `0.352895`, mean probability `0.188415`, and event rate
`0.177874` across 82 events and 461 devices.

Physical timing weight `0.25` was selected on representative early scenarios.
A 210-day remaining-observation gate then disables it for survivor-heavy late
scenarios. The threshold was selected on scenarios 14, 16, 18, 20, and 22;
scenarios 24..47 remained locked until final confirmation.

| Metric | Original E0 | Mixture-cure candidate | All defer |
| --- | ---: | ---: | ---: |
| Mean total cost, 48 scenarios | 6023.61 | 2880.87 | 3324.68 |
| P90 total cost | 11095.99 | 4400.34 | - |
| Maximum total cost | 14248.23 | 6261.86 | - |
| Early swap | 4562.57 | 610.78 | - |
| Late swap | 278.12 | 1913.12 | - |
| Operational cost | 1182.91 | 356.97 | - |
| Planned swaps | 71.08 | 10.35 | - |

The candidate reduced mean cost by 52.2% versus E0 and 13.3% versus all defer,
won 41 of 48 paired comparisons against E0, and averaged 17.60 seconds per
scenario. On the locked scenarios 24..47 it scored 2271.08 versus 2309.48 for
all defer. This is the first production checkpoint, not evidence that further
Task 1/Task 2 calibration has saturated.

Reproduce the artifact and full audit with:

```powershell
python -m src.risk.train
python tools/fit_incidence_model.py
python tools/tactical_task2.py `
  --run-name cure-adaptive-full-20260819 `
  --physical-uncertainty-days 1 `
  --physical-risk-weight 0.25 `
  --physical-shape-min-remaining-days 210
```

## 2026-08-19 cutoff-balanced incidence and timing v3

The v2 incidence classifier weighted all landmark rows so each device had total
mass one. That objective is appropriate for lifetime-level model fitting but
biased the repeated planning problem: early-failure batteries had few,
high-weight rows, while late failures were diluted across many low-weight rows.
On grouped OOF predictions, v2 overpredicted observed event counts by 18.56 per
cutoff and had event-count MAE 19.35.

V3 gives every landmark cutoff equal total training mass, matching the
competition's repeated scenario objective. Under the same cutoff-balanced OOF
metric, Brier improved from 0.092589 to 0.090562, log loss from 0.307080 to
0.299888, event-count bias from 18.56 to 3.25, and event-count MAE from 19.35
to 9.17. The grouped OOF event-count correlation is 0.918.

Cutoff balancing exposed a second error: the conditional AFT curve put too
little of the calibrated total event mass inside the 42-day horizon. Physical
timing weights `0.4`, `0.5`, `0.6`, `0.75`, and `0.9` were screened with the
210-day gate fixed. Weight `0.6` won the tuning bracket, then improved a
separate ten-scenario early holdout by 250.47 mean and 457.55 P90 versus v2.

| Metric | Mixture-cure v2 | Cutoff-balanced v3 | All defer |
| --- | ---: | ---: | ---: |
| Mean total cost, 48 scenarios | 2880.87 | 2648.61 | 3324.68 |
| P90 total cost | 4400.34 | 4360.44 | - |
| Maximum total cost | 6261.86 | 5851.71 | - |
| Early swap | 610.78 | 614.60 | - |
| Late swap | 1913.12 | 1700.00 | - |
| Operational cost | 356.97 | 334.00 | - |
| Planned swaps | 10.35 | 10.98 | - |
| Due recall | 0.251 | 0.296 | - |
| Runtime per scenario | 17.60 s | 20.90 s | - |

V3 won 32 paired scenarios, lost 13, and tied 3 versus v2. It reduced mean
cost by another 8.1%, improved the maximum, and remained below the 30-minute
evaluation limit at 17.6 minutes for all 48 scenarios. Against the original E0
pipeline, cumulative mean reduction is 56.0%; against all defer it is 20.3%.

### Submission runtime profile

The full-search v3 profile (`2.0` CP-SAT seconds, 160 local-search evaluations,
70 uncertain-case evaluations) took 1058.1 seconds on 48 train scenarios. That
was too close to the 30-minute limit when conservatively projecting both public
and private splits.

The submission profile uses `1.0 / 80 / 35`. On all 48 scenarios it scored
2644.87 versus 2648.61 for full search: 44 plans tied, 3 improved, and 1
regressed. Mean scenario runtime fell from 20.90 to 16.09 seconds, maximum
runtime fell from 37.19 to 20.39 seconds, and measured wall time was 827.8
seconds. A 96-scenario projection is 25.75 minutes by summed scenario runtime,
or 27.6 minutes by conservatively doubling measured wall time.

Reproduce v3 with the default commands:

```powershell
python -m src.risk.train
python tools/fit_incidence_model.py
python tools/tactical_task2.py `
  --run-name cure-cutoff-balanced-full-20260819 `
  --physical-uncertainty-days 1 `
  --physical-risk-weight 0.6 `
  --physical-shape-min-remaining-days 210
```

## 2026-08-20 unseen-building shift guard v4

The first v3 public submission scored `4252.33`. Its dominant error was
`early_swap=2434.99`, versus `343.23` for the public leader. Battery-change
cost also implied about 41 total actions, compared with roughly 17 actions per
scenario in local v3 evaluation. The local/public mismatch points to aggregate
risk inflation on unseen buildings rather than a routing-only failure.

Several conservative policies were tested and rejected as release defaults:

- Always applying the temporal event-rate cap was neutral on the tuning set
  but worsened a locked even-scenario sample by `225.58` mean cost because it
  removed real failures.
- A 3% hard planned-swap cap worsened tuning mean by `307.93`; a 4% cap
  worsened it by `106.46`. Both traded too much late cost for early savings.
- A direct 42-day evaluator-outcome classifier cut OOF event-count MAE from
  `9.17` to `4.04`, but its standalone recall was too low. A 25% blend improved
  planned precision from `0.380` to `0.404` on tuning while increasing mean
  cost by `84.68`, then lost `134.78` on the locked `24..47` period. Its
  release weight is therefore zero.

The v4 release uses a one-sided shift detector around the smoothed train
event-rate prior. It preserves every per-battery rank and leaves a scenario
unchanged while predicted 42-day event mass is below `1.9x` its calendar-time
prior. The largest ratio across all 48 train scenarios was `1.8721`. When the
ratio exceeds `1.9`, one shared logit intercept is shifted until total event
mass matches the prior. This targets the public failure mode without changing
in-distribution behavior.

Release evidence:

- 35 unit/integration tests pass.
- Eight release-smoke scenarios, including the highest train mass ratios,
  exactly reproduce v3 plans and every cost component (`max_abs_delta=0`).
- In a stress case with 41 expected events among 460 batteries, the shift
  guard reduces event mass to `11.64` (`71.6%`) while preserving ranking.
- No hard planned-swap cap is active, and the experimental direct-head weight
  is `0.0`.

This is deliberately a public-domain-shift intervention. Train scenarios
cannot prove the expected public reduction because none cross the activation
boundary. The next official submission is the required confirmation test;
the key acceptance signal is a large fall in `early_swap` and operational
cost without a late-swap increase that erases the gain.

Rebuild the release artifact with:

```powershell
python tools/fit_incidence_model.py `
  --horizon-rate-cap-multiplier 1.0 `
  --horizon-rate-activation-ratio 1.9 `
  --horizon-rate-smoothing-window 9 `
  --direct-horizon-weight 0
```
