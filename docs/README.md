# Documentation guide

## Start here

| Document | What you will find |
|---|---|
| [Technical case study](CASE_STUDY.md) | Problem, decisions, quantified outcomes, team contributions, and lessons |
| [Results and evidence](RESULTS.md) | Rank 5, score progression, full leaderboard breakdown, and validation context |
| [Architecture](ARCHITECTURE.md) | Forecasting, expected costs, scheduling, routing, and evaluation boundaries |
| [Reproducibility](REPRODUCIBILITY.md) | Setup, model download, data layout, training, validation, and tests |
| [Synthetic demo](DEMO.md) | Run a dataset-free planning walkthrough and inspect its outputs |
| [Model card](../models/README.md) | Shipped artifact, training scope, limitations, and historical weights |
| [Tool index](../tools/README.md) | Supported commands versus exploratory utilities |

## Experiment history and competition references

The documents below preserve the development process. Their scores, defaults, model choices, and status statements describe the revision when they were written. **Current setup and shipped settings are documented above.** A historical claim that a model is “current” or “best” is not a statement about the present submission.

| Area | Historical documents |
|---|---|
| Challenge and evaluator | [Official challenge reference](OFFICIAL_CHALLENGE_REFERENCE.md), [planner reference](TASK2_PLANNER_REFERENCE.md), [submission checklist](SUBMISSION_CHECKLIST.md) |
| Original system design | [Solution design](SOLUTION_DESIGN_SPEC.md), [Task 1 implementation](TASK1_IMPLEMENTATION.md), [Task 2 implementation](TASK2_IMPLEMENTATION.md) |
| V5–V6 | [V5 plan](PLAN_V5_TOP_SCORE.md), [V6 plan](PLAN_V6_MAXIMUM.md), [V6 implementation](V6_IMPLEMENTATION.md) |
| Wiener / V7 | [V7 plan](PLAN_V7_MARGIN.md), [V7 implementation](V7_IMPLEMENTATION.md), [saved training report](v7_training_report.json) |
| Later experiments | [V8 plan](PLAN_V8_PRECISION.md), [V8 hypothesis tests](V8_HYPOTHESIS_TESTS.md), [V9 blend](V9_BLEND.md), [tactical experiments](TACTICAL_EXPERIMENTS.md) |
| Development handover | [Handover and traps](HANDOVER.md) |
| Prior landing page | [Competition README at 328a3ac](archive/COMPETITION_README.md) |

Historical training reports remain committed as evidence. New runs should write separate reports instead of replacing those used in the results narrative. Dataset access, evaluator behavior, and competition terms are governed by the organizers; the local reference notes do not override them.
