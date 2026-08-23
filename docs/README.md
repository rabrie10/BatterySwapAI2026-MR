# BatterySwapAI 2026 documentation

This index separates the code that currently ships from historical designs and
experiments. Statements in a historical document describe that version only;
they are not descriptions of the current submission.

## Current implementation

- [V8 boosted-hazard/Wiener implementation](V8_HYBRID_IMPLEMENTATION.md): the
  Task 1 model loaded by `script.py`, its active probability path, calibration,
  service budget, validation, and reproduction commands.
- [Task 2 implementation](TASK2_IMPLEMENTATION.md): the production
  `batteryswap_solution/` planner and forecast contract.
- [Task 2 planner reference](TASK2_PLANNER_REFERENCE.md): evaluator semantics,
  cost model, and planner requirements.
- [Submission checklist](SUBMISSION_CHECKLIST.md): packaging, LFS, Docker,
  validity, and reproducibility checks.
- [Official challenge reference](OFFICIAL_CHALLENGE_REFERENCE.md): a dated local
  summary of organizer material and evaluator inspection.

The submission entry point is `script.py`. By default it loads
`models/v8_ensemble.joblib`, a gradient-boosted near-term hazard/ranking model
blended with a Wiener first-passage model, and passes its forecast to
`batteryswap_solution.planner.CompetitionPlanner`.

## Historical material

The following files are retained as experiment records and rationale, not as
current implementation documentation:

- `TASK1_IMPLEMENTATION.md` and `task1_training_report.json`: the legacy
  `src/risk` parametric-AFT/physical-prior forecaster.
- `V6_IMPLEMENTATION.md`, `v6_training_report.json`,
  `PLAN_V6_MAXIMUM.md`: the retired V6 hazard classifier.
- `V7_IMPLEMENTATION.md`, `v7_training_report.json`, `PLAN_V7_MARGIN.md`: the
  standalone V7 Wiener candidate.
- `PLAN_V5_TOP_SCORE.md`, `SOLUTION_DESIGN_SPEC.md`, and
  `TACTICAL_EXPERIMENTS.md`: proposals and experiment protocols; some ideas
  were rejected or superseded.

The canonical organizer rules take precedence over all local notes.
