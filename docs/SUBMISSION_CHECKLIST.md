# Submission Checklist

Status: working checklist, last updated 2026-08-18.

Use this before every official BatterySwapAI submission. Submissions are limited,
so each official attempt should already have passed local checks.

## 1. Code And Artifacts

- `script.py` exists at repository root.
- `script.py` creates `submission.csv`.
- `script.py` uses `BATTERYSWAP_DATASET_PATH` or defaults to `/tmp/data`.
- `script.py` handles `BATTERYSWAP_SPLITS`, defaulting to `public,private`.
- The planner implements the official `Planner.plan()` interface.
- All trained model files, pickles, configs, lookup tables, and other artifacts
  used during submission are committed.
- The submitted code does not require network access during evaluation.
- The submitted code does not require GPU.
- Runtime fits under 30 minutes on CPU.
- Memory fits under 32 GB RAM.
- Only competition-available packages are required.

## 2. Plan Validity

- For each scenario, output columns are exactly `day` and `battery`.
- Every active battery appears exactly once.
- No active battery is missing.
- No battery is duplicated.
- No `day` is before scenario start.
- Deferred batteries are scheduled strictly after the inclusive 42-day horizon.
- Rows are ordered intentionally because same-day row order defines route order.
- Local `batteryswap_public` validation passes.

## 3. Local Evaluation

- Run at least one fast smoke test on train.
- Run exact local evaluation on all train scenarios.
- Compare against baselines:
  - all-defer;
  - simple risk/voltage baseline;
  - oracle train-only benchmark.
- Inspect cost component breakdown:
  - late swap;
  - early swap;
  - travel;
  - room/building change;
  - overtime;
  - daily/weekly limit penalties.
- Confirm the final chosen model beats the current safe baseline by meaningful
  margin across scenario mean and worst cases.

## 4. Docker Test

The official example recommends Docker because it matches the submission
environment.

Build:

```bash
docker build -t batteryswapai-2026-submit .
```

Run:

```bash
docker run --name batteryswapai-submit -v ./dataset:/tmp/data batteryswapai-2026-submit bash -c "/app/env/bin/python3 script.py && /app/env/bin/python3 -m batteryswap_public.metric"
```

Copy `submission.csv` if needed:

```bash
docker cp batteryswapai-submit:/app/submission.csv ./submission.csv
```

If Docker is too slow locally, still run the closest possible local equivalent
before spending an official submission.

## 5. Git Requirements

- All code and artifacts needed for the submitted version are committed.
- The submitted commit SHA is known.
- The commit has been pushed to the Hugging Face model repository.
- Do not rewrite, delete, squash away, or force-push over submitted commits.
- New improvements should be submitted as new commits.
- The repository is in `owner/repository-name` format on Hugging Face.

## 6. Licensing And Prize Eligibility

- Root-level `LICENSE` exists and contains MIT License text.
- Participant-authored code is MIT licensed.
- Third-party dependencies and copied components are documented with source,
  version, and license.
- Dependency licenses are permissive or have written organizer approval.
- No secrets, passwords, access tokens, or private keys are committed.
- If prize-claiming, repository can be made public within 1 hour after the
  official deadline.
- Reproduction instructions are complete enough for organizers to rebuild the
  submission in a clean environment.

## 7. Reproducibility Documentation

Document:

- exact submitted commit SHA;
- data acquisition/loading steps;
- preprocessing steps;
- feature generation;
- training commands;
- model selection method;
- random seeds;
- artifact paths;
- inference/submission command;
- environment assumptions;
- package versions.

## 8. Final Submission Selection

- Maximum 5 submissions per day per competition entry.
- Select up to 3 successfully evaluated submissions as final submissions.
- Failed submissions cannot be selected.
- At least one final submission must be selected before the deadline.
- Keep notes on why each final candidate was selected.

## 9. Do Not Do

- Do not use hidden/private evaluation labels.
- Do not try to reconstruct hidden labels through leaderboard probing.
- Do not hard-code predictions from hidden feedback.
- Do not use unauthorized network calls or persistent background processes.
- Do not bypass submission limits with extra accounts or repositories.
- Do not conceal artifact, model, dependency, or data provenance.

## 10. Fast Pre-Submit Decision

Only submit officially when all of these are true:

- local validation passes;
- local metric is better than current best candidate;
- runtime is comfortably below 30 minutes;
- no missing artifact risk;
- Git commit is pushed;
- this attempt has a clear purpose, such as stronger model, safer risk profile,
  or bug fix.
