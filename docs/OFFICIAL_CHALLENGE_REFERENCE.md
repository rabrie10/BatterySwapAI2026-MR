# BatterySwapAI 2026 Official Challenge Reference

Status: working reference, last updated 2026-08-18.

This document condenses the official BatterySwapAI 2026 website information into
a local project reference. It is meant to help us avoid missing rules,
constraints, or submission details while building the solution.

Canonical source: the official NORA rules page takes precedence over participant
guides and this local summary if there is any conflict.

## Official Links

- Challenge page: https://www.nora.ai/competitions/batteryswapai/batteryswapai2026.html
- Documentation index: https://www.nora.ai/competitions/batteryswapai/docs/
- Canonical rules: https://www.nora.ai/competitions/batteryswapai/docs/04-competition-rules-and-prize-eligibility.html
- Example repository: https://huggingface.co/batteryswapaichallenge/BatterySwapAI2026-Example
- Hugging Face competition repository docs: https://huggingface.co/docs/competitions/main/competition_repo
- Hugging Face submission docs: https://huggingface.co/docs/competitions/main/submit
- Hugging Face leaderboard docs: https://huggingface.co/docs/competitions/main/leaderboard

## Challenge Story

We operate a fleet of battery-powered IoT sensor devices deployed across many
buildings over a wide area. As sensors run out of battery, we need to decide:

- which batteries need replacing;
- when each replacement should happen;
- how to create a realistic work-order plan that accounts for travel time,
  worker availability, and finite workdays.

The dataset comes from real sensors deployed across buildings in Norway. For
each battery, we get voltage and temperature time series. For a subset of
batteries, we also get the time when the battery ran out and the sensor stopped
functioning.

The full solution has two connected tasks:

- Task 1: estimate Remaining Useful Life (RUL) from battery time series.
- Task 2: create an automatic work-order plan for battery swaps.

Task 2 should use RUL estimates, but must also optimize the operational plan.
Good prediction alone is not enough because travel, batching, work hours, and
late replacement penalties all affect score.

## Who The Challenge Is For

The challenge targets data scientists, engineers, and researchers interested in:

- time-series forecasting;
- prognostics and health management;
- survival analysis / time-to-event modeling;
- decision-making under uncertainty;
- optimization and scheduling.

## Eligibility And Team Rules

- Open to residents of Norway.
- Participants may compete individually or in teams.
- Maximum team size is 8 people.
- Individual participants may form a team after registration.
- Team or entry changes must be reported to the organizers.
- Only registered individuals and registered teams may submit.
- Each registered individual or team counts as one competition entry.
- Each entry must use the restricted competition link supplied by organizers.
- The restricted link must not be shared outside the registered entry unless
  organizers explicitly permit it.
- Participants must not use extra accounts, registrations, or other entries to
  bypass access or submission limits.

## Official Schedule

- Competition launch and registration opening: 2026-05-13.
- Registration deadline: 2026-08-11.
- Dataset release: 2026-08-17.
- Final submission deadline: 2026-08-23.
- Winner announcement: 2026-09-01.

The competition page and official organizer notices define the actual cutoff
time and any later schedule amendments.

## Prizes

- Total cash prizes: NOK 50,000.
- First place: up to NOK 40,000.
- Second place: up to NOK 10,000.
- To be eligible for the full prize amount, at least one team member must be a
  Master's student, PhD candidate, or hold equivalent academic affiliation in
  Norway. For an individual entry, the individual must satisfy this condition.
- Prize candidates may need to document qualifying academic affiliation.
- Leaderboard position alone does not guarantee a prize; all rules must be
  satisfied.

Participants are also invited to submit a technical report to the peer-reviewed
journal Nordic Machine Intelligence (NMI).

## Dataset Overview

The provided data includes:

- battery/device location: room and building;
- battery time-series: voltage and temperature;
- travel times between buildings;
- battery end-of-life times;
- planning scenarios.

Only the train split is available to participants. Public and private splits are
used during official submission evaluation.

- Public split: public leaderboard.
- Private split: private leaderboard and final ranking.

The splits contain devices from different buildings, but are designed to be
otherwise similar. Our solution therefore must generalize to unseen devices and
unseen buildings.

## Dataset Files

### `devices.csv`

Describes which batteries/devices exist and where they are logically located.

Columns:

- `device_id`: primary identifier for the device/battery.
- `room_id`: room containing the device/battery.
- `building_id`: building containing the device/battery.
- `start_time`: when the device/battery was first deployed.

### `battery_metrics.parquet`

Hourly battery/device measurements aggregated from one-minute measurements.

Columns:

- `device_id`: primary identifier for the device/battery.
- `end_time`: measurement timestamp.
- `voltage`: measured voltage.
- `temperature`: measured temperature in Celsius.

Important notes:

- Measurements are one-hour averages.
- Gaps can occur, ranging from a few hours to multiple weeks.
- The solution must handle missing data.

### `eol_times.csv`

Observed battery end-of-life events.

Columns:

- `device_id`: primary identifier for the device/battery.
- `end_time`: timestamp when the battery hit the EOL criteria.

Many devices do not reach EOL before the dataset ends. These are right-censored
examples and should not be treated as known long-life failures.

### `scenarios.json`

List of planning scenario objects.

Each scenario has:

- `name`: scenario identifier used in submission files.
- `start_time`: start of the planning window.
- `settings`: scenario evaluation settings.
- `travel_costs`: travel costs between buildings for this scenario.

## Scenarios

For each split there are multiple planning scenarios. Each scenario involves the
same batteries, but at different points in time.

Official documentation states:

- 48 scenarios.
- Scenarios move forward week by week.
- Planning horizon is 6 weeks.
- Scenario locations/travel costs are randomized.

Because raw training data is available, it is possible to synthesize additional
training scenarios for local validation.

Local train diagnostics confirmed:

- Train has 48 scenarios.
- All train starts are Mondays at 00:00.
- Train scenario starts are seven days apart.
- Horizon is 42 days.
- With the evaluator's inclusive endpoint, there are 43 candidate in-window
  service dates: day 0 through day 42.

## Sensors

The devices are Soundsensing vibration sensors used for monitoring HVAC
equipment in commercial buildings.

Most devices are installed in air handling units, especially on fans/blowers and
motors for heat exchanger wheels. Some devices may be installed in other systems
such as pumps and compressors in heating/cooling systems.

The exact sensor placement is not included. Placement can affect ambient
temperature, which can affect voltage measurements and battery lifetime.

All devices in the dataset have the same firmware version and configuration.
They follow a fixed repetitive cycle:

- wake from sleep every minute;
- perform measurements;
- send data;
- return to sleep.

Measurement and transmission are deterministic and happen at the same point in
the cycle each time. The long-term process can be treated as approximately
constant load.

## Batteries

- Battery type: CR2477T coin cell.
- Chemistry: non-rechargeable Lithium Manganese Dioxide, Li-MnO2.

## Device Selection

- Devices were installed at different times during the dataset period.
- Dataset period stated by documentation: 2022-07-01 to 2026-07-31.
- Devices are limited to Norway.
- Devices were included if they belong to buildings where one or more devices
  reached EOL criteria.
- Many devices have not reached EOL by dataset end, creating censoring.

## End Of Life Criteria

A battery is considered to have reached EOL when voltage goes below 2.4 volts
for an extended time.

The official helper `smooth_series()` in `batteryswap_public.utils` defines the
details. The documentation suggests `smooth_series()` can be useful as a
preprocessing step.

Battery swaps should happen before EOL. RUL models should use the EOL point as
the prediction target.

## Dataset Loading

The official helper `batteryswap_public.utils.load_dataset()` is recommended for
loading the dataset. Example usage is available in the official example
repository.

The public Hugging Face dataset can also be loaded with:

```python
from datasets import load_dataset

ds = load_dataset("batteryswapaichallenge/BatterySwapAI-2026-Public")
```

Access may require Hugging Face login, for example with `huggingface-cli login`.

## Submission Overview

Each submission runs our code on the public and private dataset splits and then
updates leaderboard results.

Submission uses a Hugging Face model repository in `owner/repository-name`
format. The same repository may be used for multiple submissions.

Each submission is associated with a specific submitted Git commit SHA. Submitted
commits must remain in Git history and must not be rewritten or removed.

Before using New Submission, run local checks carefully because submissions are
limited.

## Submission Files And Entry Point

The repository must contain the files and entry point required by the official
example repository.

The important local entry point is `script.py`. The example repository uses:

- `batteryswap_public.utils.make_submissions()`;
- a pickled planner artifact;
- environment variable `BATTERYSWAP_DATASET_PATH`, defaulting to `/tmp/data`;
- environment variable `BATTERYSWAP_SPLITS`, defaulting to `public,private`;
- output file `submission.csv`.

Any trained models or artifacts used by `script.py` must be committed to the
submission repository.

## Planner Interface

The primary interface is a `Planner` class implementing:

```python
class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        timeseries: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings: EvaluationSettings,
    ) -> pd.DataFrame:
        ...
```

`make_submission()` / `make_submissions()` calls this planner for each scenario.

## Work-Order Plan Format

For each scenario, the planner returns a DataFrame with:

- `day`: service day;
- `battery`: battery identifier.

Example:

```csv
day,battery
2026-03-01,battery_1
2026-03-01,battery_13
2026-03-01,battery_2
2026-03-02,battery_3
```

Each row represents one battery swap. Rows are executed in the order provided.
If consecutive batteries are in different rooms or buildings, the plan implies
moving to those locations and paying the associated costs.

Plans must be complete: every battery in the scenario must appear exactly once.
To indicate that a battery should not be swapped inside the planning window,
place it on a day after the last day of the planning window.

Internal `submission.csv` contains additional split/scenario columns handled by
the helper utilities:

```csv
day,battery,scenario,split
2026-03-01,battery-1,s1,public
2026-03-01,battery-13,s1,public
2026-03-01,battery-2,s1,private
2026-03-02,battery-3,s1,private
```

## Cost Model

Lower score is better. The total score is the sum of cost components across the
full work plan.

Official cost components:

- time per battery change;
- time per room change;
- time per building change;
- travel time between locations;
- overtime when planned work exceeds an 8-hour workday;
- early replacement penalty;
- late replacement / downtime penalty;
- total worker-day availability limit.

Each day starts and ends at a base location.

Local evaluator inspection for `batteryswap_public==0.3.4` found these default
settings:

- planning window: 42 days;
- early replacement penalty: 0.5 per day early;
- late replacement penalty: 10 per day late;
- battery work time: 0.25 hours;
- room change time: 0.5 hours;
- building change time: 1.0 hour;
- overtime starts after 8 hours/day and is charged with factor 2;
- daily max and weekly max are enforced by large penalties in the evaluator.

The evaluator source remains the final technical reference:
`batteryswap_public.evaluate.EvaluationSettings` and
`batteryswap_public.evaluate.evaluate_plan()`.

## Submission And Compute Limits

- Maximum 5 submissions per day per registered competition entry.
- The platform determines when a submission day resets.
- Before final deadline, each entry must select up to 3 successfully evaluated
  submissions as final submissions.
- Failed submissions cannot be selected.
- At least one final submission must be selected by the deadline.
- Each evaluation run has maximum wall-clock runtime of 30 minutes.
- Evaluation is CPU only.
- A submission must run within 32 GB RAM.
- Participants must not bypass or interfere with time, memory, hardware, or
  submission limits.

## Available Packages

The competition runtime package set is fixed for all participants. Changing
`requirements.txt` does not change the official runtime.

The local official example repository lists these competition packages:

- `pandas`
- `plotly`
- `pydantic-settings`
- `structlog`
- `requests`
- `joblib`
- `fastparquet`
- `pyarrow`
- `tqdm`
- `scikit-learn`
- `huggingface_hub`
- `batteryswap_public`
- `scipy`
- `numpy`
- `statsmodels`
- `polars`
- `ortools`
- `lifelines`
- `torch`

Avoid adding custom dependencies unless organizers explicitly allow it.

## Local Testing

The official example recommends Docker testing because it matches the submission
environment.

Build:

```bash
docker build -t batteryswapai-2026-example .
```

Run:

```bash
docker run --name batteryswapai -v ./dataset:/tmp/data batteryswapai-2026-example bash -c "/app/env/bin/python3 script.py && /app/env/bin/python3 -m batteryswap_public.metric"
```

Copy generated submission:

```bash
docker cp batteryswapai:/app/submission.csv ./submission.csv
```

## Licensing Requirements

To be prize-eligible:

- participant-authored submission code must be released under MIT License;
- the public repository must contain a root-level `LICENSE` file with MIT text;
- third-party dependencies must have MIT, BSD-2-Clause, BSD-3-Clause,
  Apache-2.0, ISC, Zlib, or comparably permissive licenses;
- dependencies with other licenses require written organizer approval before
  deadline;
- third-party components retain original licenses and must be documented with
  source, version, and license;
- participants must have the right to use and redistribute all submitted code,
  artifacts, data-derived artifacts, and materials.

## Reproducibility Requirements

The public repository must include the full pipeline needed to reproduce every
trained or stored artifact used by the submitted commit.

Document, where applicable:

- data acquisition;
- preprocessing;
- feature generation;
- training or fine-tuning;
- model selection and configuration;
- serialization/export/conversion of stored models;
- inference and submission generation;
- environment and dependency specifications;
- exact commands;
- configuration files;
- random seeds;
- artifact identifiers;
- exact submitted commit.

Instructions must be detailed enough for organizers to reproduce the submission
in a clean environment.

## Pretrained Models

Pretrained models are allowed only when open source, documented, and
reproducible.

Document:

- model name and source;
- exact version, revision, or commit hash;
- license;
- how the artifact is obtained and used;
- preprocessing, fine-tuning, or conversion applied by the team.

Closed models, inaccessible artifacts, and external inference APIs cannot be part
of a prize-eligible submission.

## External Datasets

External datasets are allowed only when openly licensed, documented, and
reproducible.

Document:

- dataset name and source;
- exact version or release;
- license;
- files or records used;
- cleaning/filtering/transformation/combination steps.

The data must be publicly obtainable on equivalent terms by all participants.
Private, leaked, unlawfully obtained, or undisclosed evaluation data is
prohibited.

## Public Repository Deadline

For prize-claiming submissions:

- the Hugging Face model repository must be made public no later than 1 hour
  after the official competition deadline;
- the exact submitted commit, Git history, model artifacts, license information,
  and reproduction pipeline must be publicly accessible within that period;
- repository and commit must remain public during prize verification and until
  final results are announced.

## Prohibited Conduct

Participants must not:

- access, extract, reconstruct, or try to discover hidden evaluation data or
  labels outside the intended process;
- use leaked evaluation data, hidden labels, or predictions from unauthorized
  access;
- probe the evaluator or leaderboard to reconstruct hidden targets;
- hard-code predictions from hidden evaluation feedback;
- exploit, tamper with, overload, or interfere with the platform, evaluator,
  repositories, logs, scores, or another participant's work;
- use unauthorized network access, persistent processes, or mechanisms for
  compute or information outside the evaluation environment;
- bypass daily submission limits or other restrictions through multiple
  accounts, entries, teams, or repositories;
- conceal or misrepresent origin, license, version, or reproducibility of code,
  models, or data;
- include passwords, access tokens, private keys, or secrets in the repository.

Normal use of displayed score and feedback is allowed. Attempts to turn that
feedback into hidden target access are not.

## Enforcement

Organizers may reject, invalidate, or disqualify a submission, individual, or
team for violating rules, registration terms, or competition integrity.

A prize may be withheld if the submission fails licensing, publication,
reproducibility, provenance, compute, or conduct requirements.

Where facts are unclear, organizers may request source files, logs,
configuration, provenance records, or live reproduction.

## Practical Implications For Our Team

- Keep every official submission commit in Git history.
- Never force-push away submitted commits.
- Commit model artifacts used by `script.py`.
- Do not rely on network calls during evaluation.
- Stay within 30 minutes CPU and 32 GB RAM.
- Produce complete plans for every scenario.
- Build local validation around the exact public evaluator.
- Treat unseen building generalization as a core validation requirement.
- Keep a `LICENSE` file and third-party license notes ready before final
  submission.
