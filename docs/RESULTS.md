# Results and evidence

## Leaderboard: 5th place, 1468.44

**Team Dynamic duo placed 5th on the supplied leaderboard snapshot**, with a total score of **1468.44** for the submission timestamped **2026-08-23 23:58:31**. Relative to the initial recorded public score of **4252.33**, that is a **65.5% reduction**:

```text
(4252.33 - 1468.44) / 4252.33 × 100 = 65.47%
```

The latest result and rank were transcribed from a leaderboard image by a team member and supplied on 18 October 2026. The image itself is not stored here. The split, timezone, source commit, and number of ranked teams were not supplied. Accordingly, this is described as a **leaderboard result**, without claiming a final/private ranking or percentile. The percentage compares the recorded numbers; identical evaluation splits have not been established for the initial and latest entries.

![Leaderboard progression and 65.5 percent numerical score reduction.](assets/results.svg)

## Recorded progression

| Submission | Score ↓ | Evidence |
|---|---:|---|
| Initial v3 | 4252.33 | Historical public score in [handover](HANDOVER.md) |
| V6 | 2915.68 | Historical public score in [handover](HANDOVER.md) |
| Wiener model + planner budgets | 1985.43 | Public score for `7792b78`, recorded in [README snapshot](archive/COMPETITION_README.md) |
| Stale-transmission veto | 1555.80 | Public score for `89573f5`, recorded in [README snapshot](archive/COMPETITION_README.md) |
| **Latest supplied leaderboard entry** | **1468.44 · rank 5** | Team-provided transcription; submission 23 August 2026 |

The independently recorded public improvement from **4252.33 to 1555.80 is 63.4%**. The veto alone reduced public cost from **1985.43 to 1555.80 by 21.6%**. The repository does not yet tie the latest 1468.44 score to an exact commit, so it is not attributed to the current probability scale of 1.5.

These are competition objective values. They combine operational time and penalties and are not measured currency savings. No live leaderboard verification is implied.

## Latest leaderboard cost breakdown

The leaderboard labels the total **Total Time**. The components sum to **1468.4398**, matching **1468.44** after rounding.

| Component | Score contribution |
|---|---:|
| Battery swap | 4.2865 |
| Building change | 11.9792 |
| Room change | 7.1354 |
| Travel | 40.7476 |
| Overtime | 74.1871 |
| Daily limit | 89.5833 |
| Weekly limit | 66.6667 |
| Late swap | 623.5420 |
| Early swap | 550.3120 |
| **Total** | **1468.44** |

Early and late replacement account for **79.9%** of the score. That quantifies why identifying the right batteries and timing matters more than optimizing travel alone: travel contributes **2.8%** of this result. These percentages are derived from the supplied rounded components.

## Forecast quality: building-grouped out-of-fold evaluation

Source: the committed [V7 training report](v7_training_report.json), `scale_1.0` metrics.

| Metric | Value |
|---|---:|
| ROC-AUC | **0.9823** |
| PR-AUC / average precision | **0.4303** |
| Precision at 50 | **72.0%** |
| Precision at 100 | **66.0%** |
| Precision at 200 | **63.5%** |
| Event prevalence | **0.979%** |
| Cutoff examples | **88,013** |
| Positive cutoff examples | **862** |
| Increment windows used for fitting | **546,042** |
| Causal features | **64** |

Precision at 50 means **36 of the 50 highest-ranked cutoff examples were positive**. The top-50 event rate is approximately **73.5 times** the overall event prevalence. These are repeated device/time cutoff examples, not 88,013 independent batteries or 50 distinct maintenance jobs. Fold models hold out entire buildings. The calibration-scale sweep uses these same out-of-fold predictions; it is not an additional untouched test set.

Ranking quality is separate from calibration: the scale-1.0 model predicted **0.637 times** the observed event count on this population. A high ROC-AUC does not establish accurate probabilities or guarantee a better service schedule. The current submission also applies a separate **probability multiplier of 1.5** after inference; the metrics above predate that adjustment.

## Local operational evaluation

Historical V7 local results, recorded in the [archived competition README](archive/COMPETITION_README.md):

| Configuration | Mean total cost ↓ | Evaluation scope |
|---|---:|---|
| Defer all batteries | 3324.7 | 48 training scenarios |
| Earlier v3 | 2644.9 | 48 training scenarios |
| V6 hazard classifier | 2526.0 | 48 training scenarios |
| V7 Wiener model + planner | **2293.2** | 48 scenarios, building-grouped out-of-fold forecasting |
| Perfect-information planner reference | 77.8 | **Only scenarios 0–11**, using true end-of-life information |

V7 reduced mean local cost **31.0% vs. no service** and **13.3% vs. v3**. These describe historical configurations, not fresh measurements of the current defaults. The perfect-information reference is a diagnostic bound on a smaller subset and is not a deployable result.

Adjacent scenarios start one week apart and cover six weeks, so windows overlap by about 85%. The 48 scores are not independent samples. Local/public cost composition also differs; the [case study](CASE_STUDY.md) explains why local gains are not advertised as public improvements.

## Reproduce or inspect

- [Machine-readable score ledger](results.json): source values used to generate the README chart.
- [Saved training report](v7_training_report.json): forecast metrics and fitting dimensions.
- [Reproducibility guide](REPRODUCIBILITY.md): current commands, configurations, and data requirements.
- `python tools/check_portfolio.py`: checks the score calculations, saved metric claims, and documentation links.
- `python tools/render_results.py`: regenerates the chart; requires matplotlib.

The historical evidence files are preserved rather than rewritten to match newer defaults. New benchmark runs should write a new report with its mode and complete configuration.
