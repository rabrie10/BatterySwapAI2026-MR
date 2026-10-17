# Case study: connecting battery prognostics to maintenance decisions

**Outcome:** rank **5** on the supplied leaderboard snapshot, with score **1468.44**, compared with an initial recorded public score of **4252.33**. The numerical reduction is **65.5%**. [Evidence, scope, and component breakdown](RESULTS.md).

**Team:** Mahdi Nasser and Robel Weldemichael Ghebremedhin, Team Dynamic Duo. Both authors collaborated across forecasting, optimization, experiments, and integration throughout the project.

**Stack:** Python, NumPy, pandas, SciPy, scikit-learn, OR-Tools CP-SAT, joblib, unittest, Docker, GitHub Actions.

## A forecast only matters if it changes the right decision

The task was to maintain battery-powered building sensors using voltage and temperature histories. The system needed to select batteries, schedule work over six weeks, and account for travel and worker capacity. The evaluator charges 0.5 per early day and 10 per late day, alongside work, travel, overtime, capacity, and emergency penalties.

That asymmetry changes the learning objective. A single expected remaining lifetime obscures the probability of a costly near-term failure. The solution instead predicts a failure-time distribution and integrates it into expected service and deferral costs. The planner can then compare a preventive visit with the risk of leaving a battery untouched.

## Find signal before adding model complexity

The early pipeline used features of smoothed voltage. A simple margin-over-slope control matched a much larger hazard classifier at precision@12: **0.309 versus 0.300**, as recorded in the [Wiener training notes](../tools/train_wiener.py). That control suggested the feature representation, rather than classifier complexity, was the bottleneck.

The next approach retained within-day voltage response to temperature. Daily smoothing removes that variation, even though it can reveal changing battery behavior. A Wiener first-passage model then learned drift and volatility from **546,042 increment windows**, while predicting the probability of crossing the failure threshold at multiple horizons.

The saved building-grouped out-of-fold report records **ROC-AUC 0.9823**, **PR-AUC 0.4303**, and **72% precision at 50** across **88,013 cutoff examples** with a **0.979%** event rate. These ranking results support the representation choice; they do not establish calibration or public operational savings on their own.

## Make the planner reason about the evaluator's actual costs

The optimizer jointly chooses service or deferral and the day of service. It combines OR-Tools CP-SAT with candidate reduction, local search, and building-route optimization. This captures interactions that independent per-battery rules miss: a battery that is expensive to visit alone can be worthwhile when another service already brings the worker to the same building.

The scoring model also treats capacity overruns as penalties where the evaluator does, rather than forbidding them universally. Avoiding a flat capacity penalty can be the wrong choice when it causes a larger emergency cost.

The final plan must include every active battery exactly once, including deferred work. A versioned forecast contract and official evaluator checks make that boundary explicit and testable.

## Investigate failures in the measurement process

The public leaderboard exposed a mismatch in early versus late penalties that local validation did not capture. The team examined the population being serviced, including observation ending and devices that stopped transmitting.

A stale-transmission veto for near-threshold devices reduced the recorded public score from **1985.43 to 1555.80: a 21.6% improvement**. The reasoning is tied to the competition's definition of a recorded failure. In a deployed maintenance system, silence would need a separate communication-loss or missing-data policy; it cannot establish that a physical battery is healthy.

The latest supplied score, **1468.44**, contains **79.9% early and late replacement costs** but only **2.8% travel costs**. That breakdown reinforces the central engineering lesson: selecting and timing service has greater remaining impact here than making routes marginally shorter.

## What makes the evaluation credible

Buildings are held out together because the competition splits involve unseen buildings. Random row splits would let device and building history cross the validation boundary. Overlapping weekly scenario starts are also acknowledged: 48 scenario scores are not 48 independent observations.

The repository preserves failed approaches, training reports, historical configurations, and local/public discrepancies. Current training, validation, and submission now share shipped settings, and new validation reports record their full arguments. The synthetic demo and dataset-independent regression suite provide a lightweight way to inspect the engineering without redistributing competition data.

## Limits and next steps

This is a competition system, not evidence of a deployed maintenance service. Its cost units combine time and penalties, and no monetary savings are claimed. The latest leaderboard transcription does not identify its split or exact source commit. The current model's probability scaling and observation rules need separate validation before transfer to another fleet.

Useful next experiments would isolate improvements under matched configurations, assess calibration on new buildings and seasons, and model communication loss separately from battery failure. Those are future research directions; they are not included in the reported results.
