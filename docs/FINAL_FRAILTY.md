# Device-level frailty: does surviving V8's own hazard carry information?

_Branch `claude/final-j2w-precision`, 2026-08-23. Code: `tools/fj_frailty.py`,
`tests/test_frailty.py` (8 tests). No frailty model built, no planner run._

V8 is memoryless. At every cutoff it reads the current state and returns a
probability; nothing carries forward the fact that it assigned the same device a
high hazard last week, and the week before, and was wrong both times. If the six
repeat false positives are individually frail rather than evidence of wrong
population dynamics -- and `docs/FINAL_PROCESS.md` showed the population dynamics
are calibrated to within 3 % -- then the accumulated hazard a device has
*survived* is a legitimate posterior update, and it is exactly what V8 discards.

**The hypothesis is right as a description and fails as a ranker.** The signal is
real, it is not manufactured by the six devices, and it does not improve
cross-margin ordering. The reason is quantified in section 4.

---

## 1. Construction, and the leakage rules

Per device, from its own past telemetry only:

* **weekly** pseudo-cutoffs across the whole history, so the seven-day intervals
  tile the past without overlapping and no risk is counted twice;
* at each, the frozen V8 fold model **that never saw that device's building**,
  evaluated at a 7-day horizon;
* the device is observably active at the scenario cutoff, so every *completed*
  weekly interval before it is a survival observation;
* `H_surv = sum over completed weeks of -log(1 - p7(s))`.

50,365 weekly pseudo-cutoffs, 100 % coverage of the 19,890 scenario rows.

**The probability is the raw first-passage value, before
`RemainingCalibration`.** That correction keys on `end_time - cutoff`, and
`end_time` is the dataset export date -- a fact about the future at a historical
pseudo-cutoff, even though the deployed model legitimately receives it inside a
scenario. Dropping it keeps every input causal.

No device identity, no EOL label, no scenario outcome, no overlapping horizons.
`tests/test_frailty.py` pins the four ways this could have been invalid: counting
an unfinished week, double-counting overlapping intervals, reading anything at or
after the cutoff, and treating "no history" as "zero hazard survived".

The distribution is extreme, as the hypothesis predicts: median `H_surv` is 0.000
and the maximum is **42.36** -- one device survived a stretch V8 gave odds of
e^-42.

## 2. Falsification: it separates, and not because of the six

Among V8 high-risk rows (`p >= 0.20`): 659 rows, 104 devices, due rate 0.361, of
which the six repeats are 209 rows (31.7 %).

AUC for "lower survived hazard implies genuinely due" -- above 0.5 means frailty
works:

| feature | row AUC | row, **ex-six** | device AUC | device, **ex-six** |
|---|---:|---:|---:|---:|
| `H_last_4` | 0.6461 | 0.5823 | 0.6878 | **0.6818** |
| `H_surv_log1p` | 0.6319 | 0.5948 | 0.6712 | **0.6692** |
| `H_last_8` | 0.6377 | 0.5865 | 0.6793 | **0.6760** |
| `weeks_over_20` | 0.6621 | 0.6038 | 0.6785 | 0.6635 |
| `max_prior_p7` | 0.6399 | 0.5847 | 0.6599 | 0.6500 |
| _`weeks_observed` (control)_ | _0.6984_ | _0.5064_ | _0.6186_ | _0.5628_ |

**Device-level AUC barely moves when the six are removed** -- 0.6878 to 0.6818
for `H_last_4`. The control row is the point of including it: `weeks_observed`,
which is just how long a device has been around, looks strong pooled and
collapses to 0.506 without the six. That is what a six-device artefact looks
like, and the `H_surv` family does not look like it.

Medians among high-risk rows: `H_surv` **0.212 for genuinely due against 1.229
for survivors**, a six-fold separation in the predicted direction.

**This is the first auxiliary signal on this branch whose device-level
separation survives removing the six.** Gate 2 passes.

## 3. The cross-margin gate: fails, with the interval excluding zero

Same landmark population as every other candidate (top 40 by V8 probability per
scenario, censored rows excluded), where **V8 scores 0.7280 overall and 0.7359
cross-margin**.

Gamma frailty in logit space, `logit(p_V8) - log1p(theta * H_surv)`, which is the
form the multiplicative hazard `lambda_i = z_i * lambda_V8` implies with
`E[z | survived H] = 1 / (1 + theta H)`:

| form | all | **cross-margin** | same-margin | f0 | f1 | f2 | f3 | f4 | folds > V8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V8** | **0.7280** | **0.7359** | **0.5846** | 0.945 | 0.799 | 0.616 | 0.776 | 0.692 | — |
| theta = 0.1 | 0.7239 | 0.7308 | 0.5985 | 0.945 | 0.799 | 0.635 | 0.773 | 0.689 | 1/5 |
| theta = 0.5 | 0.7149 | 0.7206 | 0.6115 | 0.945 | 0.804 | 0.672 | 0.768 | 0.681 | 2/5 |
| theta = 1.0 | 0.7097 | 0.7155 | 0.6041 | 0.961 | 0.804 | 0.683 | 0.765 | 0.680 | 3/5 |
| theta = 5.0 | 0.6906 | 0.6952 | 0.6059 | 0.922 | 0.793 | 0.698 | 0.766 | 0.667 | 1/5 |

Rank-space blending is worse still (best cross-margin 0.7102). Every setting of
every summary degrades cross-margin ordering, monotonically in theta.

Device bootstrap of the cross-margin delta, 300 draws:

| | delta [95 % CI] | P(delta > 0) |
|---|---|---:|
| theta = 0.5 | **-0.0160 [-0.0293, -0.0014]** | 0.01 |
| theta = 1.0 | **-0.0209 [-0.0381, -0.0032]** | 0.00 |

**Both intervals exclude zero on the negative side.** The gate does not merely
fail to pass; it fails with confidence.

Same-margin concordance improves (0.5846 to 0.6115) -- the fourth signal on this
branch to help once margin is held fixed and hurt across it.

```
GATE 3 FAILED. No frailty model built, no planner run. Direction closed.
```

## 4. Why: survived hazard is not specific to the frail

Median probability at `p >= 0.20`, under the frailty:

| theta | the six repeats | all *other* high-risk devices |
|---:|---:|---:|
| 0 (V8) | 0.711 | 0.438 |
| 0.5 | 0.568 | 0.365 |
| 1.0 | 0.504 | 0.331 |
| 2.0 | 0.412 | 0.292 |
| 5.0 | 0.325 | 0.246 |

The other high-risk devices have a realised due rate of **0.529** -- they are
mostly genuine failures. And the frailty demotes them at almost exactly the rate
it demotes the six: at theta = 1.0, the six fall 29 % and the true positives fall
24 %.

That is the whole result. **Survived hazard is high for anyone who has spent time
in a high-hazard state, and a battery on its way to failing spends time in a
high-hazard state before it fails.** The quantity does not distinguish "frail" from
"further along the same road", so no setting of theta buys the six without paying
nearly the same price on the population that actually dies.

It is the arithmetic `docs/task1_investigation_findings.md` states for dwell --
"demoting a zombie at margin 0.02 (due rate 0.24) frees a slot filled by a margin
0.10 candidate (due rate 0.17), a loss every time" -- reappearing for a
better-motivated signal, and now measured directly rather than inferred.

## 5. Verdict

The frailty hypothesis is **correct as a description and useless as a ranker**.
There is genuine device-level heterogeneity that V8's memoryless form cannot
express, it is detectable at device-level AUC 0.68 without the six devices
driving it, and converting it into a ranking makes held-out-building ordering
worse with 95 % confidence.

Sixth structural attempt on this branch, sixth distinct failure mode:

| attempt | outcome |
|---|---|
| residual losses on the engineered state | 1 of 18 fits beat V8 out of fold |
| matched-volatility state | real, 5/5 folds, orders only at matched margin |
| EOL-aligned trajectory templates | larger matched edge, +5.54 through the planner |
| the EOL event definition | V8 already models the official event exactly |
| the path law | calibrated 1.00 where it acts; the zombies are six devices |
| **device-level frailty** | **separates at device level, cross-margin delta -0.021 [-0.038, -0.003]** |
