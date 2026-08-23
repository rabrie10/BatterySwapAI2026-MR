# Is V8 modelling the wrong event? Reconstructing the official EOL rule

_Branch `claude/final-j2w-precision`, 2026-08-23. The last structural hypothesis:
V8 predicts first passage below 2.4 V while the competition's EOL is *sustained*
low voltage after smoothing, and the two would differ exactly for the
near-threshold devices that sit at 2.402-2.416 V for months._

**The hypothesis is falsified at the definition, and a second arithmetic check
closes the weaker version of it too. No simulator was built.**

---

## 1. The rule, reconstructed

`batteryswap_public` *consumes* `eol_times.csv` and never generates it, so the
rule has to be recovered from the data. `smooth_series` is the only transform
involved: readings filtered to 10 < T < 30 degC, daily median requiring at least
5 readings, then `rolling(7, min_periods=3).median()`.

For each of the 82 devices with a recorded EOL, comparing the recorded date
against the first day its smoothed series goes below 2.4 V:

| | |
|---|---:|
| devices matching **to the day** | **81 of 82** |
| mean / sd of (recorded − first crossing) | **0.0 / 0.0** |
| the one mismatch | `d_cfdd8b69ddd4`, minimum smoothed **2.4000000953674316** |

That single exception is a float32 storage artefact of the cached series, not a
rule difference: its true minimum is fractionally under the threshold.

Two further checks make it airtight:

| | |
|---|---:|
| devices whose smoothed series ever goes below 2.4 V | **81** |
| of those, devices with **no** EOL record | **0** |
| devices that go below 2.4 V and later **recover above it** | **47 of 81** |

The last row is the one that settles it. If EOL required sustained low voltage,
those 47 recoveries would either carry no record or a later one. Every one of
them is recorded on the day of its **first** touch.

```
EOL = the first day the smoothed series is below 2.4 V. There is no persistence
rule. Confirmed on all 82 recorded devices.
```

**The premise was half right, and the half that is right is already handled.**
Persistence *is* in the definition -- a 7-day rolling median needs roughly four
of seven daily medians under the threshold before it dips -- but it lives inside
the construction of the series, not as an extra condition on top of it. And V8's
state is `m(t) = smooth_v(t) - 2.4` on that same smoothed series. **V8 is already
modelling the official event.**

### A corollary worth recording

Because a dip-and-recover *is* an EOL, the reflection term in
`first_passage_probability` -- which counts paths that touch the barrier and come
back up -- is not over-counting. It is exactly right. That independently explains
`docs/V10_FINDINGS.md`'s measurement that `reflection_weight = 0` "does not kill
zombies and wrecks mid-range calibration": weight 0 prices the endpoint only,
which is the *wrong* event.

## 2. Even the weaker version cannot work

Suppose the rule had been about geometry rather than persistence -- that V8's
continuous-time first passage over-counts against a discrete daily minimum. Any
simulation of the correct event must still reproduce the probability that the
path simply *ends* below the barrier, so the endpoint term is a floor on what
any such correction can achieve.

Decomposing V8's probability into its endpoint and reflection terms on the six
devices that carry half of all wasted swaps (medians over their 48 scenario rows
each):

| device | rows | ever due | margin | predicted 42 d drop | sigma | endpoint | reflection | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d_b5b678a3f79f | 48 | **0** | 0.0049 | 0.0451 | 0.0214 | **0.970** | 0.026 | 0.931 |
| d_3d26e12378f1 | 48 | **0** | 0.0291 | 0.0379 | 0.0284 | **0.622** | 0.141 | 0.689 |
| d_c9a2ce794b68 | 48 | **0** | 0.0085 | 0.0146 | 0.0357 | **0.568** | 0.315 | 0.795 |
| d_d4b4272d5229 | 48 | **0** | 0.0021 | 0.0000 | 0.0343 | 0.465 | 0.465 | 0.440 |
| d_a85bae19463d | 48 | **0** | 0.0626 | 0.0190 | 0.0505 | 0.193 | 0.135 | 0.273 |
| d_d9d695df1683 | 48 | **0** | 0.0526 | 0.0228 | 0.0296 | 0.157 | 0.084 | 0.200 |

Pooled over the six: **endpoint 0.388, reflection 0.084, realised due rate
0.000** across 288 scenario rows.

**Deleting the reflection term entirely** -- a more aggressive correction than
any persistence simulation could be, and one that would model the wrong event --
leaves their probability at **0.388 against a truth of 0.000**. Gate 1
("persistent-EOL probability should fall materially on the zombies") cannot pass,
because the over-confidence is not in the passage geometry.

## 3. Where the over-confidence actually is

The same table names it. For these six the drift regressor predicts a **median
42-day fall of 0.0228 V against a median margin of 0.0291 V**, and predicts a
fall exceeding the margin on **38.2 %** of their rows. The model expects them to
cross because it expects them to *move*, and they do not.

`d_d4b4272d5229` is the clean illustration of the residual case: predicted drift
is exactly 0.0000, so its probability of 0.44 is entirely diffusive -- pure
volatility against a 2.1 mV margin. Simulating the correct event under the same
sigma reproduces that number. Only a smaller sigma changes it, and that is the
scatter model, not the event definition.

So the zombie problem is a **dynamics** error: drift too steep, and sigma too
large relative to how quietly these particular cells actually behave. It is not
a target error and not a geometry error.

## 4. Verdict

```
GATE FAILED before it could be run: the premise is false. Task 1 is closed.
```

No path simulator was built, no planner run was spent, and `script.py` is
untouched.

This is the fourth structural attempt on this branch and the cheapest to settle.
For the record, all four failed for different and now-understood reasons:

| attempt | why it failed |
|---|---|
| residual losses on the engineered state (`FINAL_RESIDUAL_OBJECTIVES`) | 373 positives from 75 devices; 1 of 18 fits beat V8 out of fold |
| matched-volatility state (`FINAL_TERMINALITY`) | real and 5/5 folds, but only orders at matched margin, which is 36 % of decisions |
| EOL-aligned trajectory templates (`FINAL_TEMPLATES`) | same shape, larger matched-margin edge, measured +5.54 through the planner |
| the EOL event definition (this document) | V8 already models the official event exactly |

And the dynamics defect this document isolates has itself been attacked and
closed: the drift model is not data-starved (the stride-2/400 retrain measured
+57 and +106 in two independent generations), the censor-aware target that
un-biases it at the knee is V10's and was measured, and adding the volatility
ratio to drift and scatter buys +0.05 % out of fold across 275,951 windows
(`FINAL_TERMINALITY` section 7).
