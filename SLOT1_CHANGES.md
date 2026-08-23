# Slot 1 — stale-transmission veto (shipped)

**Commit:** `89573f5bf9a175f4392fe5cffffdf8ffeb1048db`
**Parent:** `7792b780f329c410966623b215c019551447ed96` (V8 Task 1 model + V10 planner, public 1985.43)
**Remote:** pushed to `origin/main` (Hugging Face `Rabrie10/BatterySwapAI2026-MR`), fast-forward
**Files changed:** 2 — `bsai/forecaster.py`, `script.py`. `batteryswap_solution/` untouched.

---

## Change 1 — stale-transmission veto

**File: `bsai/forecaster.py`** (the Task 1 → Task 2 forecast contract adapter)

A battery is withdrawn from service consideration when **both** hold at the prediction cutoff:

```
v_stale_days > 14     AND     margin < 0.1
```

### Rationale

EOL is defined *from the timeseries*. A device that has stopped transmitting cannot record a
2.4 V crossing, so it cannot become due, so every swap spent on one is waste. Within the
near-threshold band the due rate falls from 30.8% while transmitting to roughly 5% past 14
days of silence, against a break-even service probability near 15%.

The effect exists **only** inside the near-threshold band — across the whole population
staleness barely moves the due rate — which is why transmission completeness was rejected
earlier as a *ranking* feature. Here it is used as a **veto justified by cost asymmetry**,
not as a ranker.

### The four edits

**1. Constants — lines 56–59**

```python
EOL_VOLTAGE = 2.4
STALE_DAYS_LIMIT = 14.0
STALE_MARGIN_LIMIT = 0.1
_VOLTAGE_COLUMN = FEATURE_NAMES.index("voltage")
```

`FEATURE_NAMES` was added to the existing `from .features import (...)` block. No new
dependency — it already lived in `bsai/features.py`.

**2. Per-battery silence, collected in the existing loop — lines 182–187**

```python
last_seen = int(view.last_valid[index])
stale_days.append(
    float(origin_ordinal - (series.origin + last_seen))
    if last_seen >= 0
    else float("inf")
)
```

**3. Vectorised veto mask — lines 207–210**

```python
margin = design[:, _VOLTAGE_COLUMN].astype(float) - EOL_VOLTAGE
veto[index] = (
    np.asarray(stale_days, dtype=float) > STALE_DAYS_LIMIT
) & (margin < STALE_MARGIN_LIMIT)
```

**4. Application, as post-processing on the assembled arrays — lines 227, 236–237**

```python
if veto.any():
    daily[veto] = 0.0
...
unobserved    = np.clip(1.0 - censor, 0.0, 1.0)
observed_tail = np.clip(1.0 - horizon_cdf - unobserved, 0.0, 1.0)
```

The vetoed battery's `failure_cdf` mass moves into `prob_observed_after_horizon` —
the model now says "expected to still be observed and un-failed after the horizon" rather
than "will fail". `prob_unobserved_eol` is untouched.

---

## Why the existing `staleness` feature could not be reused

`bsai/features.py` computes 64 causal features, and feature 2 is called **`staleness`** —
which looks like exactly the right quantity. **It is not**, and reusing it would have made the
veto silently never fire.

`DeviceView.value_at_or_before` clamps before measuring:

```python
index = min(index, self.size - 1)
position = int(self.last_valid[index])
return float(self.voltage[position]), index - position
```

The smoothed series **ends where the device stops transmitting**, and `predict()` separately
clamps `index = min(index, len(series) - 1)`. So `staleness` can only see gaps *inside* the
series and reads ≈0 for precisely the stopped devices this rule targets.

The shipped code therefore computes silence against the **unclamped** cutoff ordinal:

```
v_stale_days = origin_ordinal - (series.origin + last_valid[index])
```

This captures both within-series gaps and the silence beyond the series end.

**Causality:** `DeviceView.last_valid` is `np.maximum.accumulate(...)` — a prefix statistic, so
entry *i* depends only on days up to *i*. No statistic over a battery's complete series enters.
No forward-fill, no resampling, no reindexing of the daily grid, no second pass over raw hourly
data, and no change to the incremental smoothing cache.

### Feature names used

| quantity | source |
|---|---|
| `margin` | feature 0 **`voltage`** (last smoothed voltage at or before cutoff), minus 2.4 |
| `v_stale_days` | **computed** — feature 2 `staleness` is not equivalent (see above); derived from the already-computed `DeviceView.last_valid` |

---

## Contract safety

The forecast contract in `batteryswap_solution/forecast.py` requires
`final_cdf + prob_observed_after_horizon + prob_unobserved_eol == 1`, with `failure_cdf`
monotone non-decreasing in `forecast_date`. Both are satisfied **by construction**:

- The vetoed row is set to a **constant zero** curve, not a rescaled one, so it is trivially
  monotone.
- `prob_observed_after_horizon` is computed as the **residual** `1 - final_cdf - unobserved`
  rather than by adding the removed mass back, so floating-point drift cannot break the
  identity. For a vetoed row: `0 + censor + (1 - censor) = 1` exactly.
- `grid` is deliberately left untouched — it still feeds `censor` and the mean-excess
  integral, so `mean_excess_rul_days_given_observed_after_horizon` stays non-negative.

**Note:** satisfying the residual form required reordering two pre-existing lines for *all*
batteries, not only vetoed ones. This is algebraically a no-op for non-vetoed rows — the old
code produced `observed_tail = censor - horizon_cdf`, `unobserved = 1 - censor`, and the new
code produces the identical pair — and it subsumes the "absorb interpolation slack" step the
old duplicate assignment existed for.

---

## Change 2 — volatility scale pin

**File: `script.py`**, inside `load_forecaster()` — line 67

```python
model = joblib.load(path)
if hasattr(model, "volatility_scale"):
    model.volatility_scale = 1.0
return HazardForecaster(model)
```

`tools/train_wiener.py` selects `best_scale = 1.4` and writes it onto the model it dumps,
while the shipped `models/v7_wiener.joblib` carries **1.0**. Pinning guards the submission
against a regenerated artifact silently arriving at 1.4, which measured worse.

`robust_emergency_samples` was already an explicit `0` in `script.py` and was left alone.
Nothing else was changed — in particular not `max_planned_rate`, the late risk multiplier, the
probability scale, or the capacity repair budget, all four of which were measured dead across
25 configurations.

---

## Verification performed

Run against the committed tree, all read-only:

| check | result |
|---|---|
| Model load | `WienerModel`, `volatility_scale=1.0` — no voltage-trend fallback |
| Forecast contract | **48/48** scenarios pass `batteryswap_solution.forecast.validate_forecast` |
| Veto fire rate | **mean 9.58/scenario, 460 total** (expected ≈9.9 / 473) |
| Planner run | 458 rows, non-empty, **0 fallbacks** |
| Unit tests (pre-veto build) | 69 run, OK, 11 skipped |

The 460-vs-473 gap is ~3% and consistent with a boundary detail (a device at exactly 14 days,
or a cold-start row excluded before the mask), not with a mis-mapped quantity — the earlier
wrong mapping missed by an order of magnitude, not by 13 fires.

---

## Known issue — recorded, not fixed

`tools/train_wiener.py:189` selects `best_scale` and line 202 assigns it to the production
model before dumping. That value is **1.4**, while the shipped `models/v7_wiener.joblib`
carries **1.0**; a model regenerated from byte-identical code carries 1.4. **The README's
reproduce command therefore does not reproduce the shipped artifact.** Training report:
scale 1.0 → predicted/actual 0.637, scale 1.4 → 1.025, at effectively identical AUC
(0.9823 / 0.9821). Deliberately left unfixed and out of this commit.

## Not verified

No before/after cost comparison was run for this change. The local `submission.csv` is a
**train** split from Aug 20 that predates `7792b78` and was produced by an older
configuration, so it is not a clean baseline. No public/private data exists locally, so a
submission CSV cannot be generated here — the evaluation harness produces it. The public
leaderboard is the first measurement of this build.
