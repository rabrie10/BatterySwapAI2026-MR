# V7: close the gap to first place

Written 2026-08-21 against the first real leaderboard signal.

## 1. What the leaderboard actually says

Our V6 submission scored **2915.68**, down from v3's 4252.33 — a 31% gain, and
`early_swap` fell 2434.99 -> 1222.85, almost exactly halved. The censoring-label
fix transferred as predicted. We are still 17th of 19; first place is 1160.67.

Decomposing our 1755.0-point gap to J2W:

| component | us | J2W | gap | share |
|---|---:|---:|---:|---:|
| early_swap | 1222.8 | 302.7 | **+920.1** | 52.4% |
| late_swap | 1031.7 | 605.8 | **+425.8** | 24.3% |
| daily_limit | 235.4 | 85.4 | +150.0 | 8.5% |
| weekly_limit | 154.2 | 39.6 | +114.6 | 6.5% |
| overtime | 158.0 | 68.3 | +89.7 | 5.1% |
| travel | 78.2 | 38.3 | +39.9 | 2.3% |
| building_change | 18.1 | 10.5 | +7.6 | 0.4% |
| room_change | 10.9 | 6.3 | +4.6 | 0.3% |
| battery_swap | 6.4 | 3.7 | +2.6 | 0.2% |

`battery_swap` is 0.25 h per swap and includes the emergency swap for every
missed battery, so it counts *all* swaps. `late_swap` is 10 per day late, and a
missed battery runs about 27 days late (measured out-of-fold on train). That
recovers the counts:

| team | swaps total | planned | missed | early per planned swap |
|---|---:|---:|---:|---:|
| J2W | 14.9 | 12.6 | 2.24 | **24.0** |
| YassY | 17.5 | 15.4 | 2.17 | 32.3 |
| astar | 18.9 | 16.7 | 2.23 | 36.2 |
| thibautforest | 12.1 | 7.6 | 4.49 | 18.1 |
| **us** | **25.4** | **21.6** | **3.82** | **56.6** |

## 2. The entire gap is one number

**Routing is not our problem.** Building changes per planned swap: 0.839 for us,
0.833 for J2W — identical. Travel per swap 3.62 against 3.03, marginally worse.

**Swap count is our problem.** We plan 71% more swaps than J2W *and still miss
70% more due batteries*. That is strictly dominated: more work, worse outcome.

**Early cost per planned swap, 56.6 against 24.0**, says what those extra swaps
are: batteries far from their end of life. A swap on a battery that never dies
costs `0.5 x (substitute EOL - swap day)`, which runs to 85+ points. A swap on a
genuinely due battery costs a few points. Our swaps are mostly the former.

The capacity penalties follow mechanically. The far building is a 20.5 h round
trip against a 24 h *weekly* limit, so every extra trip risks a flat 100. Those
354 points of overtime and limit penalties are a symptom of swapping 9 batteries
too many, not an independent defect.

So: **the 1755-point gap is a forecast discrimination problem.** Precision, not
calibration, not scheduling. Matching J2W's operating point would move roughly
765 of early, 430 of late, and about 260 of capacity and logistics that scale
with swap count.

## 3. Why more feature engineering will not fix it

The model is a rare-event classifier and the train split contains **82 EOL
events**. That is the effective sample size, whatever the row count says. V6
already stacks 2.8 M rows over 24 horizons, and the horizon-42 PR-AUC is 0.405.
Adding features to an 82-event problem buys very little, which is what the V6
knob sweeps showed: eight separate interventions all landed inside the noise
floor.

## 4. The change: regress the margin, do not classify the event

End of life is defined as `smooth_series(voltage) < 2.4`. The smoothed series is
observed for **26,366 device-days**. The quantity that decides EOL — the running
minimum of the voltage margin — is therefore observable for *every* device-day,
not just the 82 on which the event fires.

For device `b`, cutoff `t`, horizon `k`:

```
observable = min(k, E_b - t)                 # E_b: last day a record can exist
y(b, t, k) = min over j in 1..observable of [ smooth_v(b, t+j) - 2.4 ]

EOL recorded within k   <=>   y(b, t, k) < 0
```

This is an exact restatement, not an approximation, and it has three properties
the classifier does not:

1. **~300x the supervision.** Every device-day carries a continuous target.
2. **No information thrown away.** A battery that ends the window at margin 0.004
   and one that ends at 0.31 are both "negative" to a classifier; the regression
   sees that the first was one bad week from crossing.
3. **Censoring is exact and needs no feature.** No record can be filed after
   `E_b`, so truncating the minimum there *is* the definition. V6 needed a
   `remaining_observation_days` feature to patch this; here it falls out.

Probability comes from quantile regression: fit `y` at quantiles
`{0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7}` with horizon as a feature, then read
`P(y < 0)` off the fitted quantile curve by interpolation. Monotone constraint in
horizon, because a longer window can only push the running minimum down.

## 5. Build order

1. **`bsai/margin.py`** — the target builder and the quantile model.
2. **Out-of-fold validation by building**, same protocol as V6. The acceptance
   test is horizon-42 PR-AUC and, above all, **precision at the swap counts we
   actually operate at** (10-25 swaps per scenario), because that is what the
   leaderboard charges us for.
3. **Ensemble** with the V6 hazard classifier if the two are decorrelated and the
   stack beats both out-of-fold. Keep whichever wins; do not keep both out of
   sentiment.
4. **Re-validate end to end** with the unchanged Task 2 planner and confirm the
   swap count falls toward 13-16 with recall held or improved.

## 6. Acceptance criteria

Ship only if, out-of-fold on all 48 train scenarios:

- mean total cost beats V6's 2526.0 by more than the ~100-point noise floor;
- planned swaps per scenario fall below 18 **without** recall dropping;
- early cost per planned swap falls below 40 (V6: local equivalent of 56.6);
- runtime stays under 15 minutes projected for 96 scenarios.

If the margin model does not beat V6 out-of-fold, it does not ship, and this
document records why.

## 7. Explicitly not doing

- **Threshold tuning.** Our precision at the current operating point is roughly
  at break-even (a wasted swap costs about 87, a missed one about 270, so the
  break-even probability is 87/357 = 0.26). Both we and J2W sit near our own
  marginal break-even; theirs is better because their ranking is better. Moving
  the threshold trades early against late along the same curve and gains little.
- **Rebuilding Task 2.** Per-swap routing efficiency already matches first place.
- **A neural sequence model.** 82 events cannot support it, and the margin
  regression captures the same trajectory information with far less variance.
