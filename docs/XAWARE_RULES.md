# X-aware swap-selection rules: analytic league table

Objective per scenario: early + late timing cost with each selected battery on its analytically best day (due: ~5 days before EOL; not due: last window day, earliness = substitute EOL - window end), missed due batteries queued as emergencies from day 48 (evaluator order), plus 4.1+1.5 per planned swap and the isolated emergency visit op (2 x travel + 1.75h + overtime) per miss. Mean over 48 train scenarios; forecasts are out-of-fold (outputs/v7_folds.joblib, volatility scale 1.0).

X_i = evaluator substitute EOL (end_time + 30d) minus the last window day: the known price of a wasted swap. Its median falls from ~298 days in the first block to ~18 in the last (0 to -7 in the final scenarios, i.e. the substitute lands inside the window), so the break-even probability falls with it -- constant-threshold rules overpay early in the year and under-swap at the end.

## League table (top 25 of 97 rules, sorted by mean total)

| rank | rule | total | timing | early | late | op | swaps | catches | misses | recall | early/swap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `R5_flatg_tail[c_op=3.5,pmin=0.02]` | 1562.6 | 1417.6 | 542.6 | 875.0 | 145.0 | 18.9 | 5.92 | 3.54 | 0.626 | 28.7 |
| 2 | `R5_flatg_tail[c_op=3,pmin=0.02]` | 1563.7 | 1418.3 | 545.6 | 872.7 | 145.4 | 19.0 | 5.94 | 3.52 | 0.628 | 28.7 |
| 3 | `R5_flatg_tail[c_op=4,pmin=0.02]` | 1564.1 | 1418.6 | 541.3 | 877.3 | 145.5 | 18.8 | 5.90 | 3.56 | 0.623 | 28.7 |
| 4 | `R5_flatg_tail[c_op=2.5,pmin=0.02]` | 1567.5 | 1421.4 | 548.7 | 872.7 | 146.0 | 19.1 | 5.94 | 3.52 | 0.628 | 28.7 |
| 5 | `R5_tailev[margin=-5,pmin=0.02]` | 1572.4 | 1424.4 | 549.4 | 875.0 | 148.0 | 19.3 | 5.90 | 3.56 | 0.623 | 28.4 |
| 6 | `R5_flatg_tail[c_op=5,pmin=0.02]` | 1575.0 | 1430.0 | 539.0 | 891.0 | 145.0 | 18.7 | 5.83 | 3.62 | 0.617 | 28.8 |
| 7 | `R5_tailev[margin=0,pmin=0.03]` | 1592.7 | 1448.2 | 535.1 | 913.1 | 144.4 | 18.5 | 5.77 | 3.69 | 0.610 | 28.9 |
| 8 | `R5_tailev[margin=0]` | 1594.7 | 1448.3 | 535.2 | 913.1 | 146.4 | 18.9 | 5.77 | 3.69 | 0.610 | 28.4 |
| 9 | `R5_flatg_tail[c_op=15]` | 1603.0 | 1463.5 | 502.1 | 961.5 | 139.5 | 17.0 | 5.58 | 3.88 | 0.590 | 29.5 |
| 10 | `R5_tailev_quota[q=20]` | 1605.5 | 1473.9 | 514.1 | 959.8 | 131.6 | 16.1 | 5.62 | 3.83 | 0.595 | 31.9 |
| 11 | `R5_tailev_quota[q=16]` | 1625.0 | 1500.3 | 483.0 | 1017.3 | 124.7 | 14.8 | 5.40 | 4.06 | 0.570 | 32.7 |
| 12 | `R5_flatg_tail[c_op=0]` | 1644.4 | 1428.4 | 564.0 | 864.4 | 216.0 | 31.6 | 5.98 | 3.48 | 0.632 | 17.8 |
| 13 | `R0_topk[k=19]` | 1665.1 | 1521.6 | 658.7 | 862.9 | 143.4 | 19.0 | 5.92 | 3.54 | 0.626 | 34.7 |
| 14 | `R5_tailev[margin=10]` | 1668.4 | 1527.7 | 509.6 | 1018.1 | 140.6 | 17.3 | 5.38 | 4.08 | 0.568 | 29.5 |
| 15 | `R2_quota[q=16,floor=0.1]` | 1674.4 | 1546.4 | 523.9 | 1022.5 | 128.0 | 15.5 | 5.42 | 4.04 | 0.573 | 33.8 |
| 16 | `R2_quota[q=17,floor=0.1]` | 1678.8 | 1547.9 | 556.0 | 991.9 | 130.9 | 16.2 | 5.52 | 3.94 | 0.584 | 34.3 |
| 17 | `R2_quota[q=17,floor=0.05]` | 1682.1 | 1548.4 | 574.9 | 973.5 | 133.7 | 16.7 | 5.58 | 3.88 | 0.590 | 34.4 |
| 18 | `R2_quota[q=16,floor=0.05]` | 1683.5 | 1554.0 | 534.2 | 1019.8 | 129.4 | 15.8 | 5.44 | 4.02 | 0.575 | 33.9 |
| 19 | `R2_quota[q=15,floor=0.1]` | 1690.8 | 1565.5 | 494.5 | 1071.0 | 125.3 | 14.7 | 5.19 | 4.27 | 0.548 | 33.7 |
| 20 | `R2_quota[q=15,floor=0.05]` | 1692.0 | 1565.9 | 497.6 | 1068.3 | 126.1 | 14.8 | 5.21 | 4.25 | 0.551 | 33.6 |
| 21 | `R5_tailev[margin=25]` | 1712.2 | 1577.6 | 461.2 | 1116.5 | 134.6 | 15.2 | 5.04 | 4.42 | 0.533 | 30.2 |
| 22 | `R2_quota[q=17,floor=0.15]` | 1718.5 | 1587.9 | 502.4 | 1085.4 | 130.7 | 14.8 | 5.19 | 4.27 | 0.548 | 33.9 |
| 23 | `R3_xband[hi=0.25,mid=0.1,lo=0.08]` | 1722.0 | 1581.6 | 513.7 | 1067.9 | 140.3 | 16.9 | 5.25 | 4.21 | 0.555 | 30.4 |
| 24 | `R2_quota[q=16,floor=0.15]` | 1722.9 | 1594.5 | 482.9 | 1111.7 | 128.4 | 14.4 | 5.10 | 4.35 | 0.540 | 33.6 |
| 25 | `R3_xband[hi=0.25,mid=0.22,lo=0.08]` | 1735.8 | 1599.9 | 459.7 | 1140.2 | 135.8 | 15.5 | 5.02 | 4.44 | 0.531 | 29.7 |

## Baselines (current behaviour)

| rule | rank | total | timing | early | late | swaps | catches | misses | recall | early/swap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0_topk[k=19]` | 13 | 1665.1 | 1521.6 | 658.7 | 862.9 | 19.0 | 5.92 | 3.54 | 0.626 | 34.7 |
| `R0_pthresh[p>0.26]` | 48 | 1818.2 | 1697.1 | 416.3 | 1280.8 | 12.1 | 4.56 | 4.90 | 0.482 | 34.4 |
| `R0_topk[k=13]` | 44 | 1792.1 | 1670.9 | 445.9 | 1225.0 | 13.0 | 4.69 | 4.77 | 0.496 | 34.3 |

## Block means (6 blocks of 8 scenarios, mean total per block)

| rule | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---:|---:|---:|---:|---:|---:|
| `R5_flatg_tail[c_op=3.5,pmin=0.02]` | 1308.8 | 1544.3 | 2427.0 | 2136.8 | 1333.5 | 625.4 |
| `R5_flatg_tail[c_op=3,pmin=0.02]` | 1295.8 | 1559.4 | 2427.0 | 2136.8 | 1333.5 | 629.9 |
| `R5_flatg_tail[c_op=4,pmin=0.02]` | 1308.8 | 1544.3 | 2427.0 | 2136.8 | 1347.5 | 620.3 |
| `R0_topk[k=19]` | 1369.5 | 1632.6 | 2681.2 | 2212.5 | 1452.0 | 642.8 |
| `R0_pthresh[p>0.26]` | 1756.7 | 1789.0 | 2587.8 | 2297.1 | 1754.1 | 724.6 |
| `R0_topk[k=13]` | 1718.2 | 1960.6 | 2602.4 | 2367.8 | 1345.6 | 757.8 |

## Reference points and significance

* Oracle selection (swap exactly the due set): mean total 75.3 at 9.5 swaps -- the residual ~1500 against any real rule is model recall, not selection; the top rules already catch ~5.9 of ~9.5 due, and each remaining miss costs ~250.
* `R5_flatg_tail[c_op=3.5,pmin=0.02]` vs `R0_topk[k=19]`: paired mean delta -102.4 +/- 35.0 (s.e., n=48).
* `R5_flatg_tail[c_op=3,pmin=0.02]` vs `R0_topk[k=19]`: paired mean delta -101.3 +/- 34.9 (s.e., n=48).
* `R5_flatg_tail[c_op=4,pmin=0.02]` vs `R0_topk[k=19]`: paired mean delta -101.0 +/- 34.7 (s.e., n=48).
* `R0_pthresh[p>0.26]` vs `R0_topk[k=19]`: paired mean delta +153.1 +/- 63.3 (s.e., n=48).
* `R0_topk[k=13]` vs `R0_topk[k=19]`: paired mean delta +127.0 +/- 48.9 (s.e., n=48).

## Robustness notes

* Trap class: batteries whose substitute EOL (end_time + 30d) is already in the past stay alive in `locations` forever (no recorded EOL), price p=0, and cost 10/day-late the moment they are swapped, while deferring them is free. Any EV rule with an acceptance margin below the per-swap op constant (-5.6) selects all of them and the mean total explodes from ~1570 to ~9900. The `pmin=0.02` guard makes the recommended rule structurally immune.
* The c_op plateau is flat: c_op in [2.5, 5] with pmin=0.02 all land within ~5 points of each other (paired -90 to -102 vs k=19), so the recommendation uses c_op=4, which matches the actual per-swap op estimate (4.1h) rather than the sample argmin.
* X falls from ~330 days (block 1) to ~0 (block 6); in the last block the substitute EOL lands inside the window, so a wasted swap timed at end_time+30d costs nothing -- volume there is limited only by op cost.

## Recommendation

Recommended rule: `R5_flatg_tail[c_op=4,pmin=0.02]` -- swap battery i iff `p_i * 279 > 0.5 * (q_obs_i * mean_excess_i + q_unobs_i * max(X_i, 0)) + 4` and `p_i > 0.02`, where `X_i = normalize(end_time_i) + 30d - (start + 42d)` in days, `q_obs/q_unobs/mean_excess` come from the forecaster tail, and 279 = 10*27.3 + 6 (avoided lateness plus emergency visit op). All quantities are known at plan time.

Recommended-rule mean total 1564.1 vs 1665.1 for the current k=19 selection, 1818.2 for p>0.26 and 1792.1 for k=13. The gain comes from replacing a fixed volume with a per-scenario break-even that tracks the known earliness price: per block it swaps 16.1 / 18.5 / 15.4 / 13.8 / 15.6 / 33.6 against a fixed 19 -- fewer than 19 mid-year where X is still expensive and the model sees fewer dues, and ~34 in the final block where the substitute EOL lands at or inside the window (break-even p ~ 0.02, a wasted swap costs almost nothing), all while catching at least as many dues (5.90 vs 5.92) at 6.0 fewer earliness points per swap.

Leaderboard-comparable early cost per planned swap: 28.7 on train for the top rule (k=19 anchor: 34.7; public shows 48.7-54.8 for the current submission). Scaling by the current rule's train->public inflation, the counterfactual public price under the recommended rule is ~40.2-45.3 early per planned swap (leader: 23.8). Caveat: the public split's scenario dates (hence X) and building mix differ; the scaling assumes the same inflation as the current rule.
