# Error audit at the operating point (censored model, lm 1.8, cap 15)

Written 2026-08-22 by the error auditor. Sources: one instrumented planner run
(`tools/audit_operating_point.py` -> `outputs/audit_ledger.csv`, 2,544 per-battery
rows + `outputs/audit_operating_point.json`), the recorded A/B pair
`val_ship_final.json` (lm 1.0) vs `val_latemult18.json` (lm 1.8), and exact
per-battery early/late from the evaluator's own transition log. Model verified as
`outputs/v8_folds_cens.joblib` (expected_due matches 7.64/10.07/13.98 on s_0..s_2).

Operating-point replicates (CP-SAT 1 s time-limit is the only nondeterminism):

| run | total | late | early | served | missed |
|---|---:|---:|---:|---:|---:|
| val_latemult18 | 1924.5 | 1034.4 | 650.7 | 14.19 | 4.12 |
| val_lm18_rerun | 1939.1 | 1049.2 | 654.0 | 14.25 | 4.15 |
| val_lm18_cappass | 1972.5 | 1078.5 | 657.1 | 14.21 | 4.25 |
| this audit | 1980.6 | 1077.5 | 667.9 | 14.23 | 4.23 |

Run-to-run band ~55 pts on the mean, ~45 on late. Class numbers below are from
the audit replicate; read them with that band, and classes under 0.5/scen as
indicative. The slot limit is min(15, ceil(1.6*sum(p)+1)): the flat 15 binds in 33/48
scenarios, the due-budget binds below 15 in the other 15 (limits 11-14, s_26-s_39
region); planned count sits at the limit in 47/48 scenarios.

## 1. Why late-multiplier 1.8 works under the cap

A/B on the recorded pair (lm 1.8 minus lm 1.0, per-scenario means, same model,
same cap): **total -132.0 = late -118.8, early -7.2, capacity -5.7, ops -0.3.**

Channel decomposition of the -119 late:

| channel | points/scen | evidence |
|---|---:|---|
| (a) acceptance/selection: misses converted to hits | **-76 to -109** | net hits +0.31/scen x 246/miss (exact avg emergency late) = -76; the whole late delta inside hit-changed scenarios = -5220/48 = -109. In the 14 hit-up scenarios hits rose +17 on served +6: >60% of the conversions are **substitutions** (same slot count, better occupant), not extra volume. |
| (b) timing: same batteries, earlier days | **-10 to -42** | hit-unchanged scenarios (n=32) move late by only -480 total = -10/scen; the analytic best-service-day (identical forecast, both multipliers) is 0.72 d earlier at lm 1.8 (opening block +1.5-2.7 d, mid ~+0.3 d). Residual timing loss at 1.8 is small anyway (planned-late = 36.5/scen, below). |
| (c) calmer plans / capacity | **-5.7** | almost all weekly_limit (64.6 -> 58.3); the weekly bucket accrues emergency days, so most of this is a knock-on of (a), not calmer planned days (daily_limit unchanged at 52.1). |
| early side | **-7.2** | flat despite +0.21 served: the marginal acceptances are likelier-due. |

Where in the machine the knob acts (from the captured cost tables at both
multipliers): the **standalone economics ordering barely changes** — the greedy
top-limit sets share 13.79 of ~14.1 members (churn 0.46 in / 0.23 out per
scenario), and their due content moves only +0.06/scen (5.19 -> 5.25). What
changes is **adherence of the joint search to its own economics**: at lm 1.0 the
planner realizes 5.02 hits against 5.19 in-set dues (the local search trades
marginal dues away for operational cost — the bundling/defer margin is thin at
1.0x), at lm 1.8 it realizes 5.33 against 5.25 (full adherence, plus the 0.7 d
day-pull converting would-be TIMING losses). The multiplier is an
acceptance-enforcement knob, not a ranking knob.

Why it needed the cap (and why the V6-era uncapped sweep read as noise): uncapped,
a late tilt manufactures volume at marginal precision ~0.2 — early buys back the
late (wash). Capped, volume cannot grow, so the tilt can only change *who* holds
the 15 slots and *when* they are served. The contrast is measured: lm 1.0 -> 1.8
adds +0.21 served and +0.31 hits (substitution included, hits grow 1.5x faster
than swaps), while cap 15 -> 17 at lm 1.8 adds +1.23 served for +0.29 hits
(marginal volume precision 0.24) and +70 early — net worse. lm 2.2 overshoots:
day-pull too far and churn quality drops (late +25, early +14 vs 1.8).

## 2. The remaining ledger (audit replicate: late 1077.5, early 667.9 per scenario)

### 2a. Late side — 9.46 dues/scen: 4.79 hit on time, 0.44 hit late, 4.23 missed

| class | n/scen | late pts/scen | avg late | med p | med margin | med beta30 | med days-to-EOL |
|---|---:|---:|---:|---:|---:|---:|---:|
| INVISIBLE (p<0.02, unswapped) | 1.96 | **487.9** | 249 | 0.0002 (p90 0.014) | 0.179 V | 0.0105 | 27 |
| VISIBLE-BUT-OUTRANKED (gain>0, rank>limit) | 1.10 | **269.6** | 244 | 0.18 (0.07-0.49) | 0.078 V | 0.013 | 29 |
| VISIBLE-BUT-UNECONOMIC (gain<=0 at its p) | 0.98 | **226.5** | 231 | 0.048 | 0.096 V | 0.013 | 28 |
| VISIBLE-DROPPED (rank<=limit, search deferred it) | 0.19 | **57.1** | 304 | 0.334 | 0.053 V | 0.014 | 17 |
| TIMING (swapped in-plan but after EOL) | 0.44 | **36.5** | 83 (8.3 d) | 0.54 | 0.053 V | 0.013 | 7 |

Would a swap have been accepted at any p? Exclusion layer of the 203 misses:
121 sat **outside the candidate filter** entirely (med p 0.0015; 29,660 late pts —
the invisibles), 20 in-filter with negative gain (4,630), 53 positive-gain but
**rank beyond the slot limit** (12,940), 9 in the money and dropped by the search
(2,740). So for ~60% of missed-due points no decision layer ever saw a choice;
for ~30% the cap was the binding edge; ~3% is search slack.

Class notes:
* **INVISIBLE** is knee-entry, not near-threshold: median margin 0.179 V failing
  27 days later, beta30 at 0.0105 — *below* the 0.013 of caught dues. Neither
  level, slope, nor the IR channel flags them at cutoff. 97% never enter the
  candidate set. This is the mid-block information wall (foresight ladder: even a
  21-day peek only reaches 0.151 on these).
* **OUTRANKED** misses are shallow: median 3 ranks (p75 5) beyond the limit, and
  the class is cap-bound in every case. What outranks them is the certainty
  inversion below.
* **TIMING** is mostly the forecast, not the router: in 62% of cases the cost
  table's own best day was already past EOL; the planner's median slip past its
  own best day is 0.0 d.

### 2b. Early side — 667.9/scen across 13.6 early-charged swaps

| class | n/scen | early pts/scen | avg | avg days early | med p |
|---|---:|---:|---:|---:|---:|
| (ii) NEVER-DUE wasted (no EOL on record) | 6.77 | **519.9** | 77 | 154 | 0.563 |
| (iii) POST-WINDOW-DUE (EOL after window) | 2.23 | **114.0** | 51 | 102 | 0.367 |
| (i) genuinely-due, swapped early | 4.63 | **34.1** | 7.4 | 14.7 | 0.542 |

* **NEVER-DUE is 78% of all early and it is concentrated**: 29 distinct devices,
  and the five documented floor-zombies (d_b5b678a3f79f 48/48 scenarios,
  d_3d26e12378f1 40, d_c9a2ce794b68 39, d_d9d695df1683 33, d_d4b4272d5229 32;
  med p 0.48-0.93 at margins 0.002-0.05 V) occupy **4.0 of the 15 slots every
  scenario and cost 339.8 pts/scen**; the remaining 24 devices cost 180.1.
* **POST-WINDOW-DUE is mostly the same inversion, not near-miss insurance**:
  median EOL lands 51 days past the window; only 28% (~32 pts) are within 21 days
  and defensible as timing spread.
* **DUE-EARLY (34.1) is the irreducible-looking part**: 14.7 days early on
  average against a forecast whose crossing-date sigma is +-7-13 d.

### 2c. The coupling that defines the ordering prize

The slots the outranked dues need are held by never-dues at *higher* model
confidence (med p 0.563, zombies 0.83-0.93) than the dues themselves (med p
0.18). Demoting the four zombie slots per scenario admits ranks 16-19 — the
median outranked due sits 3 ranks out. One caution from the record and the
arithmetic: the top-5 zombies carry sum(p) about 3.0 per scenario, so any p-LEVEL
knockdown also shrinks the due-budget ceil(1.6*sum(p)+1) by ~5 slots and strangles
volume — likely why dwell/knee knockdowns kept measuring worse through the
planner. Demote in the ORDER (or exempt the budget's sum(p) from the demotion);
do not shrink the p mass that sets the cap.

## 3. Binding constraints, named and priced (points/scenario)

| constraint | size | what moves it |
|---|---:|---|
| INVISIBLE misses | **488** | Nothing in ranking/planning. Needs a forecast that sees knee-entry from 0.18 V / 27 d out (rise-ratio invariant retrain, raw-daily channel). Information-limited: treat most as unreachable this generation. |
| VISIBLE-BUT-UNECONOMIC | **227** | p-level honesty on real dues predicted 0.02-0.1 (break-even ~0.1-0.33). A level fix, not an ordering fix; the planner correctly refuses at today's p. |
| Ordering prize: OUTRANKED late 270 + zombie NEVER-DUE early 340 + far-out POST-WINDOW ~78 | **~690 gross** | One top-15 reordering: demote the 5 floor-zombies (persistent low-margin non-crossers), promote the p 0.07-0.49 dues sitting 1-5 ranks out. This is the single largest planner-reachable pool; realized capture is bounded by the promoted candidates' ex-ante rate (~0.2-0.3 mid-block), so expect **250-400 net** if the demotion does not leak into the due-budget. |
| Search slack (VISIBLE-DROPPED) | **57** | More search budget / firmer defer anchoring; small and partly noise (n=9). |
| Timing spread (TIMING 36.5 + DUE-EARLY 34.1) | **~71** | Irreducible without sharper timing: 62% of TIMING is the forecast's own best day past EOL; the router already sits on its table's optimum (median slip 0 d). |
| Rational insurance (near POST-WINDOW ~32, non-zombie NEVER-DUE tail ~180) | **~212** | The price of uncertainty at 14.2 swaps/scen; only precision or volume reduction moves it, and cap 17 / cap 13 sweeps both measured worse. |

Sanity: 488 + 227 + 690 + 57 + 71 + 212 = 1745 ~= late 1078 + early 668 (audit
replicate). The researcher's target ordering must move the 690-pool; a candidate
method that reshapes probability LEVELS without changing the top-15 order or that
shrinks sum(p) (and with it the budget) has measured dead five times and will again.
