# RED TEAM: Leaderboard Gap Decomposition — Independent Verification

Date: 2026-08-22. Author: red-team pass, everything below recomputed from primary sources only:
`.venv/Lib/site-packages/batteryswap_public/evaluate.py` (+ `utils.py`, `metric.py`),
`dataset/train/scenarios.json`, `dataset/train/devices.csv`, `dataset/train/eol_times.csv`,
and the public leaderboard table (48 public scenarios, per-scenario means).

Leaderboard aggregation verified in `metric.py`: per-scenario overall costs are averaged
(`combined.mean(axis=1)`), and `total = sum of the 9 component columns`. For all 8 teams the
component sum reproduces the printed total to <=0.01. **Confidence: certain.**

Scenario settings (`scenarios.json`, all 48 train scenarios): every numeric setting is
**constant across scenarios** — window 42d, early 0.5/d, late 10/d, unobserved_eol_days 30,
overtime start 8h factor 2, swap 0.25h, room 0.5h, building 1.0h, daily limit 24h/100,
weekly limit 24h/100. Only `base_location` (16 unique), `base_room` (27 unique) and the
travel matrix vary. All 48 starts are Mondays, 7 days apart (2025-09-01 → 2026-07-27).
Travel: diagonal = 0.0333h everywhere; base→building one-way: median ≈ 0.2–1.0h,
mean ≈ 2.07h, max = 10.25h in every scenario. **Confidence: certain (train); public assumed
similar per official docs.**

---

## 1. How each column is computed (from `evaluate.py`)

| Column | Mechanic | Emergencies included? |
|---|---|---|
| `battery_swap` | `time_per_battery_hours` = **0.25 × every executed swap** (`swap_battery`, line 419–423) | **Yes** — emergencies use the same `visit_battery_location` path |
| `building_change` | 1.0h each time the technician enters a different building (`change_building`). Return-to-base at end of day charges **travel only**, no building_change | Yes — 1.0h per emergency (unless battery is in the base building, then no travel/bldg either) |
| `room_change` | 0.5h when the room differs from current (`change_room`). `state.room` persists across days (end-day does not reset it) | Yes, usually 0.5h each |
| `travel` | matrix hours on each building change + return-to-base leg in `end_previous_day`. Days with no actions cost nothing; the final end-day always fires once even for all-defer (diagonal 0.033h) | Yes — 2 × travel(base↔building) per emergency |
| `overtime` | per worked day: `max(total_daily − 8, 0) × 2` where `total_daily` includes the return leg | Yes — emergency day hours = 2·t + 1.75 (+OT if > 8h, i.e. one-way t > 3.125h) |
| `daily_limit` | flat 100 if `total_daily > 24` (strict >) | In principle; needs one-way travel > 11.125h — **never possible on train matrices (max 10.25 → 22.25h day)** |
| `weekly_limit` | flat 100 at each 7-day week transition if accumulated work-hours `>= 24` (inclusive), counter then resets; final partial week force-checked once at the very end | Yes — emergency hours accumulate; a week transition fires before emergencies #2, #9, #16, … |
| `late_swap` | in `swap_battery`: `delta = eol − swap_day`; if `delta <= 0`: `|delta| × 10` | Yes (this is where emergency cost lives) |
| `early_swap` | if `delta > 0`: `delta × 0.5` — **no cap anywhere** | Emergencies are never early (their EOL is in-window, swap is after) |

**Emergency scheduling** (lines 488–496): after the whole calendar (which runs to
`end_day = end_time + (6 − weekday)` = **start + 48d for Monday starts**), every battery with
*observed* EOL ≤ window end that was never swapped gets its own synthetic day, **one per day,
in sorted battery-id order**, starting on day 48, each day = base → building → swap → base:

```python
eol_not_swapped = set(eol_batteries) - set(state.swapped_batteries)
for battery in sorted(eol_not_swapped):  # sorted: deterministic cost
    check_weekly_limit(day=state.day)
    visit_battery_location(battery)
    end_previous_day(new_day=state.day + pandas.Timedelta(days=1))
```

So the k-th missed battery is swapped on day **48 + (k−1)** and hits: late_swap,
battery_swap, building_change, room_change, travel, overtime (if far), weekly_limit
(accumulation), and — only theoretically — daily_limit. **Confidence: certain (read from code).**

## 2. `early_swap` reference and the missing cap

`evaluate_plan` fills unobserved EOLs before scoring (lines 288–292):

```python
eol_unobserved_devices = eol_times.index[eol_times.isna()]
end_times = locations.loc[eol_unobserved_devices]['end_time']
eol_assume = end_times + pandas.Timedelta(days=settings.unobserved_eol_days)
eol_times.loc[eol_unobserved_devices] = eol_assume.dt.normalize()
```

- Reference = **recorded EOL date** when one exists **anywhere in the split data** (even months
  after the window), else **substitute EOL = `locations.end_time` (device's last data timestamp
  in the full split, from devices.csv — NOT truncated at the scenario) + 30 days**.
- A battery that never fails during observation and transmits to dataset end (445/461 train
  devices) has proxy ≈ 2026-08-31. Swapping it mid-window costs `0.5 × (proxy − swap_day)`:
  **343 days-early (≈ 172 pts) in scenario 1, declining ~7 d/week to 14 days (≈ 7 pts) in
  scenario 48**. Pooled train quantiles of days-early for a day-21 swap of a censored battery:
  q25 = 84, q50 = 175, q75 = 259, mean = 170.
- Swapping a battery whose EOL is recorded *after* the window: early runs from swap date to
  that actual EOL (median excess beyond window end = 78d; days-early at day-21 swap:
  median 91, falling from ~122 (s01) to ~22 (s42)).
- **There is no cap.** The only "cap-like" statement anywhere is the 0.5/day rate itself.
- Trap: ~2.5% of censored batteries have proxy *before* mid-window (data stopped >51d before);
  swapping those lands in the `else` branch = **late at 10/day**. Deferring them is free
  (a NaN EOL can never enter `eol_batteries` because `NaN <= end_time` is False).

**Confidence: certain (code); the X magnitudes assume public devices' end_times mirror train
(same collection end ~2026-08-01) — high confidence given "splits otherwise similar".**

## 3. `late_swap` — verified

`10/day × (swap_day − eol)` whenever swap ≥ EOL, including emergency swap dates (day 48+k−1).
Observed EOLs in eol_times.csv are pure dates, plan days are dates → integer day deltas.
Swapping exactly on the EOL date costs 0 (strict `delta > 0` test puts delta=0 in the late
branch with |delta| = 0). **Confidence: certain.**

## 4. Daily / weekly limits — verified with two corrections to folklore

- Daily: flat 100 per day where hours **> 24** (strict), *not* "at 24".
- Weekly: flat 100 per week-bucket where hours **≥ 24** (inclusive), checked only at 7-day
  transitions anchored to the start Monday, plus one forced check at the very end.
- Both count all four time components; **emergency visits count** (their week buckets are day
  {42-48 incl. day-42 planned work and final return leg}, then {49–55}, …).
- With m ≤ 4 misses at mean travel (~5.9h/emergency-day), emergency weeks usually stay < 24h
  → for realistic miss counts the weekly/daily columns are dominated by the **planned** weeks.
**Confidence: certain (code), high (magnitudes).**

## 5. Deferral semantics — verified

`plan = plan[plan['day'] <= end_time]` (line 449) cuts deferred rows;
`eol_batteries = set(eol_times[eol_times <= end_time].index)` (line 284) decides emergencies.

| Battery | Deferred (day > start+42) | Swapped in-window |
|---|---|---|
| EOL recorded in-window | Emergency: late 10×(48+k−1−eol) + full op day | early/late vs recorded EOL |
| EOL recorded after window (but within observation) | **Zero cost — no charge of any kind** | early = 0.5 × (recorded EOL − swap day), even though EOL is post-window |
| EOL never recorded | **Zero cost** | early = 0.5 × (last-data + 30 − swap day); late 10/day if data stopped >30+d before swap |

**Confidence: certain.**

---

## 6. Reverse-engineering the teams (S = battery_swap/0.25; m from the late quadratic)

Late for m misses (Monday starts, first emergency day 48, empirical mean in-window EOL
position Ē = 20.74d, n = 454 train events, ~uniform as expected from weekly sliding windows):

`late(m) = 10 × [ (48 − 20.74)·m + m(m−1)/2 ]` → mean 272.6/miss at m=1, 278.6 at m=2.2, 288.6 at m=4.2.

Self-consistency check: m = 4.20 predicts 1212.1 vs rule-mgt-zero's observed **1212.29**;
m = 2.18 predicts 606 vs J2W's **605.83**. The quadratic queue term is visibly present in the data.

| team | total | S swaps | m misses | P planned | caught (D−m) | wasted (P−C) | precision | early/planned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J2W | 1160.67 | 14.88 | 2.18 | 12.70 | 7.28 | 5.42 | 0.57 | 23.8 |
| YassY | 1370.90 | 17.44 | 2.11 | 15.33 | 7.35 | 7.98 | 0.48 | 32.0 |
| CarlAlbert | 1407.63 | 15.48 | 2.49 | 12.99 | 6.97 | 6.02 | 0.54 | 30.3 |
| sackriti | 1468.23 | 14.48 | 2.86 | 11.62 | 6.60 | 5.02 | 0.57 | 29.5 |
| AttnHeads | 1476.73 | 16.50 | 2.53 | 13.97 | 6.93 | 7.04 | 0.50 | 33.7 |
| astar | 1487.21 | 18.83 | 2.18 | 16.65 | 7.28 | 9.37 | 0.44 | 35.6 |
| rule mgt 0 | 1661.45 | 12.08 | 4.20 | 7.88 | 5.26 | 2.62 | 0.67 | 17.4 |
| **US** | 2078.28 | 21.69 | 2.27 | 19.42 | 7.19 | 12.23 | **0.37** | **48.7** |

(caught/wasted/precision computed at D = 9.46; m assumes all late is emergency late — if teams
carry ~10–30 pts of in-window late, subtract ~0.05–0.1 from m.)

**D (true in-window dues per public scenario):**
- Hard consistency bounds from the leaderboard alone: **D ≥ 4.2** (rule-mgt-zero's m — misses
  can't exceed dues) and **D ≤ 12.1** (their planned 7.88 can't catch more than D − 4.2).
  J2W & US alone only require 2.3 ≤ D ≤ 14.9 — they are consistent with any plausible D.
- Train ground truth: **D = 9.46 mean (median 9, SD 3.98 across scenarios, SEM 0.57,
  range 2–19)**, strongly declining over the year (first 24 scenarios 12.3, last 24 6.6,
  corr(D, index) = −0.80).
- Verdict: **D ≈ 9.5, credible band 8–11** for the public split.

Headline: **US and J2W have the same recall (7.2 vs 7.3 dues caught; 2.27 vs 2.18 misses).
US plans 6.7 more swaps per scenario and gets nothing for them.** Wasted-swap early cost:
J2W ≈ (302.7 − ~15 caught-margin)/5.4 ≈ 53 pts (≈107 days early each); US ≈ 931/12.2 ≈ 76 pts
(≈153 days each) — US both wastes more swaps and wastes them on worse (longer-tail / earlier-
in-calendar) batteries. **Confidence: high for S, m (mechanical); medium for the D-dependent
split into caught/wasted (quoted at D = 9.46).**

## 7. Gap decomposition, US (2078.28) − J2W (1160.67) = 917.61

| bucket | US | J2W | Δ | share |
|---|---:|---:|---:|---:|
| early_swap | 945.68 | 302.72 | **+642.96** | **70.1%** |
| capacity = overtime + daily + weekly | 409.64 | 193.27 | **+216.37** | **23.6%** |
|   · overtime | 120.05 | 68.27 | +51.79 | 5.6% |
|   · daily_limit | 164.58 | 85.42 | +79.17 | 8.6% |
|   · weekly_limit | 125.00 | 39.58 | +85.42 | 9.3% |
| late_swap | 633.96 | 605.83 | **+28.12** | **3.1%** |
| operations = swap+bldg+room+travel | 89.01 | 58.85 | **+30.15** | **3.3%** |
|   · travel | 59.01 | 38.30 | +20.71 | |
|   · building/room/swap time | 30.00 | 20.55 | +9.44 | |
| **total** | 2078.28 | 1160.67 | **+917.60** | 100% |

Ops per swap is nearly identical (US 4.10h vs J2W 3.96h/swap) → the +30 ops is pure volume.
**Confidence: certain (arithmetic on published columns).**

## 8. Break-even due-probability p* for one extra swap (actual mechanics)

Swap a marginal battery (bundled into an existing route, marginal op c_pl ≈ 1.0h; safety
margin Δ ≈ 4 days if it is due) vs defer:

- E[cost | swap] = c_pl + p·0.5Δ + (1−p)·0.5·X, X = E[days early | not due]
- E[cost | defer] = p·(10·L_em + c_em); L_em ≈ 27.3 + (queue position −1) ≈ 27.3–29.5;
  c_em = 2·t̄ + 1.75 + OT ≈ 6h at mean base travel 2.07h (weekly/daily hits negligible for m ≤ 4)

`p* = (c_pl + 0.5X) / (10·L_em + c_em + 0.5X − 0.5Δ)`  e.g. X=170: (1+85)/(273+6+85−2) = 86/362 = 0.238

| X (days early if not due) | 14 (s48 censored) | 30 | 60 | 91 (post-window median) | 120 | 150 | 170 (censored mean) | 190 | 240 | 343 (s01 censored) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p* (L_em=27.3) | 0.028 | 0.055 | 0.101 | 0.144 | 0.181 | 0.216 | 0.238 | 0.258 | 0.305 | 0.385 |

Sensitivity: to L_em (i.e. to D / queue position) — weak: L_em 27.3→29.5 moves p*(X=170) only
0.238→0.224. To c_em (near vs 10.25h-far building, 6→65h-equiv) — moves p*(X=170) 0.238→0.207.
**The dominant variable is X**, which is set by the battery's tail type and the scenario's
calendar position (censored X falls linearly 343 → 14 across the 48 scenarios).

Corrections to the team's single number "p* ≈ 0.265": it is **only the top of the band** —
correct for a censored-forever battery in a first-half scenario (X ≈ 190). For batteries whose
plausible not-due outcome is "fails shortly after the window" (the typical marginal candidate),
**p* ≈ 0.10–0.15**, and in the last ~10 scenarios wasted swaps are almost free (p* ≈ 0.03–0.10).
A single global threshold is wrong; it must be per-battery-tail and per-scenario-date.
**Confidence: high (formula exact from code; X table exact on train).**

## 9. "Capacity is a symptom of volume" — REJECTED as stated

Fit of capacity (overtime+daily+weekly) on total swaps S across the 8 teams:
- all 8: cap = 13.8·S + 27.9, **R² = 0.38** — driven entirely by the US point;
- excluding US: cap = **−3.4·S + 286.0, R² = 0.13** (slightly *negative* slope);
- corr(cap, S) = +0.62 all, **−0.35 excluding US**; corr(cap, misses) ≈ 0.00.

The 7 non-US teams sit in a flat band 193–252 (mean 232, SD 21) while spanning 12.1–18.8
swaps/scenario — astar runs 18.8 swaps at 225.6 capacity. US's 409.6 is a plan-shape outlier
(1.65 daily hits, 1.25 weekly hits, 60 excess overtime-hours per scenario), not an unavoidable
volume cost. Volume does add hours (~4.1h ops/swap against a 24h/week penalty threshold), so
capacity repair should accompany, not replace, the volume cut — but ~85–100% of the +216 is
recoverable at *unchanged* volume on the cross-team evidence. **Confidence: high.**

## 10. What the winners do differently — levers ranked (points/scenario for US)

| # | lever | evidence | points |
|---|---|---|---:|
| 1 | **Precision of swap selection** (cut ~6.7 lowest-p planned swaps; per-tail/per-date p* threshold; never swap long-censored batteries in early-calendar scenarios) | early/planned: J2W 23.8 vs US 48.7; identical recall | **~643** |
| 2 | **Plan shape / capacity repair** (split >24h days; keep week buckets <24h replaying the evaluator's Monday-anchored buckets incl. the day-42+emergency bucket; trim >8h days) | top-7 capacity flat in volume; US 409.6 vs band 193–252 | **~216** |
| 3 | Ops (comes ~free with lever 1: 6.7 swaps × 4.1h) | ops/swap already par | ~30 |
| 4 | Late / recall | US misses 2.27 vs best 2.11 — already competitive; each extra catch is worth ~280 but only if its p ≥ p* | ~28 (vs J2W); ≤ ~590 pool vs perfect for anyone |

To *match* J2W: levers 1+2 are 94% of the gap. To *beat* J2W afterwards: the only big pool
left is the ~606 late that everyone pays (2.1–2.2 misses × ~279): +1 caught due battery per
scenario at maintained precision ≈ −270; that is a Task-1 calibration problem, not a planner one.

## Where the evaluator contradicts prior team assumptions

1. **No early cap exists**; the reference is the recorded EOL *anywhere in the data* (even
   post-window), else last-data + 30d — with early costs up to ~172 pts/swap in early scenarios.
2. **Emergency lateness is not a constant 27**: it is `48 − 20.74 + (k−1)` per queue position
   (272.6 → 288.6 pts/miss as m goes 1 → 4.2); confirmed to 0.2 pts on rule-mgt-zero.
3. **Daily-limit hits cannot come from emergencies** on train-like travel (max day 22.25h < 24);
   US's 164.6 daily points are self-inflicted >24h *planned* days.
4. **Deferring a battery whose EOL falls after the window (or is never observed) is entirely
   free** — there is no post-window charge of any kind; and ~2.5% of censored batteries are
   late-traps if swapped (proxy already passed).
5. **Weekly buckets are Monday-anchored fixed bins with an inclusive ≥ 24 test**, checked
   before the day's work, and the day-42 work shares a bucket with the final return leg and
   emergency #1.
