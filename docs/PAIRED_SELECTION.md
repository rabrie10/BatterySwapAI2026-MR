# Paired-incumbent selection A/B (exact deltas, no reroll noise)

_Stability engineer, 2026-08-22T17:34:29. One incumbent per scenario at the operating point (lm1.8, search 240/240, robust 0, budget 1.6x+1 cap 15, capacity pass on; incumbents reproduce the audit anchors, e.g. s_0 1350.8 vs audit 1350.2). Every arm is a minimal selection diff on that same plan, scored with the official `evaluate_plan` against true `eol_times`, so each per-scenario delta is exact: the +-52 CP-SAT reroll term and the ~100 scenario-overlap floor cancel by construction. Incumbent mean 1935.29 over 48 scenarios; runtime 18.6 min._

Cache: `outputs/paired_incumbents.joblib` (plans + cost tables + forecast p; replays rerun in ~2 min with `--reuse`). Rows: `arm_a_rows` / `arm_b_rows` in the JSON.

## Arm results (delta vs incumbent, negative = better)

| arm | mean d/scen (all) | SE | active scen | mean d (active) | win/loss/tie | sign p | late / early / ops per scen |
|---|---:|---:|---:|---:|---|---:|---|
| A gate forced-include (union) | **-47.56** | 37.5 | 31 | -73.65 | 17/14/17 | 0.7201 | -3.12 / -43.74 / -0.7 |
| A dark-decay only | **-52.42** | 33.79 | 24 | -104.85 | 16/8/24 | 0.1516 | -24.58 / -31.46 / 3.62 |
| A raw-dip only | **1.91** | 25.03 | 16 | 5.74 | 9/7/32 | 0.8036 | 22.71 / -9.69 / -11.11 |
| B1 zombie defer + top-p replacement | **83.43** | 47.8 | 45 | 88.99 | 26/19/3 | 0.3713 | 83.54 / -14.41 / 14.3 |
| B2 zombie defer, no replacement | **66.19** | 39.25 | 45 | 70.6 | 26/19/3 | 0.3713 | 181.67 / -115.06 / -0.42 |
| A+B (no refill) | **-51.52** | 46.84 | 46 | -53.76 | 31/15/2 | 0.0259 | 64.17 / -119.58 / 3.89 |
| A+B refilled to limit | **-79.24** | 55.83 | 46 | -82.69 | 33/13/2 | 0.0045 | -19.58 / -74.32 / 14.66 |

Block means (6 non-overlapping blocks of 8, the honest effective-sample unit):

- A gate forced-include (union): [-19.5, 19.5, -94.2, -179.4, -54.0, 42.2]
- A dark-decay only: [-19.5, -39.8, -9.0, -258.7, -36.1, 48.6]
- A raw-dip only: [0.0, 59.3, 22.0, -86.9, 23.9, -6.8]
- B1 zombie defer + top-p replacement: [-180.7, -26.7, 227.6, 78.2, 172.9, 229.3]
- B2 zombie defer, no replacement: [-103.9, -70.7, 241.5, 12.9, 148.5, 168.9]
- A+B (no refill): [-123.4, -119.1, 46.5, -258.2, -61.1, 206.2]
- A+B refilled to limit: [-200.1, -143.1, -66.6, -258.2, -49.7, 242.4]

## Regime split (plan-time-known: closing block s_40-47 has the whole fleet ending in-window)

| arm | all 48 | s_0-39 | s_40-47 |
|---|---:|---:|---:|
| A gate forced-include (union) | -47.6 | -65.5 | +42.2 |
| A dark-decay only | -52.4 | -72.6 | +48.6 |
| A raw-dip only | +1.9 | +3.7 | -6.8 |
| B1 zombie defer + top-p replacement | +83.4 | +54.3 | +229.3 |
| B2 zombie defer, no replacement | +66.2 | +45.7 | +168.9 |
| A+B (no refill) | -51.5 | -103.1 | +206.2 |
| A+B refilled to limit | -79.2 | -143.6 | +242.4 |

Every arm reverses sign in the closing block: with the whole fleet's unobserved-EOL proxy (end_time+30) inside the window, every slot carries in-window value, so any deferral -- a zombie demotion or the removal half of a forced-include -- buys (proxy) lateness. Gating the mechanism on X (known at plan time) keeps s_0-39 value and zeroes the closing block: A+B refilled at -143.6/scen over s_0-39 = **-119.7/scen on the 48-mean**.

## ARM A mechanics: the substitution question answered

Per swap-in (add gate battery, remove lowest-p planned when at the cap), n=61, mean -39.3:

| gate due | removed due | n | mean d | late d | early d |
|---|---|---:|---:|---:|---:|
| False | False | 20 | -40.3 | +0.0 | -44.2 |
| False | True | 10 | +356.0 | +355.0 | +18.4 |
| True | False | 19 | -298.6 | -230.5 | -69.1 |
| True | True | 12 | +43.7 | +69.2 | -6.5 |

- The removed lowest-p planned battery was DUE in 36% of swap-ins (audit ledger predicted 0.354): those rows cost +44 (both due, 1:1 substitution) to +356 (dropped a real catch for a gate FP). Rows whose removal victim was NOT due won -40 to -299. Late-side substitution is real; the arm's net value comes from the early channel.
- Injection day: building_visit: n=36, mean -92.7, cost_optimal: n=25, mean +37.6 -- adding on an existing building visit is where the value is; opening a new day for the battery loses.

## ARM B mechanics: one true zombie, many swept dues

- 95 planned slot_demote flags across 15 distinct batteries.
- Documented floor-zombie flags (only d_b5b678a3f79f fires the fingerprint): n=45, due rate 0.00, defer-only mean **-97.4/flag** -- a clean early-cost refund.
- All other flags: n=50, due rate 0.48; deferring a due one costs +363 on average. The fingerprint sweeps real dues, exactly as the demotion-only cross-run read feared -- and no measured axis (p, margin, dwell, raw_min3, beta30) separates the floor battery from the swept dues. What does separate them is PERSISTENCE: d_b5b678a3f79f has 45 flags and 0 deaths; swept batteries die within ~3-7 flags of first firing (one reached 9).
- The other four documented floor-zombies (d_3d26e12378f1, d_c9a2ce794b68, d_d4b4272d5229, d_d9d695df1683) hold 33-39 planned slots each in these incumbents at p 0.36-0.74 but NEVER fire the fingerprint (margins 0.04-0.09, dwell 11-23): the shipped rule misses ~4 of the 5 documented never-due slot-holders (~250-340 early pts/scen untouched).
- Evidence thinness: the swept-due harm is ~24 flag-rows from ~5-6 distinct battery deaths (42-day windows overlap 6x); the floor benefit is 45 rows from ONE battery. Any zombie rule generalizes from a handful of device lives.

## Verdicts at paired resolution

- **Resurrection gate (A, union): REAL GAIN** (mean -47.6/scen, SE 37.5, wins/losses/ties 17/14/17, sign-test p 0.7201) Dark-decay carries it (-52.4, 16W/8L); raw-dip is dead (+1.9). Point estimate ~2.5x the old 'dead' read's noise floor, but sign-mixed: the value is mid-block and early-channel, not the late-channel rescue story.
- **Zombie demotion w/ replacement (B1): REAL HARM** (mean +83.4/scen, SE 47.8, wins/losses/ties 26/19/3, sign-test p 0.3713)
- **Zombie demotion, defer only (B2): REAL HARM** (mean +66.2/scen, SE 39.25, wins/losses/ties 26/19/3, sign-test p 0.3713) The old cross-run verdict 'harmful +60-100' is CONFIRMED as stated (paired +66), but for a decomposable reason: -97/flag on the one true floor-zombie vs +363/flag on swept dues.
- **Combined A+B (no refill): REAL GAIN** (mean -51.5/scen, SE 46.84, wins/losses/ties 31/15/2, sign-test p 0.0259)
- **Combined A+B refilled: REAL GAIN** (mean -79.2/scen, SE 55.83, wins/losses/ties 33/13/2, sign-test p 0.0045) Strongest arm; with a plan-time X gate switching it off in the closing block it projects to ~-120/scen.

**The substitution-saturation law's fate:** half right, half measurement artifact. TRUE on the late channel: under the binding cap, gate catches displace planned catches nearly 1:1 (ARM A late component -3.1/scen net), and 36% of forced-include removals hit a real due. FALSE as a value claim: the same substitution recovers wasted-early slots, so the gate is worth -48/scen (dark-only -52) rather than ~0, and pairing it with zombie exclusion + refill compounds to -79/scen (sign-test p 0.004) -- selection-layer edits CAN move the frontier when the displaced slot is a never-due. The old instrument could not have seen any of this: every one of these means sits inside its +-52 reroll / ~100 overlap band.

## Arm C (emergency-rank fix, analytic only)

- defer_cost rebuilt with expected_rank x 0.4807 (realized 2.24 / predicted 4.66); recompute check vs shipped tables: max err 0.0.
- standalone swap->defer sign flips: 0.62/scen; dues among them: 4 total.
- greedy slot-boundary churn: 0.33/scen; dues entering 6, dues leaving 2.
- phantom defer cost removed: 16.75/due battery, 27.95/slot-holding battery.

## Composition

ARM A swap-ins: 61 total, per-swap-in mean -39.3 (negative share 0.639).

| battery | gate | n scen | mean d | due scen | mean late d | mean early d |
|---|---|---:|---:|---:|---:|---:|
| d_4643496a7525 | dark | 1 | -361.8 | 1 | -270.0 | -91.0 |
| d_be0fde6cb60d | dark | 1 | -314.8 | 1 | -230.0 | -84.0 |
| d_5a32d5bc2799 | dark | 4 | -298.1 | 4 | -247.5 | -82.8 |
| d_bd045b2e32a2 | dark | 5 | -166.6 | 5 | -136.0 | -30.1 |
| d_eb7b60348254 | dark | 1 | -155.9 | 1 | -140.0 | -12.5 |
| d_124f1e85339f | dark | 11 | -80.1 | 6 | -47.3 | -41.5 |
| d_8ba95cfeaaa6 | dip | 4 | -78.6 | 4 | -35.0 | -70.1 |
| d_102b0defe0cc | dip | 2 | -65.4 | 1 | 120.0 | -36.5 |
| d_d622ad296e2c | dip | 1 | -50.5 | 0 | 0.0 | -1.5 |
| d_e13ae085ad7d | dip | 2 | -25.5 | 0 | 0.0 | 2.0 |
| d_6b475a7a7d3d | dip | 1 | -1.5 | 0 | 0.0 | 0.0 |
| d_87604c88656b | dip | 4 | 14.6 | 1 | 67.5 | -28.5 |
| d_497cf69db25f | dip | 1 | 32.6 | 1 | 50.0 | -18.5 |
| d_db6b21a2bcd5 | dark | 15 | 49.6 | 6 | 80.0 | -34.2 |
| d_f11c15757bf1 | dip | 4 | 118.7 | 0 | 115.0 | -1.8 |
| d_259ad0f929c8 | dark | 3 | 129.6 | 0 | 120.0 | 9.3 |
| d_7fb0c50f956d | dip | 1 | 191.1 | 0 | 390.0 | 10.5 |

ARM B zombies (per-battery, individually deferred):

| battery | n scen | due scen | mean d defer-only | mean d with replacement |
|---|---:|---:|---:|---:|
| d_1420f5ff758c | 2 | 0 | -117.5 | -46.8 |
| d_b5b678a3f79f | 45 | 0 | -97.4 | -126.3 |
| d_a431a5e84fcd | 1 | 0 | -66.2 | -48.5 |
| d_4406ae97dc09 | 1 | 0 | -52.2 | -25.5 |
| d_2af7891ce12d | 7 | 0 | -28.4 | -26.7 |
| d_9ae8a0552434 | 2 | 0 | -28.2 | -2.0 |
| d_70d09dad9888 | 4 | 0 | -25.9 | -112.4 |
| d_7d80d3aa458c | 1 | 0 | -17.8 | 0.0 |
| d_cfdd8b69ddd4 | 9 | 3 | 61.2 | 24.9 |
| d_ccd3a65228a9 | 7 | 6 | 265.8 | 165.9 |
| d_4643496a7525 | 6 | 5 | 281.8 | 210.7 |
| d_7a18144dbc8c | 3 | 3 | 368.3 | 247.3 |
| d_3b189e96404c | 3 | 3 | 422.3 | 296.0 |
| d_7aa40cb4e3a2 | 3 | 3 | 424.5 | 377.9 |
| d_7139cdf34e3f | 1 | 1 | 505.9 | 236.7 |

## Per-scenario paired deltas, A+B refilled (top arm)

-156 +0 +0 -281 -53 -469 -153 -489 -101 -105 -103 -384 -102 -174 -131 -45 -511 -792 +137 +490 -186 -102 +498 -68 +108 +647 -1001 -224 -548 -515 -65 -467 -531 +125 +206 +371 -570 +6 -2 -3 -4 +569 +337 -274 +1211 -2 +104 -1

## Notes

- Incumbent QA: per-scenario totals track the audit replicate at corr 0.985, mean |diff| 72.5 (CP-SAT reroll scale -- the noise this harness removes within-run); incumbent mean 1935.3 vs audit 1980.6, inside the recorded op band 1924.5-1980.6.
- Incumbents replicate the AUDITED operating point: per-scenario slot limit min(15, ceil(1.6*sum(p)+1)) on the filtered candidate set (audit_ledger: planned==limit 47/48). `CompetitionPlanner.plan()` at HEAD mutates `self.config` when it computes the full-fleet budget, so a reused planner instance freezes the limit at the first scenario's value -- this harness calls the internals per scenario instead (worth a look before the next submission build).
- Gate membership comes from `outputs/research_rowfeat.parquet` columns (dark: staleness>30 & margin-0.001*staleness<0.02; dip: raw_min3<2.40 & margin>0.03 & p_cal<0.1), keyed by (scenario, battery).
- Zombies are the forecaster's own `slot_demote` fingerprint (margin<0.05 & dwell>42d & p>0.4, dwell from the smoothed 2.45 crossing).
- Removals drop the plan row verbatim (triangle inequality verified: skipping a stop never lengthens a leg); additions take the exact-replay cheapest insertion position on the target day.




## Integration fidelity (_selection_exchange, paired replay)

_Generated by tools/paired_selection_fidelity.py: the integrated `CompetitionPlanner._selection_exchange` replayed on the SAME cached incumbents and scored with the official evaluator -- every gap below is a rule difference, not noise. JSON: `outputs/paired_fidelity.json`._

| variant | mean d/scen (48) |
|---|---:|
| integrated `_selection_exchange` as written | **-47.5** |
| same membership+days, order-preserved application | -52.9 |
| incumbent merely re-sorted by (day, battery) | +77.8 |
| measured arm (A+B refilled, X-gated s_40-47) | -119.6 |

**Verdict (v2 exchange, re-replayed after the fixes): FAITHFUL in machinery.** As-integrated now matches the order-preserved application of its own decisions within insertion minutiae (end-of-day-group vs cheapest-slot, ~5/scen; day mismatches 0). The residual gap to the measured -119.6 is FLAG content and gate set, not machinery: clamped staleness (item 2 below), dead dip (3), the p>0.05 refill floor and full-fleet cap basis, and the X<50 skip set -- the A2 section below closes most of it in the flags. For the record, the v1 integration measured +16.6/scen; items 1 and 4-5 are FIXED in v2:

1. **Route destruction (v1 bug, FIXED in v2).** v1 returned `sort_values(["day", "battery"])`, re-ordering EVERY worked day's route alphabetically (the incumbent order is the local search's routed order; the evaluator prices row order). Re-sorting the incumbent alone costs +77.8/scen; mean 0.75 untouched-membership days per scenario come back re-ordered. Fix: keep incoming row order, insert additions into the day's sequence (cheapest-insertion), drop removals in place.
2. **Dark-gate staleness is clamped to the grid end (flag-source bug).** `forecaster.predict` (and the summaries the exchange reads) computes staleness after `index = min(index, len(series)-1)`: a device whose smoothed grid ENDED months before the cutoff (the cold-room dark channel -- the gate's entire target) reads staleness ~0 and never fires. Verified: d_124f1e85339f at s_23 has grid overhang 95 d, runtime staleness 0, true staleness 95; runtime-vs-measured dark sets disagree on 36 of 60 union rows, and d_124f1e85339f (the measured arm's top earner, 11 injections at -80 each) is missed in every scenario. Fix: staleness += max(origin_index - (len(series)-1), 0) before the clamp.
3. **The dip gate is dead at the operating config.** `row_raw_min3` is NaN unless a raw feature variant or a resurrection gate is active (`raw_cache.update` is gated), so `isfinite(raw3)` never passes. Measured dip-only was +1.9 (neutral) -- either wire `raw_cache.update` when `selection_exchange` is on, or delete the dip branch deliberately.
4. **No displacement removals.** The integrated gate loop `break`s at the cap; the measured arm removed extra lowest-p planned batteries to make room (21 extra removals over the 32 both-active scenarios).
5. **Refill placement + pool.** Integrated refills go to the cost-optimal day from the FULL fleet with a p>0.05 floor; the measured arm placed nearest-visit-first from the candidate set (measured placement split: visit -92.7 vs cost-optimal +37.6 per swap-in); 20 common additions land on different days (gate anchors are also computed post-zombie-removal instead of on the incumbent).
6. **Cap basis.** `config.optimizer.max_planned_count` at the exchange call is the mutated FULL-fleet budget; the measured limit was the filtered-candidate budget (+-1 slot in budget-bound scenarios).
7. **X-gate set.** median X<100 skips s_32-47 (16 scenarios) vs the measured projection's s_40-47 (8). The extra 8 skips forfeit only -8.3/scen net (s_33-35 losses roughly offset s_32/s_36 wins) -- defensible, but it is not the measured gate.

**Corrected expectation for the integrated pass:** as written -47.5/scen (ships a regression); with the re-sort fixed (order-preserving application of the same decisions) -52.9/scen; recovering the measured -120/scen additionally requires the staleness overhang fix (2) plus displacement (4) and visit-first refill placement (5).

## A2: any-temperature dark gate (paired replay on the shipped exchange)

_Generated by tools/paired_selection_a2.py. Every arm is the SHIPPED `_selection_exchange` (v2: order-preserving, displacement, visit-first, X<50) on the cached incumbents with only `gate_include` swapped; deltas are exact paired differences vs the incumbent. dark2 reads the device's actual any-temperature voltage (bsai/v12_rawany.py) in an UNCLAMPED 14-day window ending at cutoff-1 instead of extrapolating mext through the blackout; `true staleness` is measured from the cutoff, not the grid end._

| arm | mean d/scen | SE | W/L/T | flags/scen | unplanned due rate | blocks |
|---|---:|---:|---|---:|---:|---|
| today | **-47.5** | 43.45 | 21/16/11 | 0.5 | 0.667 | [-185.3, -81.6, 16.8, -20.6, -14.3, 0.0] |
| dark2_last_0.00 | **-69.1** | 49.4 | 20/16/12 | 1.44 | 0.435 | [-167.7, -25.2, -30.1, -231.0, 39.4, 0.0] |
| dark2_last_0.01 | **-35.6** | 45.21 | 20/16/12 | 1.54 | 0.405 | [-167.7, -25.2, 31.7, -91.8, 39.4, 0.0] |
| dark2_last_0.02 | **-42.5** | 44.55 | 20/16/12 | 1.77 | 0.365 | [-167.7, -25.2, 18.9, -89.2, 7.9, 0.0] |
| dark2_min3_0.00 | **-54.3** | 49.54 | 20/16/12 | 1.56 | 0.413 | [-167.7, -25.2, -30.1, -110.9, 7.9, 0.0] |
| dark2_min3_0.01 | **-39.1** | 44.79 | 20/16/12 | 1.77 | 0.365 | [-167.7, -25.2, 31.7, -81.1, 7.9, 0.0] |
| dark2_min3_0.02 | **-41.0** | 44.87 | 20/16/12 | 1.98 | 0.326 | [-167.7, -25.2, 18.9, -79.7, 7.9, 0.0] |

Best single axis: **dark2_last_0.00** (-69.1/scen); vs today it is heterogeneous (isolated diff -21.6/scen, W/L 12/11): the clamped mext gate and dark2 catch DIFFERENT populations -- in-grid gaps on live series (today) vs fully dark grids with a fresh any-temp reading (dark2). They are complements, not substitutes.

**UNION (today's dark | dark2_last_0.00): -90.3/scen** (SE 50.7, W/L/T 23/14/11, blocks [-185.3, -81.6, -107.2, -175.6, 7.9, 0.0], 1.6 unplanned flags/scen at due rate 0.455) -- a -42.8/scen upgrade over the shipping flags. Per-scenario deltas:

-140 +0 +0 -281 +50 -469 -153 -489 -301 -2 -8 -381 +1 -168 +247 -41 -508 -788 +156 +496 -206 +189 +397 -594 +281 +946 -1091 -145 -362 -435 -308 -292 -470 +191 +267 +524 -500 +3 +48 +0 +0 +0 +0 +0 +0 +0 +0 +0

**Exact flag spec for the integrator** (the dark branch of `gate_include` in bsai/forecaster.py becomes the union):

```
# branch 1 -- keep as shipped (clamped staleness, in-grid gaps):
dark1 = (stale_clamped > 30) & (margin - 0.001*stale_clamped < 0.02) \
        & (remaining >= 30)
# branch 2 -- observed any-temperature channel (bsai/v12_rawany.py):
stale_true = origin_index - last_valid_smoothed_position  # NO grid-end clamp
any_margin = RawAnyCache last daily median - 2.4, from an UNCLAMPED
             14-day window ending at cutoff-1   # NaN when channel is gone:
             # RawAnyCache.features_at clamps like the smoother -- do not
             # reuse it as-is; no fresh reading => gate cannot fire
dark2 = (stale_true > 30) & (any_margin < 0.00) & (remaining >= 30)
gate_include = dark1 | dark2   # dip branch stays dead at the base variant
```

Notes: tau=0.00 dominates 0.01/0.02 on both axes (precision 0.435 vs 0.326-0.405) -- the gate should fire only when the unfiltered channel already reads AT/below the 2.40 EOL line. The earlier overhang-mext attempt flooded (+520 cross-run) because mext extrapolates through long-ended devices; dark2 cannot flood -- no fresh any-temp reading, no flag. RawAnyCache must be updated in `predict` when the exchange is enabled (it is currently gated on `needs_raw`, the same dead path as the dip gate).

## Persistence-keyed zombie rule (paired replay, V16 unblock #1)

_Generated by tools/paired_selection_persistence.py. Every arm is the full combined pass -- persistence-zombies out + union gate (dark1|dark2_last_0.00) in + refill -- through the SHIPPED `_selection_exchange` on the cached incumbents; deltas are exact paired differences vs the incumbent. flag = margin<0.05 & p>0.4 (& dwell>42 in the dwell variants); demote at scenario s iff flagged NOW and flagged in >= N cutoffs STRICTLY BEFORE s (causal; alive batteries have zero recorded deaths by construction, first flags are exempt). JSON: `outputs/paired_persistence.json`._

| arm | mean d/scen | SE | W/L/T | demotions/scen | due sweeps (48 scen) | floor-5 demotions | blocks |
|---|---:|---:|---|---:|---:|---:|---|
| shipped fingerprint (dwell>42, no persistence) | **-90.3** | 50.7 | 23/14/11 | 1.98 | 24 | 45 | [-185.3, -81.6, -107.2, -175.6, 7.9, 0.0] |
| no demotion (gate+refill only) | **-12.1** | 33.87 | 11/13/24 | 0.0 | 0 | 0 | [-17.5, 6.7, 33.0, -104.1, 9.5, 0.0] |
| N3_dwell | **-124.0** | 43.78 | 27/7/14 | 1.25 | 10 | 42 | [-97.8, -117.0, -172.1, -175.6, -181.3, 0.0] |
| N3_nodwell | **-101.3** | 60.61 | 21/16/11 | 4.17 | 33 | 136 | [26.1, -23.6, -200.1, -313.7, -96.7, 0.0] |
| N6_dwell | **-107.4** | 42.77 | 24/7/17 | 0.92 | 4 | 39 | [-17.5, -130.4, -164.3, -175.4, -156.7, 0.0] |
| N6_nodwell | **-218.9** | 49.98 | 27/7/14 | 3.04 | 11 | 124 | [-97.8, -154.8, -488.8, -259.0, -312.9, 0.0] |
| N10_dwell | **-117.7** | 44.11 | 21/6/21 | 0.73 | 0 | 35 | [-17.5, -119.2, -237.4, -175.4, -156.7, 0.0] |
| N10_nodwell | **-171.0** | 45.95 | 24/6/18 | 2.38 | 5 | 108 | [-17.5, -130.2, -328.4, -244.4, -305.4, 0.0] |

Winner: **N6_nodwell** (-218.9/scen, 11 due sweeps vs 24 for the shipped fingerprint). Per-scenario paired deltas:

-140 +0 +0 +0 +0 +0 -153 -489 -301 -2 +98 -81 +1 -417 -88 -449 -511 -1327 -368 +12 -234 -420 +22 -1084 +373 +407 -1163 -332 -367 -549 -245 -197 -555 -118 -213 -710 -519 +6 -395 +0 +0 +0 +0 +0 +0 +0 +0 +0

**Exact causal spec for the integrator** (replaces the `slot_demote` fingerprint in bsai/forecaster.py):

```
flag_now = (margin < 0.05) & (p42 > 0.4)   # WITHOUT dwell
slot_demote = flag_now & (prior_flag_count >= 6)
# prior_flag_count: per-device count of PRIOR predict() calls whose
# flag_now was True. The forecaster is stateful and processes scenarios
# sequentially -- keep `self.flag_history: dict[str, int]` and increment
# AFTER computing slot_demote (history = prior cutoffs only, never the
# current one). Batteries with a recorded death never reach predict()
# again (iterate/harness drops EOL'd devices), so 'zero deaths' is
# automatic; a fresh split starts at zero counts -> no demotions in the
# first N scenarios (graceful, matches this measurement).
```
