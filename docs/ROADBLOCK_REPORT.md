# ROADBLOCK AUDIT — what is quantitatively holding back 2145.1 → ~1145

_Roadblock-detector role, 2026-08-22. All numbers computed this session from
`outputs/research_rowfeat.parquet` (19,890 scenario rows), `outputs/audit_ledger.csv`,
`outputs/v8_folds_cens.joblib`, `dataset/train` metadata and the 35 recorded
`outputs/val_*.json`. New artifacts: `tools/roadblock_l1.py` → `outputs/roadblock_l1.json`
(analytic frontier, xaware-lab convention), `tools/roadblock_l2.py` → `outputs/roadblock_l2.json`
(constraint re-derivation), `tools/roadblock_l3.py` → `outputs/roadblock_l3.json`
(process audit). Zero planner validations spent._

Position: baseline 2145.1 → best recorded op point 1924.5 (lm1.8/cap15; replicates
1924.5 / 1939.1 / 1972.5 / 1980.6). Remaining to ~1145: **−780 ± 30**.

---

## L1. The target arithmetic: ~1145 is INSIDE the information frontier

Analytic league (exact `tools/xaware_rule_lab.py` convention: best-day timing,
evaluator emergency queue, 5.6/planned swap, isolated emergency op per miss;
`roadblock_l1.json`):

| plan | total | early | late | swaps | hits | miss |
|---|---:|---:|---:|---:|---:|---:|
| ORACLE (swap exactly the due set) | **75.3** | 22.3 | 0 | 9.46 | 9.46 | 0 |
| GOD-VISIBLE+gates, optimal k (perfect ranking of every battery with p_cal≥0.001 OR dark/dip gate) | **250.4** | 20.1 | 172.5 | 8.56 | 8.56 | 0.90 |
| GOD-VISIBLE+gates, k≤15 | **263.8** | 19.7 | 186.7 | 8.42 | 8.42 | 1.04 |
| GOD-VISIBLE no gates (p_cal≥0.001) | 346.7 | | 268.5 | 8.25 | 8.25 | 1.21 |
| GOD-VISIBLE at p_cal≥0.02 (auditor's line) | 451.0 | | 368.8 | 7.75 | 7.75 | 1.71 |
| MODEL ranking, k≤15 (best k per scenario) | 1578.1 | 359.9 | 1107.5 | 11.4 | 5.08 | 4.38 |
| MODEL ranking, shipped budget min(15,⌈1.6Σp+1⌉) | **1718.5** | 488.3 | 1102.7 | 14.3 | 5.06 | 4.40 |

(The prompt's oracle floor 77.8 reproduces here as 75.3 — identical to
`outputs/xaware_rules.json` `oracle_reference`; the 2.5 delta is convention noise.)

**Analytic → realized conversion.** The shipped selection's analytic twin scores
1718.5 vs realized 1924.5–1980.6 → realization overhead **+206…+262** at 14.3
swaps / 4.4 misses. Decomposed: ops reality 235 vs analytic 127 (**+108**);
early realization (planner bundles insurance swaps earlier than window end)
+163…180; late −25…−70 (analytic queue is exact). Scaling overhead to the god
plan's 8.5 swaps / 0.9 misses: god realized-equivalent ≈ **350–420**.

**Verdict: 1145 sits ~725–795 points INSIDE the frontier.** Perfect ordering of
only what already carries signal (p_cal ≥ 0.001 or a gate) reaches ~350–420
realized. Corroborating external anchor: first place holds 1160 on public.

**Where the bound binds.** 69% of the god bound is late cost from the 43 truly
invisible dues (0.90/scen): median margin **0.205 V**, median 21 d to failure,
**84% fresh** (median staleness 0; only 16% dark) — genuine knee-entries with no
measured axis. By block the god plan costs [219, 275, 398, 240, 274, 98] — the
mid-year block s16–23 binds even for god (invisible dues 1.75/scen there; worst
scenarios s16: 3 invisible dues at margins 0.18–0.22 all fresh, s25, s15).

**The gap is ordering extraction, not information.** MODEL_cap15 − GOD_cap15 =
**1314 analytic pts/scen**, by block [1162, 1416, 2038, 1673, 1008, 590]. The
model needs 15 slots to catch 5.1 dues; god catches 8.4 with 8.4 slots.

**Capture arithmetic to 1145** (realized terms, from 1924.5): needed 780.
Auditor-realistic pools: ordering prize net 250–400 + gates ~95 (at 50%
conversion) + closing-block calibration repair 30–80 + search slack ~30 =
**405–605**. Shortfall **175–375** must come from a categorically better
mid-block ranking (the 0.214 → ~0.4+ top-12 range). So ~1145 requires BOTH the
full planner-reachable pool AND one forecast-generation step; neither alone
suffices.

---

## L2. Self-imposed constraints — re-derived, each answered

**(i) Hard cap 15 — NOT a roadblock.** God-plan optimal k: median 8, p75 10.25,
max 18; the cap clips **3/48** scenarios for a mean **13.4 pts/scen** (max 297.8,
s16). The model-ranking "optimal k" (median 17.5, 28/48 above 15, 257/scen) is an
oracle-k artifact — k chosen per scenario with realized labels — and is
contradicted by the measured cap17 A/B (+20.4, inside reroll noise) and marginal
volume precision 0.24. Keep 15.

**(ii) Stride-4 sampling — real dilution, wrong axis to blame.** Training
knee-share: within 21 d of crossing **0.49%** of 88,013 cutoffs vs **1.19%** at
scenario cutoffs (ratio **0.41**); within 42 d: 0.98% vs 2.28% (ratio 0.43). The
cause is uniform-history sampling vs the scenarios' late-life placement (fleet
ages into the tail), not the stride itself (uniform at any stride). Knee-week
evidence: 433 rows from 82 devices (~5.3 rows/crossing). The training
distribution under-weights the decision-relevant regime **2.3–2.4×**.

**(iii) Horizon grids — 42 d exact; short-horizon drops thin.**
`HORIZON_GRID[11] == 42` verified (exact, never interpolated). Drift-fit windows
(re-counted, total 550,560 = training report): h=7: 74,872 windows / **105
crossed** (0.14%); h=14: 73,601/213; h=42: 69,593/568 (0.82%). With
min_samples_leaf=60, at most ~1–3 crossing-dominated leaves are even possible at
h≤14 — short-horizon steepness rests on ~105 events fleet-wide.

**(iv) min_samples_leaf=60 / l2=1.0 — pooling hypothesis REFUTED; the defect is
magnitude shrinkage.** On margin-band 0.05–0.20 training rows: knee rows
(cross ≤42 d, n=480) produce **473 unique** drift predictions and **0.0%** exact
collisions with plateau predictions (n=4,146) — they are NOT pooled into plateau
leaves. But the level: predicted 42 d drop on knee rows averages **0.036 V**
against required ~0.12–0.20 V; only **12.1%** of knee rows are predicted to
reach the barrier (plateau mean 0.006 V — separation 6×, magnitude ~4× shallow).
Squared-loss regression on a 0.8%-event population shrinks the crossing drop
toward the mass; resolution is fine, amplitude is not. This — with (ii) and
(iii) — is the trainable root of the mid-block ordering gap.

**(v) Calibration clamp & emergency rank.** Clamp [0.35, 2.75]: **not binding**
— observed factors span 0.374–1.651 (production 0.374…1.411). The REAL defect is
the `min_events=25` bucket skip: **3/5 folds carry factor 1.0** in the 0–45 d
remaining bucket; on those folds' rows with remaining<45, Σp_cal = **56.4 vs 27
realized dues (2.09× hot)** (fitted folds: 4.3 vs 1) — ~29 phantom expected dues
concentrated in closing scenarios, inflating both slots and budget exactly where
wasted swaps are bought (block-6 model waste: analytic 729 vs god 98).
Emergency-rank approximation: E[rank] over the whole fleet's p gives dues mean
**4.66** vs realized miss-queue position **2.24** → **+24.3 pts/due** phantom
defer cost, ordered by battery id — same magnitude as the slot-boundary gain gap
(weakest planned gain ≈ 33), so it arbitrarily reorders the boundary. Cheap fix:
emergency_rank_scale ≈ 0.5.

**(vi) Candidate filter (margin 24 h, max 150) — NOT binding.** Of 203 missed
dues, 121 sat outside the filter with median p 0.0015 and gain p90 −28.5; only
**2/121** had gain > −24 (the only rows the 150-cap ever touched). The excluder
is the probability level, not the filter geometry.

**(vii) The 10–30 °C filter on FEATURES — a second, larger dead-channel layer.**
Scenario rows whose official channel is dark >30 d: 1,430. Of those, **651
(45.5%)** have an any-temperature raw daily median within 7 d of cutoff (601
even at ≥5 readings/day); mean freshness recovered on dark rows: **46.6 d**.
Dues on dark rows: 48, of which **41 (85%)** are raw-fresh. Of the 110
invisible dues (p_cal<0.02): 36 dark → **35 raw-fresh**. The gates only
*extrapolate* through the gap; an any-temperature feature channel (with the
per-device V/°C compensation already specced for V12) would *observe* it —
~0.73/scen of currently-invisible dues, ≈180 gross pts/scen, overlapping the
gate pool but with real measurements instead of decay guesses.

---

## L3. Process roadblocks: the instrument, not the throughput

35 validations recorded 03:46→16:11 (12.4 h). Mean compute **6.0 min/run** (not
8–15); wall clock ~21 min/decision; total validation compute 3.5 h of 12.4 h —
**cycle time is not the binding constraint** (only 3 ideas queued-unrun vs 35
runs; 5 more killed analytically without planner spend).

The binding constraint is **statistical resolution**: 16/28 audited decisions
landed inside their noise band (reroll ±52 for knob A/Bs, ~100 overlap floor for
design arms; lm1.8 replicate spread 1924.5–1980.6 confirms ±55). **8 decisions
asserted a direction while inside the band.** Realized damage this session:

* cens-forecast KEEP at −43.5 local (inside ±100) → **+179/scen on public**
  (the single measured decision error; later contained by the cap).
* vol1.2 KILLED at −36.8; gate-only KILLED at +63.3 — both unresolved coin
  flips; the gate-only read is the most expensive ambiguity because the entire
  remaining planner-reachable pool (every item worth 50–150/scen) is
  unmeasurable in single runs.
* capacity-pass initial −62 read was reroll noise; the paired harness
  (tools/capacity_pass_paired.py) resolved it to −2.5 — proof the fix exists
  and works, but it only covers capacity moves, not selection/order changes.

---

## Ranked TRUE roadblocks (realized pts/scen at the op point)

| # | roadblock | size | status |
|---|---|---:|---|
| 1 | Visible-set ORDERING extraction, mid-blocks (model vs god 1314 analytic; auditor-net realistic slice) | **250–400** | mechanism designed (order-level zombie demotion + budget exemption, gate forced-include) but every arm so far measured inside noise |
| 2 | Measurement floor: reroll ±52 + overlap ~100 vs effect sizes 50–150 | **gates #1, #3; cost ≥179 realized** | paired harness exists for capacity only |
| 3 | 10–30 °C filter applied to FEATURES (any-temp channel dark on 85% of dark dues) | **~180 gross / ~95 net** | not in any model; V12 spec lacks it |
| 4 | Closing-block calibration bucket-0 skip (min_events=25; 3/5 folds ×1.0; Σp 2.09× hot at remaining<45) | **30–80** | one-line fit change + refit |
| 5 | Emergency-rank defer inflation (+24.3/due, id-ordered, ≈ boundary gain gap) | **10–30** | one flag (`--emergency-rank-scale 0.5`) |
| 6 | True invisible wall at p≥0.001+gates (0.90 dues/scen, margin 0.205 V, fresh) | **~173 analytic** | accept this generation |
| — | NOT roadblocks (verified): cap 15 (13.4/scen, 3 scen), candidate filter (2/121 cap-bound), calibration clamp (0.374–1.651 inside), 42 d grid (exact), leaf pooling (0% collisions) | — | stop revisiting |

**Single highest-leverage unblock:** extend the paired-incumbent harness to
selection/order arms (same CP-SAT incumbents, per-scenario paired deltas + sign
test — ~±15–20 resolution), then push roadblock #1's two mechanisms (order-level
zombie demotion with Σp-exempt budget; gate/raw forced-include) through it.
Every remaining capturable pool is 50–150/scen; today's instrument cannot see
that size, and #1+#3 together (~350–500) are what stand between 1924 and the
mid-1400s — the rest of the way to ~1145 is the V12-generation mid-block ranking
(knee re-weighting per (ii)/(iii)/(iv) + any-temp features per (vii)).

---
---

# V16 RE-AUDIT (post selection-exchange, 2026-08-22 evening)

_The unblock above was built (tools/paired_selection.py) and it worked: it
overturned two "dead" verdicts (gate −47.6, exchange −79.2 / −119.6 X-gated)
that cross-run noise had buried, and V16 shipped the exchange. New op point:
val_v16 1958.3 / rerun 1944.2 / this audit's instrumented replicate **1879.0**
(the one allowed run: `tools/audit_operating_point.py` at HEAD →
`outputs/roadblock_ledger_v16.csv`, `outputs/roadblock_audit_v16.json`;
cross-run band on identical config now 1879–1958). Pre-V16 ledger preserved at
`outputs/roadblock_ledger_v13.csv`. New analysis: `tools/roadblock_v16.py` →
`outputs/roadblock_v16.json`._

## V16.1 Updated class ledger (pts/scen, n/scen) — what the exchange consumed

| class | V13 (audit 1980.6) | V16 (audit 1879.0) | delta |
|---|---:|---:|---:|
| INVISIBLE late | 487.9 (1.96) | **332.7 (1.50)** | **−155**: gate injections + refill caught invisible dues |
| VISIBLE-OUTRANKED | 269.6 (1.10) | **211.9 (0.85)** | **−58**: refill consumed a quarter |
| VISIBLE-UNECONOMIC | 226.5 (0.98) | 225.2 (0.98) | −1 (untouched) |
| TIMING | 36.5 (0.44) | 89.4 (0.67) | **+53**: nearest-visit adds land after some EOLs |
| VISIBLE-DROPPED | 57.1 (0.19) | **158.5 (0.48)** | **+101 = the fingerprint SWEEP, now named**: d_ccd3a65228a9 ×6, d_4643496a7525 ×5, d_7aa40cb4e3a2 ×3, d_3b189e96404c ×3 at med p 0.551 — in-slot dues the exchange deferred (paired predicted +363/defer at flag due-rate 0.48) |
| NEVER-DUE early | 519.9 (6.77) | **453.8 (6.23)** | −66 net; zombie-slot early 339.8 → 252.7 (**−87**, all from d_b5b678a3f79f: 48 → 12 served, remaining only s0-2 + the X-gated s39-47); refill added back ~+21 non-zombie never-due |
| POST-WINDOW early | 114.0 (2.23) | 127.5 (2.67) | +14 |
| DUE-EARLY | 34.1 | 36.5 | +2 |
| totals | late 1077.5 / early 667.9; served 14.23; recall 0.553 | late 1017.7 / early 617.8; served 14.54; recall 0.597 | −60 late / −50 early |

**Zombie pool: 87 of 340 consumed.** The other four documented floor-zombies
(d_3d26e12378f1 39, d_c9a2ce794b68 38, d_d4b4272d5229 33, d_d9d695df1683 33
served scenarios; p 0.20–0.93) never fire the fingerprint — their dwell 11–23
is under the required 42 (one margin 0.053 > 0.05) — so **253/scen of zombie
early remains untouched**. Outranked: 58 of 270 consumed. Give-back +168
(sweep 101 + timing 53 + post-window 14); net class movement −110 ≈ the
replicate delta (−101.6).

## V16.2 Updated god-gap and remaining extraction pools

God-visible bound unchanged (~250 analytic ≈ 350–420 realized). Realized V16
band 1879–1958 → extraction gap **~1460–1600**. Ranked by realistic net:

| # | pool | gross | realistic net | state |
|---|---|---:|---:|---|
| 1 | 4 unfingerprinted floor-zombies + sweep give-back | 253 + 101 | **200–320** | persistence separator measured (45 flags/0 deaths vs die-within-3–7-flags); 2-min test on cached incumbents (`--reuse`) |
| 2 | INVISIBLE remainder | 333 | **40–110** | ~0.48 dues/scen are dark dues the staleness CLAMP forfeits (V16.3c); rest info-limited |
| 3 | mid-block ranking (pi-hybrid 0.292, pending) | — | **50–90** | V16.4; two-phase and V12 both NO-GO, this is the live bet |
| 4 | OUTRANKED remainder + UNECONOMIC | 437 | 50–150 | the exchange proved selection-layer conversion works; deepen refill quality |
| 5 | TIMING give-back | 53 of 89 | 30–50 | timing-aware placement: visit day only if ≤ the battery's best day |
| 6 | insurance / post-window / due-early | ~330 | ~0 | price of uncertainty at 14.5 swaps |

Realistic nets sum to ~370–720 from ~1920 (band center) → **~1200–1550**.
~1145 still needs the mid-block forecast generation on top of full pool
capture — same conclusion as round 1, but conversion is no longer hypothetical.

## V16.3 NEW self-imposed constraints, quantified (`roadblock_v16.json`)

* **(a) X-gate 50 — NOT binding; exactly right.** Skips 9 scenarios (s39–47):
  forfeits −5.9/scen of measured value, avoids +46.3/scen of measured harm;
  48-mean equals the measured s40-47 gate (−119.6). X<100 would forfeit −28.9.
* **(b) Refill p>0.05 floor — NOT binding.** Mean supply beyond the limit at
  p>0.05 is 10.85/scen; 2/48 scenarios ≤2. Refill increment measured −27.7.
* **(c) Dark-gate staleness CLAMP — the largest new constraint.** Production
  staleness is computed inside the causal grid (clamped at the last stable
  day), so grid-overhang devices read ~0: production fires on **12 of 60**
  frame-dark rows, **11 of 34** dark dues → **23 dues forfeited ≈ −42/scen**
  of measured dark value (top earner d_124f1e85339f converted 1 of 6 due
  scenarios; 5 remain INVISIBLE in the V16 ledger). The naive unclamp
  validated **+520** (floods: the measured class is partially
  resumption-defined). The CAUSAL discriminator is the any-temperature
  channel (L2-vii: 41/48 dark dues still report raw readings within 7 d):
  gate = true staleness>30 ∧ mext<0.02 ∧ remaining≥30 ∧ **any-temp reading
  ≤7 d old**. Untested; priced 40–110/scen.
* **(d) Displacement victim = lowest-p planned.** 35.4% of removals hit a real
  due (+44…+356 vs −40…−299 on never-due victims) ≈ +67/swap-in drag vs an
  oracle victim; no plan-time separator known beyond persistence.
* **(e) Zombie fingerprint (margin<0.05 ∧ dwell>42 ∧ p>0.4).** Catches 1 of 5
  documented zombies; sweeps dues at flag-rate 0.48 → the +101 give-back.
  dwell>42 is why the other four never fire; persistence is the separator.
* **(f) Dip gate**: 30 frame rows at due-rate 0.500 but paired-neutral (+1.9)
  — a passenger; keep or delete deliberately.

## V16.4 pi-hybrid worth (mid top-12 0.214 → 0.292)

+0.078 × 12 = **+0.94 catches per mid scenario** (16/48 scenarios). Realized
per-catch 245–300 (lm A/B 246/miss; paired gate-due conversion −298.6): pure
reorder **+76…94 on the 48-mean**; if conversions displace the weakest planned
(35.4% due) **+49…61**. L1 cross-check: 9.9% of the mid-blocks' ordering gap
(2038+1673 analytic) = 61 analytic ≈ 69–77 realized. **Book 50–90/scen**
(1920 → ~1830–1870) — real, but one-seventh of the remaining gap.

## V16.5 Ranked roadblocks (V16) and the next unblock

1. **Persistence-blind zombie rule** — 253 untouched + 101 sweep give-back;
   net **200–320**. Mechanism proven (−97/flag on the caught zombie), separator
   measured, instrument cached.
2. **Dark-gate staleness clamp** — 23/34 dark dues forfeited; **40–110**;
   causal fix is an any-temp LIVENESS condition on the gate (V12 already
   proved any-temp fails as a ranking FEATURE — this is a gate input instead).
3. **Mid-block ranking generation** — **50–90** available now via pi-hybrid;
   the remaining ~333 INVISIBLE + 437 OUTRANKED/UNECONOMIC needs the next
   model, not knobs.
4. **Timing-aware add placement** — **30–50** of the +53 TIMING give-back.
5. **Measurement discipline** — identical-config band still spans 79 pts
   (1879–1958); every next call must ride the paired harness (2-min --reuse).

**Single highest-leverage next unblock: the persistence-keyed zombie rule** —
demote batteries with ≥N historical fingerprint flags and zero deaths, exempt
first-flag batteries (kills the sweep too). Largest reachable pool (200–320),
proven mechanism, cached 2-minute instrument, touches no probability.
