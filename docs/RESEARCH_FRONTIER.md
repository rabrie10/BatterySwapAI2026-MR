# Frontier hunt: methods for the mid-year ranking gap (A) and the late pool (B)

_Researcher role, 2026-08-22. All numbers computed this session from
`outputs/frame_oof_raw_beta.parquet` + `frame_oof_cal.parquet` (19,890 rows, 454
due-rows, 48 scenarios) joined with strictly-causal features extracted at
cutoff−1 from the hourly parquet (`tools/research_extract.py` →
`outputs/research_rowfeat.parquet`, `research_traj.npz`). Probes:
`tools/research_cohort.py`, `research_signals.py`, `research_knn.py`; JSONs in
`outputs/research_cohort.json`, `research_signals.json`, `research_knn.json`.
Anchors reproduced: top-12 realized rate open 0.589 / mid 0.214 / late 0.286;
dues/scen 13.25 / 8.56 / 6.56._

## 0. The discovery that reframes both frontiers: STALENESS

The mid-block ranking failure and the invisible late pool are substantially the
same population, and a quarter of it is not a forecasting problem at all — it is
a **dead data channel**. The official smoothing (10–30 °C filter, ≥5
readings/day, 3-of-7 rolling) goes dark on cold rooms in winter; `margin` in
every model is then a months-old reading, and the Wiener p collapses to ~0.003.

| mid-block (s16–31) | n | due rate | notes |
|---|---:|---:|---|
| fresh rows (staleness ≤ 7 d) | 5,499 | 0.016 | |
| stale rows (staleness > 30 d) | 690 | **0.055** | 3.4× the fresh rate |
| mid dues with staleness > 30 d | 38 of 137 | — | **28% of all mid dues** |

Within the stale pool the last-observed state ranks the dues almost perfectly —
the information exists, the pipeline discards it:

| score on stale>30 rows (mid) | AUC (oriented) |
|---|---:|
| last-observed margin (low→due) | 0.955 |
| **margin − 0.001·staleness ("dark-decay margin", low→due)** | **0.965** |
| beta30 (ShapeCache is unfiltered → still live when smoothing is dark) | 0.757 |
| p_cal | 0.937 (but level ~0.003 → never picked: only 5/48 stale dues in top-15) |

Rank stale rows by last margin, take top-k per scenario, mid block: k=1 →
**0.625** realized, k=2 → 0.531, k=3 → 0.437 — against a 0.214 top-12 baseline.
The foresight ladder's 0.15 ceiling does not apply here: those probes scored
margin/-slope on *observed* series; these dues are dark, they cross when
readings resume (due rows peak Feb–May), so a data peek sees nothing either.
Extrapolating the decay through the dark gap (margin − 0.001·staleness < 0)
removes the calendar confound physically: late-block flags go 39 rows/6 dues
(0.15) → **6/6 (1.00)**; no month gating needed.

## 1. Hunt results (everything measured, live and dead)

### H1. Peer/cohort dynamics — DEAD as ranking, DEAD as volume
- P(due | building had k crossings in prior 60 d): k=0 → 0.0205, k=1 →
  0.0177, k=2 → 0.0258, k=3+ → 0.0686 (base 0.0228). The k=3+ lift is building
  identity in disguise: the **within-building temporal contrast is null**
  (rate 0.0166 when the building recently had crossings vs 0.0171 when not;
  higher in only 6/17 buildings). Mid-block due-location lift is **0.80** —
  dues are *less* likely in recently-hit buildings there.
- Mid top-12 with a 25% cohort rank-blend: 0.214 → **0.177** (worse).
- Scenario due-count series autocorrelates (lag-1 0.854, but windows overlap
  35/42 d; disjoint lag-6 0.409). Trailing-42d observable crossings predict the
  count (MAE 3.34 vs constant 5.67 on 2nd-half split) but add nothing over the
  model's own Σp (corr with Σp-residual **−0.16**). Analytic quota economics
  (−270·catch +133·waste): fixed k=12 → −159/scen beats every adaptive variant
  (best −116); oracle-count quota −240 shows the ceiling exists but observable
  crossings do not reach it. **No proposal.**

### H2. Within-day hourly shape beyond beta — DEAD
Per-day stats computed from all 8.5M hourly rows (pulse depth p50−p05, upper
room p95−p50, night(0-6h)−day(12-18h) median gap, deep-reading fraction,
mean−median skew), trailing-14d and rise-ratio (vs own prefix median) forms.
AUC on the knee band (margin 0.05–0.20, remaining ≥30):

| feature | pooled band | mid-block band |
|---|---:|---:|
| p_cal / margin / beta30 | 0.781 / 0.692 / 0.628 | 0.651 / 0.628 / 0.536 |
| best new hourly stat | rise_deep_frac 0.636 | h14_room 0.608 (orientation unstable) |
| raw_min3 / raw_slope7 (raw-daily channel) | 0.754 / 0.572 | **0.709 / 0.694** |

No hourly-shape statistic beats the existing axes anywhere; the beta collapse
in mid (0.536) is shared by all of them. The only channel that *gains* rank
power in mid is the raw-daily one. **No proposal from hourly shape.**

### H3. Raw-daily channel — LIVE, small and sharp
The imminent-fail rescue framing is mostly empty: of 22 dues failing in window
days 1–14 with raw_min3<2.42 ∧ margin>0.03, 17 already have p≥0.1 (model sees
them). The value is a strict gate on rows the model ranks 27th–53rd:

| gate `raw_min3<2.40 ∧ margin>0.03 ∧ p_cal<0.1` | n | due rate |
|---|---:|---:|
| all (0.62/scen; 14 batteries, 8 buildings) | 30 | **0.500** |
| open block | 5 | 0.000 (one repeated battery, cost ~90 pts each) |
| mid block | 18 | **0.667** |
| late block | 7 | 0.429 |

**0 of the 15 due-rows are in the top-15 by p_cal** — verifiably un-banked.
A raw_slope7<−0.001 guard *hurts* (0.500 → 0.278): keep the gate unguarded.
Flagged rows are fresh (staleness 0/0/0 quartiles) — this is the smoothing-lag
information, orthogonal to H0's dark-channel pocket (overlap 8 rows).

### H4. kNN on 90-day margin-trajectory shape — DEAD, decisively
15×6-day-block trajectories, three normalizations, leave-building-out (deploy-
honest: reference = resolved train split), k∈{25,50}, 16,472 usable refs:

| mid-band AUC | p_cal | margin | knn shape (anchor) | knn shape (zscore) | knn level |
|---|---:|---:|---:|---:|---:|
| | 0.694 | 0.656 | 0.444–0.471 | 0.363–0.394 | 0.566 |

Shape-normalized similarity carries **no** transferable hazard (zscore is
anti-signal); only level survives, and level is margin. Top-12 mid blends all
degrade (0.214 → 0.12–0.18). Time-resolved variant identical (0.465). This
closes the "trajectory shape" axis: with 82 events there is no shared shape
that survives leaving the building out — consistent with the physics record
(band-time CV 0.8–1.0, no shared discharge curve).

### H5. Failure-time literature conversion
Tried: dynamic frailty / self-exciting intensity (=H1, dead), case-base kNN
hazard (=H4, dead), calendar baseline-intensity quota (=H1 volume, dead).
The one that converts: **first-passage with an observation-gap clock** — the
dark-decay margin of §0 is exactly the literature's "last observation carried
forward + drift over the gap", and it is the live lever. A model-side variant
(advance the Wiener horizon by staleness days) is a probability-layer change —
the class that died 5/5 — so it is proposed only in selection-layer form.

## 2. Frontier arithmetic (what is purchasable)

110 invisible dues (p_cal<0.02, 2.29/scen; median margin 0.168, fail 26 d out,
81% beta-elevated). Measured coverage: dark-decay gate 28, raw-dip gate ~6
(overlap counted once) → **~25% of frontier B is purchasable now**; the
remaining ~80 are fresh knee-entries at margin ~0.17 that no measured axis
(shape kNN 0.44, hourly ≤0.61, cohort null, raw NaN-or-high, 21-d foresight
0.15) can rank — treat them as information-limited, not model-limited.
Ordering *within* the invisible pool by low margin realizes only 0.25 at
top-1/scen (decays to 0.10 by k=5) — no order lever there beyond the gates.

## 3. Top-3 proposals (ranked by expected points × planner survival)

Survival calibration used: probability-layer reshapes died 5/5 (isotonic, dwell
×4, rank map, knee floor); order/volume changes lived (cap 15, lm1.8, search
240). All three proposals are selection/order mechanisms; none touches p.

### P1. Dark-decay resurrection gate (frontier A+B) — EXPECTED VALUE: LARGE
**Mechanism:** batteries whose smoothed channel went dark while deep. Gate at
plan time: `staleness > 30 ∧ (margin − 0.001·staleness) < τ`, τ∈[0, 0.02];
inject into the planner's served set (forced candidate slots, max 2/scen,
ranked by extrapolated margin), never via p.
**Numbers (τ=0.02):** 60 rows (1.25/scen), realized **0.567** (34 dues; blocks
4/3, 47/25, 9/6); 29/34 dues outside top-15; 7 batteries / 6 buildings; τ=0:
42 rows at **0.667**. FPs cheap: 80% die anyway (median 66 d later).
**Value:** mid top-12 0.214 → 0.281 with ≤2 injected slots; mid catches +0.82
/scen; naive local EV (union with P2) ≈ +190/scen; at 50% planner conversion
still ≈ +95/scen. Not banked: p_cal median 0.003.
**Why it transfers:** staleness is building-invariant by construction (KS 0.02–
0.10 across hard holdouts, lowest of all features); the mechanism is the shared
official smoothing filter, which the public split runs identically.
**Risk:** 7 due batteries is thin evidence; per-battery flags repeat 4–6
consecutive scenarios (row-level value is real per-scenario, statistical
evidence is 7 independent lives). Knee-floor precedent (+127 analytic → +14
planner) does not apply mechanically — those dues were p 0.2–0.5 (partially
banked); these are p 0.003 and rank 27–53 (cannot be banked) — but the planner
conversion must still be demonstrated.
**Cheap control (minutes):** (a) leave-one-building-out gate rate: refit
nothing — the gate has no fitted parameters except τ; report per-building rate
table (done: 6 buildings all positive at τ=0.02 mid). (b) Feasibility tester:
one validate run with a forced-include list produced from the gate (selection-
layer hook, cap unchanged at 15, injected max 2/scen displacing the lowest-p
picks). Success = late component falls ≥150/scen on mid scenarios without an
early/capacity give-back; failure mode to watch = planner refuses forced picks
or reroutes tours (capacity).

### P2. Raw-dip injection (frontier A, mid-weighted) — EXPECTED VALUE: MEDIUM
**Mechanism:** the 7-day smoothing lag hides fresh dips; the raw-daily channel
(bsai/rawdaily.py) sees them ~3.5 d earlier. Gate: `raw_min3 < 2.40 ∧ margin >
0.03 ∧ p_cal < 0.1`, same forced-candidate injection, max 1/scen.
**Numbers:** 30 rows (0.62/scen) at **0.500**; mid 18 rows at **0.667**; 0/15
due-rows in top-15; 9 due batteries / 7 buildings; open block is 5 FP rows from
one battery (cheap, ~90 pts each; do NOT add a slope guard — it drops the rate
to 0.278).
**Value:** ~15 avoidable scenario-misses ≈ +60/scen naive, ~+30/scen at 50%
conversion. Union with P1: 84 rows, 43 dues, 0.512 realized, 38/43 outside
top-15, 28/110 invisible-due coverage.
**Cheap control (minutes):** per-building rate table (no fitted params);
sweep thresholds {2.40, 2.42} × margin floor {0.03, 0.05} on the frame (already
in `research_signals.json`); then ride the same forced-include validate as P1
(one run tests the union).

### P3. Mid-block re-rank: blend p_cal with raw_min3 rank — EXPECTED VALUE: SMALL, CHEAPEST
**Mechanism:** pure within-scenario reorder (no volume change, no injection):
score = (1−w)·rank(p_cal) + w·rank(−raw_min3), w=0.5.
**Numbers:** top-12 by block: open 0.589 (unchanged), mid 0.214 → **0.266**,
late 0.286 → 0.266. Net +0.5 catches/scen mid, −0.2 late ≈ +25/scen naive.
**Why ranked 3rd:** weakest effect and it touches the whole ordering (the
planner's tie-breaking on cost tables may swallow it), but it is the only
proposal that needs zero new mechanism — one line in the ranking used for
candidate selection. If P1/P2's forced-include hook is contentious, this is the
fallback that expresses the same information.
**Cheap control (minutes):** recompute the blend's rate-at-k for k∈{10,12,15}
per block on `research_rowfeat.parquet` (done for k=12); confirm w=0.25 as the
conservative point (open 0.573/mid 0.250/late 0.292 — no late give-back).

## 4. Kill list (do not spend planner runs)
- Cohort/building recent-crossing hazard, any form (within-building null).
- Trailing-crossing adaptive swap quota (fixed k dominates analytically).
- Trajectory-shape kNN / any "similar decay curve" transfer (anti-signal).
- Within-day statistics beyond beta30 (pulse depth, night-day, residual
  spread, skew, deep fraction — all ≤0.64 AUC, below margin alone).
- Ordering *inside* the p<0.02 pool by margin/beta (top-1 0.25, top-3 0.17 —
  below the mid marginal pick; beta-blend actively harmful 0.00–0.04).

## 5. Handoff artifacts
- `outputs/research_rowfeat.parquet` — frame + raw-daily + hourly-shape +
  staleness features at cutoff−1 for every scenario-battery row (the gates are
  columns away: `staleness`, `margin`, `raw_min3`, `p_cal`).
- `outputs/research_traj.npz` — 90-day margin trajectories (for reproducing the
  kNN negative).
- `tools/research_extract.py` (rebuild), `research_cohort.py`,
  `research_signals.py`, `research_knn.py` (all rerun in minutes).
- For the V12 retrain spec: `margin − 0.001·staleness` is the staleness-honest
  margin ("mext"); identical to margin on fresh rows, and the single highest-AUC
  quantity measured on the stale pool (0.965). It belongs in the invariant
  feature set regardless of what happens to the gates.
