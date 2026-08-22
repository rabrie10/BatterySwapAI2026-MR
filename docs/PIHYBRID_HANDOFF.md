# π-Hybrid Improvement Plan — Handoff for a fresh Claude Code session

You are picking up a live competition solution cold. This document is self-contained:
it tells you where the code is, what has been measured, which instruments to trust,
and the exact ordered experiments to run. Read it fully before editing anything.
Lower competition score is better.

---

## 0. THE ONE THING

The last submission (V19, local 1716) scored **2113 on the public leaderboard — worse
than the 2078 starting point.** The 430-point local gain did not transfer. Diagnosis:
we over-tuned the *planner's selection machinery* on the 24 train buildings, and it
does not generalize to the unseen public buildings. Your job is **not** to add more
train-tuned rules. It is to ship the one asset that is measured to generalize — the
π-hybrid model's ranking — through the *simplest possible* planner, and to calibrate
volume from the public row rather than from train.

**Expected realistic outcome: recover below 2078 and convert the π-hybrid's
unseen-building ranking edge (hard-holdout PR-AUC 0.547 vs the incumbent's 0.428) into
a real drop, plausibly into the ~1700–1950 public range. The ~1000 goal is almost
certainly out of reach with this model family — the gap to first place (1160) is
fundamental forecasting power on unseen buildings, not tuning. Do not promise it; chase
the honest, measured improvement.**

---

## 1. WHERE EVERYTHING IS

- **Repo:** `C:\Users\MAHDIN\.vscode\BSAI_challenge\BatterySwapAI2026-MR`
- **Remote:** `github.com/rabrie10/BatterySwapAI2026-MR`
- **Branch:** `claude/v13-pipeline` (do all work here; do NOT touch `main` until a public
  result justifies it). `git fetch && git checkout claude/v13-pipeline && git pull`.
- **Key commits on that branch:**
  - `737379a` — **π-hybrid ship candidate** (the model you are improving)
  - `157513e` — V19 ship config (script defaults, runtime governor)
  - `8a333d6` — V19 model (persistence demotion + dark gate + exchange — the machinery you will STRIP)
  - `d1b17ef` — V16 (paired-verified selection exchange)
- **Git author note:** commits land as `Nasser <mahdi.nasser@lyse.no>` (no git author
  configured). Fine, but amend if you want different attribution.

### π-hybrid artifacts (all present, committed / reproducible)
- `models/pihybrid.joblib` — production model: 66-feature GBDT + two-phase changepoint
  posterior features, volatility 1.2, RemainingCalibration, live incremental filter,
  fleet scale 1.160 mV. **Loads through `script.py` via `BATTERYSWAP_MODEL_PATH`.**
- `outputs/twophase_pihybrid_model.joblib` — the 5 fold (out-of-fold-by-building) models,
  for validation. (git-ignored; reproduce with `tools/twophase_pihybrid.py` if missing.)
- `outputs/twophase_pi.parquet` — precomputed per-device changepoint posterior π.
- `bsai/twophase.py` — the filter, `PiHybridModel`, `ProductionPiHybrid`, and
  `PiFilterCache` (incremental forward filter; batch-equivalent to 0.0 over 461 devices).

### The documents that hold the measured record (READ THESE)
- `docs/PIHYBRID_SHIP.md` + `docs/PIHYBRID_FINDINGS.md` — the π-hybrid gates, the exact
  validate command, the budget rule, runtime, artifact paths.
- `docs/TWOPHASE_FINDINGS.md` — the changepoint filter's parameters and gate results.
- `docs/ROADBLOCK_REPORT.md` — the god-ranking bound and the per-scenario cost ledger
  (which pools are reachable; INVISIBLE ~488/scenario is not).
- `docs/TRANSFER_STRESS.md` — the leave-building-out harness and its verdicts (this is
  the instrument that CORRECTLY predicted the V19 public failure — trust it).
- `docs/PAIRED_SELECTION.md` — the selection-exchange arms (the machinery to strip) and
  the exact-replay paired harness.
- `docs/V11_TRANSFER_FINDINGS.md`, `docs/V10_FINDINGS.md` — the earlier transfer saga.

---

## 2. ENVIRONMENT SETUP (do this first, ~10 min)

```bash
# Python 3.13 venv already exists at .venv; if not:
py -3.13 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # skip torch line if it stalls; pi-hybrid does not use torch
# scikit-survival in requirements.dev.txt needs MSVC — skip it.

# Dataset is NOT in git. Download the train split (needs network, one-time):
./.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('batteryswapaichallenge/BatterySwapAI-2026-Public', repo_type='dataset', local_dir='dataset')"
# -> dataset/train/{devices.csv,eol_times.csv,battery_metrics.parquet,scenarios.json}

# Sanity: 66 tests must pass.
OMP_NUM_THREADS=3 ./.venv/Scripts/python.exe -m unittest discover -s tests
```

Compute note: a planner validation is ~6–12 min on this 12-core box. Use
`OMP_NUM_THREADS=3` and run at most 2–3 validations concurrently.

---

## 3. WHAT THE π-HYBRID IS AND WHY IT IS THE RIGHT BASE

It is the censored-drift GBDT forecaster with two extra features: a per-device
**changepoint-onset posterior π** (P(the cell has entered its steep post-knee decline |
history to cutoff)) and a π-weighted drift, from a two-phase Wiener filter fitted
out-of-fold by building. Measured facts (all in the docs above):

- **It is the only ranker that generalizes to unseen buildings.** On the leave-building-out
  transfer harness it scores hard-holdout PR-AUC **0.547 vs the incumbent cens model's
  0.428, winning all 5 building holdouts with no collapse** (the incumbent collapses to
  0.29 on the hardest fold). Mid-year ranking (the frontier where dues are hardest):
  top-12 realized **0.339 vs 0.214**.
- Standalone planner score at the re-anchored budget: **1861 / 1839** (two runs). Worse
  than V19's 1716 *locally* — but local is exactly the number that lied about V19.
- Ship it today with: `BATTERYSWAP_MODEL_PATH=models/pihybrid.joblib` and
  `BATTERYSWAP_DUE_MULTIPLIER=1.25`. The `1.25 = 1.6 × 10.28/13.12` re-anchors the swap
  budget for the π-hybrid's higher (honest) expected-due level; details in
  `docs/PIHYBRID_SHIP.md`.

---

## 4. THE DIAGNOSIS YOU ARE ACTING ON (why V19 failed public)

Public component decomposition, V19 (2113) vs the 2078 baseline:

| | swaps | early | late | capacity+OT | total |
|---|---:|---:|---:|---:|---:|
| baseline | 21.7 | 946 | 634 | 410 | 2078 |
| V19 | 17.8 | 672 | **1037** | 330 | 2113 |

The volume cut worked exactly as designed (early −274, capacity −80). But **late exploded
+403 — misses jumped ~2.3 → ~3.6 per scenario, recall fell ~0.76 → ~0.62.** We cut
volume and the batteries we stopped swapping were *real dues on public*, not the waste
they were on train. A volume cut only pays if the ranking still finds the dues inside the
smaller swap budget; on unseen buildings our train-tuned selection did not.

**The pattern across all three public submissions is the real lesson:**
- V7 (light selection tuning): local 2293 → public 2167 — transferred.
- V10 (more): local 2056 → public 2179 — didn't.
- V19 (heavy: cap + exchange + persistence + dark gate + late-tilt): local 1716 → public 2113 — catastrophic.

**Transfer got monotonically worse as we added train-tuned selection machinery.** The
paired harness measured those arms as "exact on train incumbents" — which is precisely
the overfitting signature. The selection layer memorized the 24 train buildings.

---

## 5. INSTRUMENT TRUST HIERARCHY (critical — most mistakes came from here)

1. **The public leaderboard row** — the only ground truth for transfer. Decompose it:
   `swaps = battery_swap / 0.25`; misses from `late_swap` via the queue formula
   `late(m) = 10·[27.26·m + m(m−1)/2]`; then `planned = swaps − misses`, and precision/recall.
   Two env-controllable knobs are calibrated from this: the volume cap and the budget.
2. **The leave-building-out transfer harness** (`tools/transfer_stress.py`) — the best
   *train* proxy for a MODEL's unseen-building ranking. It correctly predicted the V19
   failure. Use it for every model/ranking go/no-go.
3. **The paired-incumbent harness** (`tools/paired_selection*.py`) — exact plan-diffs,
   no reroll noise. BUT it runs on *train* buildings, so it validates a planner-config's
   train behavior, NOT its transfer. **This is the trap that produced V19: a planner-config
   change that is "exact −219 on the paired harness" can still be train-overfit and fail
   public.** Use it only to confirm a change does what you think mechanically — never as
   evidence a selection rule will transfer.
4. **Plain local OOF-by-building** (`tools/validate_v6.py`) — has now over-promised on
   public three times, worsening each time (+126 → +397 gap). Treat any local gain under
   ~200 as noise, and any local gain from selection tuning as probably illusory.
   Reroll noise alone is ±52 (CP-SAT wall-clock); the scenario-overlap floor is ~100.

**Rule: MODEL/ranking changes are validated on the transfer harness. PLANNER-CONFIG
changes have NO trustworthy train instrument for transfer — keep them simple and
economically principled, and calibrate them from public.**

---

## 6. THE EXPERIMENT PLAN (ordered; each step has a gate)

### Step 1 — Wire the two missing env toggles (small edit, enables everything below)
`script.py::build_planner_config` does not expose `selection_exchange` or
`capacity_repair`. Add:
```python
selection_exchange = os.environ.get("BATTERYSWAP_SELECTION_EXCHANGE", "1") != "0"
capacity_repair    = os.environ.get("BATTERYSWAP_CAPACITY_REPAIR", "1") != "0"
```
and pass them into `PlannerConfig(...)`. Keep `capacity_repair` default ON (it is
evaluator-exact, not train-fit — it transfers). Make `selection_exchange` default ON for
now but you will test it OFF. Run the 66 tests.

### Step 2 — The isolation submission: π-hybrid, selection machinery OFF, looser cap
This is the highest-value, best-reasoned move. It tests the core hypothesis: *a better
MODEL with a SIMPLE planner (the config that transferred as V7) will convert the ranking
edge, where heavy selection tuning failed.*

Config for the submission run (`script.py` env):
```
BATTERYSWAP_MODEL_PATH=models/pihybrid.joblib
BATTERYSWAP_DUE_MULTIPLIER=1.25
BATTERYSWAP_MAX_PLANNED=20          # public proved cap-15 starves recall; loosen
BATTERYSWAP_SELECTION_EXCHANGE=0    # strip the train-overfit exchange/persistence/dark-gate
BATTERYSWAP_LATE_RISK_MULTIPLIER=1.8
BATTERYSWAP_CAPACITY_REPAIR=1
```
Before submitting, prove it runs valid and in budget:
```bash
BATTERYSWAP_DATASET_PATH=dataset BATTERYSWAP_SPLITS=train OMP_NUM_THREADS=6 \
  ./.venv/Scripts/python.exe script.py
# must print planned=48 degraded=0 deferred=0, produce submission.csv (19890 rows, 48 scenarios),
# elapsed well under the 30-min cap (projected ~16-19 min for the 96 public+private scenarios).
```
Gate before you spend the submission: run the transfer harness on the π-hybrid model
(it is already 0.547 — confirm it still loads and dispatches out-of-fold). The planner
config you cannot validate for transfer locally — that is what the submission buys.
**This is the submission to make first.** Read its public row with the Step-5 recipe.

### Step 3 — Calibrate volume from the public row (after Step 2 returns)
From Step 2's `battery_swap`/`late_swap`, compute realized swaps and misses.
- If misses are still high (recall-starved): raise `BATTERYSWAP_MAX_PLANNED` (22, 25) or
  set `BATTERYSWAP_DUE_MULTIPLIER=none` to let the cap alone bind.
- If early/`early_swap` ballooned (over-swapping waste): tighten back down.
The economically-correct swap count is where the marginal swap's due-probability crosses
the break-even ~0.21 all-in (see `docs/ROADBLOCK_REPORT.md` for the exact per-tail,
per-scenario-date break-even table). One knob, calibrated from one public number.

### Step 4 — If Step 2 confirms the model transfers: push the ranking further
Only if Step 2 beats baseline. Options, in order of evidence, each gated on the
**transfer harness** (`tools/transfer_stress.py`, hard-holdout PR-AUC must beat 0.547):
- (a) **Ensemble as a Σp-preserving level source.** `tools/ensemble_matrix.py` already
  found a remaining-keyed logit mix of {cens, two-phase, qhead} at hard-holdout mean
  0.474; the two-phase/π component drives it. It was GO specifically as transfer
  insurance. Deploy it as the ranking, keep the incumbent per-scenario Σp multiset
  (order-only) — details in `docs/ENSEMBLE_FINDINGS.md`. NB the ensemble's *reorder* was
  a NO-GO on the train paired harness, but that is a train instrument; its transfer case
  (the 0.474 hard-holdout) is the real argument.
- (b) **Full-strength π-hybrid retrain** (stride 2, more iterations) — only the FEATURES,
  no new selection rules. Gate on transfer harness, not local.
- (c) **litreview stage-2**: feed π-weighted drift deeper / per-device onset frailty.
  See `docs/RESEARCH_FRONTIER.md` and `outputs/litreview_methods.md`.

### Step 5 — Read every public row the same way
```
swaps  = battery_swap / 0.25
misses ≈ solve  late_swap/10 = 27.26·m + m(m−1)/2   for m   (≈ late_swap/280 as a quick est)
planned = swaps − misses ;  caught ≈ D − misses  (D≈9.5 dues/scenario) ;  precision = caught/planned
```
Compare early/late/capacity against the baseline table in §4. Early down + late up =
cut too much volume (loosen). Early up + late flat = over-swapping (tighten).

---

## 7. HARD CONSTRAINTS (do not violate — these are competition rules and hard-won lessons)

- Entry point `script.py`; output `submission.csv`; env `BATTERYSWAP_DATASET_PATH` and
  `BATTERYSWAP_SPLITS` (default `public,private`). All artifacts committed (models are git-LFS).
- 30-minute wall clock for the whole eval, CPU only, no network at submission time.
- Only competition packages (`requirements.txt`). The runtime governor (`bsai/runtime.py`)
  degrades then all-defers to guarantee finishing — keep it.
- **Validate out-of-fold grouped BY BUILDING.** Never learn building identity (a past
  version did and collapsed on public). Within-scenario contrasts are fine.
- Do NOT bake `ENV BATTERYSWAP_SPLITS` into the Dockerfile — the official run passes no
  override, so a baked value yields a train-only submission.
- Keep the `tools/validate_v6.py` planner-fallback abort — a silent all-defer fallback
  reads as a mediocre score instead of a crash.
- Submit sparingly. Each public row is a precious, ~irreversible measurement (it also
  publishes your standing). Decide config from principle + the harness, then submit.

---

## 8. HONEST BOTTOM LINE FOR THE NEXT SESSION

The forecast cannot rank unseen-building dues as well as first place's can — that is the
wall, and no amount of planner tuning moves it (we proved that by making it worse). The
π-hybrid is the only measured step toward a better *ranking* that generalizes. Ship it
through the *simplest* planner that transferred historically (Step 2), calibrate volume
from public (Step 3), and only then push the ranking further (Step 4). Trust the
leave-building-out harness and the public row; distrust local OOF and the train paired
harness for anything claiming to transfer. Aim for a real, measured drop below 2078 —
not a number you cannot back with the transfer harness.
