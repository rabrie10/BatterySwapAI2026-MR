# Final J2W attack: restore V8, rerank, and measure

_Branch `claude/final-j2w-precision`, opened 2026-08-23 from `origin/main`
(`6e5a615`). This file is the canonical experiment log for the branch. Lower
score is better throughout._

The working hypothesis under test, from the public evidence in §1:

> **V8 predicts approximately the right total amount of near-term risk and
> assigns too much of it to the wrong batteries.** So the job is to change
> *which* battery carries V8's high probabilities while leaving the per-scenario
> probability multiset alone.

---

## 0. The commits and artifacts this branch refers to

| generation | commit | branch it lives on | artifact | public |
|---|---|---|---|---:|
| **V8 phase 1 (incumbent)** | `db851212ea9d606457dc69a576d6c590870056c5` | `claude/v8-precision` | `models/v7_wiener.joblib` (`bsai-wiener/v1`) | **2078.28** |
| V9 blend | `c36d4a3949e7484b5f522cd39960d5a9696e577b` | `claude/v9-timing-and-packing` | `models/v9_blend.joblib` (`bsai-blend/v2`) | 2137.22 |
| V19 ship config | `157513e9545e648391ba89c5fc35c9570ca0481d` | `claude/v13-pipeline` | planner-config generation | 2113.43 |
| V13 pipeline op point | `e049a0d91ed38c386c44884c10b4bab21500e3f6` | `claude/v13-pipeline` | — | — |
| V11 / transfer harness + knee | `71bf37f` | `claude/v13-pipeline` | `docs/TRANSFER_STRESS.md`, `docs/KNEE_FINDINGS.md` | — |
| ensemble reranker (Candidate A source) | `8a333d6` | `claude/v13-pipeline` | `docs/ENSEMBLE_FINDINGS.md` | — |
| pi-hybrid ship candidate | `737379aa7361426dc33633088a475fc1ef62bf6a` | `claude/v13-pipeline` | `models/pihybrid.joblib` | never submitted |
| v13 head (isolation run) | `ec998f07c357b295be8590a2a5117e14ca0608b9` | `claude/v13-pipeline` | `tools/public_row.py` | — |
| merge point this branch starts from | `6e5a615771d78e4923aeb8f937a471cf48126a3b` | `main` | — | — |

`models/v7_wiener.joblib` has not been rewritten since `db85121`: the blob is
`800d82150b15988a8d3cbe1fd5dd745a52b412d4` at `db85121`, `de261f5`, `157513e`,
`c36d4a3`, `6e5a615` and `ec998f0` alike. Its `RemainingCalibration` factors are
`[0.4134, 0.6563, 0.7955, 1.0965, 1.7123, 2.3348]`, matching
`outputs/v8_calibration.json` from the V8 fitting run. **This is the artifact
that scored 2078.28**, and `tests/test_submission_identity.py` now pins all of
it — path, class, version, calibration factors and the fact that the file is a
real 2 MB model rather than a Git-LFS pointer.

---

## 1. Known public evidence

| | total | swaps/scen | planned | misses | caught | precision | recall | early | late | other ops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V8 phase 1** | **2078.28** | 21.7 | 19.4 | 2.27 | 7.23 | 0.372 | 0.761 | 946 | 634 | 410 |
| V9 blend | 2137.22 | 22.7 | 20.4 | 2.28 | 7.22 | 0.354 | 0.760 | 1057 | 635 | 362 |
| V19 | 2113.43 | 17.8 | 14.2 | 3.63 | 5.87 | 0.413 | 0.618 | 672 | 1037 | 330 |
| J2W (rank 1) | 1077.72 | 14.5 | — | — | — | — | — | 279 | 592 | 157 |

Decoded with `tools/public_row.py` (recovered from `claude/v13-pipeline`), which
inverts the emergency-queue late-cost formula in closed form.

**The two failure modes are opposite, and both are volume errors.**

* **V9** planned **one more swap per scenario for zero extra catches**: misses
  +0.00, recall 0.761 -> 0.760, early +111. It added risk *mass* (a learned head
  plus a x1.15 level scale) and bought only waste.
* **V19** cut 3.9 swaps per scenario and lost 1.36 catches: early −274 but late
  **+403**. It removed risk mass and starved recall.

So the direction that is left is the one that changes neither: **same mass,
different batteries.**

Against J2W the gap is almost entirely `early_swap` (946 vs 279) while
`late_swap` is close (634 vs 592). First place is not accepting more failures;
it is spending far fewer swaps to catch the same ones.

---

## 2. V8 reproduction

`tools/validate_v6.py`, out-of-fold by building, official `evaluate_plan`, at the
**shipped submission planner config** (`--solver-seconds 0.5 --candidate-margin 12
--local-search 80 --uncertain-search 35`, `move_order=interleaved`), which is what
`script.py` sends to the leaderboard:

```
python tools/validate_v6.py --folds outputs/v7_folds.joblib \
    --model models/v7_wiener.joblib --volatility-scale 1.0 \
    --solver-seconds 0.5 --candidate-margin 12 \
    --local-search 80 --uncertain-search 35 \
    --report outputs/fj_v8_baseline.json --served-out outputs/fj_v8_served.json --audit
```

| | value |
|---|---:|
| **mean total cost** | **2126.53** |
| median / p90 / max | 1957.40 / 3119.65 / 4655.63 |
| early_swap | 764.18 |
| late_swap | 1045.83 |
| battery_swap | 5.36 |
| travel | 40.32 |
| building_change / room_change | 12.85 / 8.23 |
| overtime | 70.59 |
| daily_limit / weekly_limit | 83.33 / 95.83 |
| served per scenario | 17.46 |
| due per scenario | 9.46 |
| missed per scenario | 4.00 |
| useful (caught) per scenario | 5.46 |
| wasted (false-positive) swaps per scenario | 12.00 |
| precision / recall | 0.313 / 0.577 |
| expected due per scenario (model mass) | 9.40 |
| runtime | 9.57 s/scenario, projected 17.6 min for 96 |
| block means (6 non-overlapping) | 1965.8 / 1853.9 / 3103.2 / 2501.1 / 2044.8 / 1290.4 |

The recorded V8 figure is **2145.16** (`outputs/v8_baseline.json` from the V8
session, at `validate_v6`'s own defaults of solver 1.0 s / candidate margin 24).
The −18.6 difference is the planner config, and it sits well inside the ±52
reroll band this project has measured. **The baseline reproduces.** Every
candidate below is run at this same config and reported as Δ against 2126.53.

Model mass 9.40 against a realised 9.46 due per scenario confirms the premise:
the *amount* of risk is right to about 1 %. Precision 0.313 says where it goes
is not.


---

## 3. Transfer-gate retrospective: would the hard building holdout have caught V9 or V19?

`docs/PIHYBRID_HANDOFF.md` ranks the leave-building-out harness second in its
trust hierarchy and calls it "the instrument that CORRECTLY predicted the V19
public failure -- trust it". `docs/ENSEMBLE_FINDINGS.md` awards a GO on its
0.428 bar. Neither claim had been checked against a public row. This branch
checked both.

### V9 -- run for the first time (`tools/fj_v9_retro.py`)

V8's Wiener probability is held fixed at its five-fold out-of-building value and
only V9's gradient-boosted head is refitted with an entire adversarial building
group excluded, then applied to that group. PR-AUC on the held-out rows:

| hard group | rows | due | V8 | V9 head alone | V9 blend |
|---|---:|---:|---:|---:|---:|
| hard_large5 | 13244 | 179 | 0.4416 | 0.4239 | **0.4996** |
| hard_small10 | 1270 | 88 | 0.5267 | 0.5819 | **0.5990** |
| hard_mosteol5 | 6800 | 250 | 0.3165 | 0.4755 | **0.4971** |
| hard_hirate6 | 1102 | 159 | **0.3688** | 0.3740 | 0.3618 |
| hard_betashift5 | 4223 | 40 | **0.5884** | 0.5581 | 0.5830 |
| **mean** | | | **0.4484** | 0.4747 | **0.5081** |

**The gate gives V9 a clean PASS**: mean 0.5081 against its own 0.428 bar, +0.06
over V8, winning three of five groups and never collapsing. V9 then scored
**2137.22 against V8's 2078.28**.

### V19 -- the model half, from the harness's own record

V19 shipped `models/v8_cens.joblib` behind a hard volume cap (`157513e`). The
harness's verdict on that model is in `docs/TRANSFER_STRESS.md`: *"PR-AUC on 5
hard holdouts: cens mean 0.428 vs v7 0.391 (cens better on 5/5)"*. The public
row for the cens forecast is in `docs/V11_TRANSFER_FINDINGS.md`: V10 scored
**2179.06 against V8's 2078.28**, decomposed as **+179 on early+late** for the
forecast and −111 for planner mechanics.

So the harness preferred, on all five groups, the model whose only public A/B
was 101 points worse. The V11 write-up rescues this by arguing the +179 was
*level*, not *ordering* -- which is a testable claim, and this branch tested it.
Candidate A below deploys the cens **ordering** with V8's levels and mass, so
the level explanation is removed by construction. Through the real planner it
costs **+61.3**. The ordering was not better either.

The harness also cannot address V19's other half at all: a selection-layer
change has no model to score.

### Verdict

```
TRANSFER_STRESS did not reliably identify known public failures and cannot be
treated as an oracle.
```

It passed V9 with a +0.06 margin, preferred the cens model 5/5 against the
incumbent that beat it publicly by 101, and is structurally blind to the
planner-config half of V19. Its per-fold numbers are still worth reporting --
they are the only fresh-building proxy available -- but a PASS is not evidence.
Gate 4 in the plan for this branch is therefore weighted as **weak evidence**,
and the decision below rests on the official-cost path.

---

## 4. Candidate A -- the recovered v13 order-only reranker

### What was recoverable

`docs/ENSEMBLE_FINDINGS.md`'s winner is `remkeyed_puremid`: a regime-keyed logit
mixture of three models -- `p_cens` (censored-drift Wiener), `p_tp` (two-phase
changepoint) and `p_qh` (quantile head) -- deployed as an order-only,
sum-of-p-preserving remap. Recovering it exactly needs three artifacts:

| component | state |
|---|---|
| `p_cens` | **recovered**: `outputs/v8_cens_folds.joblib` survives, loads against this branch's `bsai`, dispatches by building, 64 features |
| `p_tp` | **gone**: `outputs/twophase_model_oof.joblib` exists nowhere on disk; rebuilding needs `tools/twophase_fit.py`, which needs `outputs/twophase_series.joblib` **and** `outputs/research_rowfeat.parquet`, neither of which survives |
| `p_qh` | **gone**: `outputs/qhead_folds.joblib` and `outputs/qhead_model.joblib` exist nowhere on disk |

This matches the v13 session's own finding when it tried the same recovery
(`docs/PIHYBRID_ISOLATION.md`): *"none of which survive in this worktree;
regenerating them is a multi-hour reproduction of the whole model-fitting
chain, not a gate."*

Two further facts about the exact candidate are worth recording before spending
that day:

* its own branch already ran the reorder and recorded a **local NO-GO**. From
  `157513e`'s commit message: *"ensemble reorder NO-GO locally (+97.2/scen
  pure-ordering vs the cens control)"*. The GO in `ENSEMBLE_FINDINGS.md` was a
  pre-registration, not an end-to-end result;
* the mixture is dominated by `p_tp` (weight 0.5 in the open regime, **1.0** in
  the mid regime) with `p_cens` and `p_qh` at 0.125-0.25 -- and `p_cens`'s only
  public A/B is +101.

**Candidate A as run here is therefore the recoverable component deployed the
way the ensemble specified**: V8's own curves, handed out in the `cens` model's
order, mass preserved exactly. It is the strongest available test of the
ensemble's central transfer claim, because it removes the level entirely.

### Result

Through the real `CompetitionPlanner` and the official `evaluate_plan`, 48 train
scenarios, out of fold by building, shipped planner config:

| | V8 | Candidate A | delta |
|---|---:|---:|---:|
| **mean total cost** | **2126.53** | **2187.81** | **+61.28** |
| paired t / wins | — | t = +0.83, 26W/22L | inside noise, wrong side |
| early_swap | 764.18 | 774.62 | +10.4 |
| late_swap | 1045.83 | 1095.21 | +49.4 |
| served / scenario | 17.46 | 17.60 | +0.14 |
| useful swaps | 5.46 | 5.40 | -0.06 |
| wasted swaps | 12.00 | 12.20 | +0.20 |
| misses | 4.00 | 4.06 | +0.06 |
| precision | 0.313 | 0.307 | -0.006 |
| recall | 0.577 | 0.570 | -0.007 |
| overtime / daily / weekly | 70.6 / 83.3 / 95.8 | 72.0 / 77.1 / 102.1 | +1.4 / -6.2 / +6.3 |
| temporal blocks improved | — | 3 of 6 | — |
| hard transfer groups improved | — | 3 of 5, worst +72.0 | — |
| runtime | 9.57 s/scen | 9.87 s/scen | +0.3 |

The mass-preservation invariant was verified per scenario before the run:
`np.allclose(np.sort(candidate_p), np.sort(v8_p))` and
`np.isclose(candidate_p.sum(), v8_p.sum())` hold exactly, and the invariant is
checked at the CDF-curve level as well -- every output curve is one of V8's
input curves, unbroken (`tests/test_rerank.py`, 12 tests).

**Candidate A is rejected.** It reproduces its own branch's local verdict, and
it falsifies the claim that the cens model's hard-holdout PR-AUC advantage is an
ordering advantage.


---

## 5. Error analysis and the residual hypotheses

The full write-up is `docs/FINAL_FP_ANALYSIS.md`. The three results that decide
this branch:

**1. Inside the population where swaps are spent, V8's ordering is barely
better than one subtraction.** Within-scenario concordance among the top 25
candidates per scenario: V8's probability **0.6152**, `voltage_compensated`
**0.6408**, the censored-drift model 0.6387, raw margin 0.6324. Chance is 0.5.

**2. The waste is six devices.** 83 devices carry all 576 wasted swaps and the
worst ten carry half. Six sit within 0.03 V of the barrier, are swapped in 29 to
48 of the 48 scenarios, and never die. The first-passage law cannot express
them: as the margin goes to zero the crossing probability goes to one whatever
the drift regressor says.

**3. Nothing learnable is left at that boundary.** A 200-iteration
`HistGradientBoostingClassifier` on 72 within-scenario-ranked signals -- every
shipped feature plus CDF shape, peer contrast, dwell and staleness -- fitted out
of fold by V8's own building folds, reaches within-scenario concordance
**0.5567**. That is *worse* than V8's 0.6152 and far worse than one subtraction.
An L2 logistic on the same design reaches 0.5569. With 454 positives from about
82 distinct devices and 85 % scenario overlap, the boundary is not learnable by
a flexible model, which is the same wall the V6 51-feature model, the stacking
probe, the V9 head and the pi-hybrid all hit.

### The residual hypotheses, one line each

| hypothesis | measured | verdict |
|---|---|---|
| **cold room / peer contrast** | TP sits 0.252 V below its roommates' median, FP 0.128 V -- the direction the hypothesis predicts. Within scenario it is worth 0.5989 concordance against V8's 0.6152, and reranking on it costs +52 analytic. | rejected, fourth independent construction |
| **staleness** | median staleness is 0 in the TP, FP and miss populations alike. It is a gating question, not a ranking axis, and a gate is a volume change. | rejected |
| **dwell / persistence** | Among margin < 0.02 rows the realised 42-day death rate falls 0.724 / 0.233 / 0.169 across dwell bands, reproducing the v13 gate's 0.80 / 0.29 / 0.18. **Collapsed to one row per device it is 0.789 / 0.786 / 0.500 / 0.800** -- the row-level separation is length bias, the 42-84 d band is 59 rows from **7 devices**, and the longest band reverses. `docs/V11_TRANSFER_FINDINGS.md` already measured a dwell knockdown at +93 through the planner, and V19 shipped a demotion rule built on this and lost 403 points of late cost. | rejected |
| **season** | `season_sin` reaches AUC 0.655 inside the candidate band and is nearly constant within a scenario, so it cannot reorder anything. It moves volume between scenarios, which `bsai/calibrate.py` already corrects. | rejected as a ranking axis |
| **CDF shape** | `p07/p42` separates TP from FP at AUC 0.365 and within scenario is worth 0.5957. On top of the compensated blend it adds nothing (1668.1 against 1609.2 analytic). | rejected |
| **Wiener/head disagreement** | Population B in the FP analysis shows the head's local reordering is a training-building property. Using its disagreement imports exactly what failed. | not built |
| **temperature-compensated barrier** | The one that survived screening. Sweeping the compensation constant, within-scenario concordance peaks at **0.00463 V/degC** -- the value `bsai/features.py` measured independently -- at 0.6408, against 0.6324 at zero compensation. A per-device constant from `beta_30` is *worse* (0.6082): the within-day dV/dT carries HVAC duty cycle, not just cell sensitivity. | promoted to Candidate B |

---

## 6. Candidate B -- the temperature-compensated barrier reranker

**Construction.** Rank the scenario's batteries by V8's own decision probability;
rank them again by `-voltage_compensated`; add the two centred percentile ranks;
hand V8's curves out in that order. **No fitted parameter, no learned
probability, no building term, no new data** -- `voltage_compensated` is already
one of V8's 64 inputs, and the temperature constant is the one already measured
in the repository. The equal weight is the assumption-free combination of two
orderings of comparable quality, and the result is flat across weights 0.5 to
2.0 and across candidate-pool sizes 25 to "all", so nothing was tuned.

**On the cached frame it looks like the best thing this project has measured
since V9.** Precision and recall both improve at every operating point, six of
six temporal blocks improve, 35 of 48 scenarios win (mean -129.0, sem 38.7,
t = -3.33), and four of five adversarial building groups improve.

**Through the real planner it is worth nothing.**

| | V8 | Candidate B | delta |
|---|---:|---:|---:|
| **mean total cost** | **2126.53** | **2149.74** | **+23.21** |
| paired t / wins | — | t = +0.49, 26W/22L | inside noise, wrong side |
| early_swap | 764.18 | 763.60 | -0.6 |
| late_swap | 1045.83 | 1051.25 | +5.4 |
| served / scenario | 17.46 | 17.38 | -0.08 |
| useful swaps | 5.46 | 5.46 | 0.00 |
| wasted swaps | 12.00 | 11.92 | -0.08 |
| misses | 4.00 | 4.00 | 0.00 |
| precision | 0.313 | 0.314 | +0.001 |
| recall | 0.577 | 0.577 | 0.000 |
| overtime / daily / weekly | 70.6 / 83.3 / 95.8 | 74.6 / 87.5 / 104.2 | +4.0 / +4.2 / +8.4 |
| temporal blocks improved | — | 3 of 6 | — |
| hard transfer groups improved | — | 4 of 5, worst +42.0 | — |
| runtime | 9.57 s/scen | 9.41 s/scen | -0.2 |

### Why the screen and the planner disagree, measured

The remap did change the plan: of 838 planned swaps, **105 were dropped and 101
added**. The exchange was exactly neutral -- **16 dues out, 16 dues in**.

The screen scores top-k by probability. The planner does not select that way:
**16.5 % of its swaps sit outside the top-17 by level**, and it skips a
comparable number inside it, because it is running an expected-cost optimisation
over routing, day-packing and capacity rather than a threshold. Measured on the
same rows: dues inside the top 17 by level go from **270 (V8) to 281 (Candidate
B)** -- the ranking gain is real and the screen reported it correctly -- and
dues inside the planner's served set go from **262 to 262**. The +11 lands in
the part of the ordering the planner's economics decline, and the 105/101
exchange happens between rank 15 and rank 45, where the true rate is about 16 %
and nothing measured here separates.

**This is a methodological finding worth more than the candidate.**
`tools/rank_lab.py`'s top-k pricing was validated against the planner for
*level* changes. For a *permutation* of the levels it over-reports by more than
100 points. Any future reranking work has to be screened on the planner or on a
paired harness, never on top-k.

### The control that proves the deployment path is sound

Before concluding that order-only remapping cannot pay, the same code path was
run with the answer as the score -- V8's own curves, V8's own mass, handed out
in perfect order (`tools/validate_v6.py --rerank-oracle`, a diagnostic, not
shippable). On the first six scenarios:

| | V8 | oracle order | delta |
|---|---:|---:|---:|
| mean total cost | 1984.57 | **882.61** | **-1101.96** |
| served / scenario | 12.33 | 12.33 | 0.00 |
| misses / scenario | 5.33 | 2.67 | -2.66 |
| precision | — | 0.946 | — |
| recall | — | 0.814 | — |

**Same probability multiset, same swap count, minus 1102.** The order-only
mechanism works, the planner does convert ordering into cost, and the mission's
central hypothesis is confirmed on the official cost path. What is missing is
not the deployment path. It is a better ranker.

### How much better

Corrupting the oracle with noise and re-measuring gives the exchange rate
between within-scenario concordance and precision at the planner's operating
point (top 25 candidates per scenario, analytic pricing):

| within-scenario concordance | precision at 12 swaps | precision at 17 |
|---:|---:|---:|
| 0.6164 (**V8 today**) | 0.363 | 0.331 |
| 0.6427 (**best signal found: +0.024**) | ~0.37 | ~0.34 |
| 0.825 | 0.439 | 0.355 |
| 0.922 | 0.559 | 0.449 |
| 1.000 | 0.693 | 0.538 |

Decoding J2W's public row with `tools/public_row.py`'s arithmetic gives **12.4
planned swaps, 2.13 misses, precision 0.596, recall 0.776**. On this table that
is concordance in the neighbourhood of **0.92**. V8 is at 0.616 and every signal
measured in this branch moves it by at most 0.03. **The gap to first place is
forecasting power about which device dies next -- not tuning, not the planner,
and not the deployment path.**

---

## 7. Final league table

All rows are the real `CompetitionPlanner` and the official `evaluate_plan` over
all 48 train scenarios, out of fold by building, at the shipped submission
config (`--solver-seconds 0.5 --candidate-margin 12 --local-search 80
--uncertain-search 35`). Transfer wins are the five adversarial building groups
on the analytic screen; the oracle row is a diagnostic on scenarios 0-5 only.

| Candidate | Total | Δ vs V8 | Early | Late | Swaps | Useful | FP | Misses | Precision | Recall | Transfer wins | Worst transfer Δ | Temporal wins | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V8 (incumbent)** | **2126.53** | — | 764.18 | 1045.83 | 21.46 | 5.46 | 12.00 | 4.00 | 0.313 | 0.577 | — | — | — | 9.57 s |
| Candidate A (cens order, mass-preserved) | 2187.81 | **+61.28** | 774.62 | 1095.21 | 21.66 | 5.40 | 12.20 | 4.06 | 0.307 | 0.570 | 3/5 | +72.0 | 3/6 | 9.87 s |
| Candidate B (compensated barrier) | 2149.74 | **+23.21** | 763.60 | 1051.25 | 21.38 | 5.46 | 11.92 | 4.00 | 0.314 | 0.577 | 4/5 | +42.0 | 3/6 | 9.41 s |
| _oracle order (diagnostic, s_0-s_5)_ | _882.61_ | _-1101.96_ | — | — | _15.00_ | _11.67_ | _0.67_ | _2.67_ | _0.946_ | _0.814_ | — | — | — | — |

Operational components, per scenario:

| Candidate | overtime | daily limit | weekly limit | travel | building change | room change |
|---|---:|---:|---:|---:|---:|---:|
| V8 | 70.59 | 83.33 | 95.83 | 40.32 | 12.85 | 8.23 |
| Candidate A | 71.98 | 77.08 | 102.08 | 39.97 | 13.02 | 8.47 |
| Candidate B | 74.60 | 87.50 | 104.17 | 41.81 | 13.10 | 8.36 |

Neither candidate clears **Gate 2** (a 100-point improvement); both are on the
wrong side of zero and both sit inside the paired noise band (t = +0.83 and
+0.49 on 48 paired scenarios). Neither clears **Gate 3**: Candidate B's failure
signature is that *nothing moves at all* -- precision +0.001, recall 0.000,
misses 0.00 -- and Candidate A's is a small V19 (late +49, recall -0.007).

The operational columns behave exactly as the plan predicted they would: they
are downstream of swap volume, and since every arm holds volume fixed by
construction, they move by a few points in whichever direction the day-packing
happens to land. No conclusion about Task 2 should be drawn from them.

---

## 8. Final recommendation

**No candidate robustly beats V8, and none is enabled.** `script.py` loads
`models/v7_wiener.joblib`, the artifact that scored 2078.28, and
`tests/test_submission_identity.py` pins its path, class, version, calibration
factors and the fact that it is a real model rather than a Git-LFS pointer. The
reranking machinery is committed with its tests and its measurements because the
diagnosis it produced is this branch's real output, but the default path does not
touch it: `bsai/rerank.py` is imported by `tools/validate_v6.py` only, never by
`script.py` or `bsai/forecaster.py`.

What this branch is worth merging for:

1. **The V9 default is removed.** `main` would have shipped
   `models/v9_blend.joblib`, publicly 2137.22 against V8's 2078.28 -- a 59-point
   loss for free, and no test would have caught it.
2. **The transfer gate is demoted from oracle to weak evidence**, with the
   measurement that demotes it.
3. **The ordering ceiling is now measured on the official cost path**, not
   analytically: same mass, perfect order, -1102 on the scenarios tested. Every
   future generation can be judged against that number.
4. **The frontier is quantified**: concordance 0.616 today, about 0.92 needed for
   J2W's precision, at most +0.03 available from anything measured here.
5. **The screen that produced three generations of over-promising local numbers
   is now known to over-report a permutation by more than 100 points.**

### What the next generation should and should not do

Do not spend another cycle on: peer contrast (five constructions now), dwell or
persistence rules (length bias, seven devices, and V19's public row), season as
a ranking axis (constant within a scenario), CDF-shape reweighting, another
learned probability head (0.5567 concordance against V8's 0.6152), or the
planner (four independent attacks, all at or below noise, plus the oracle control
above showing the planner converts ordering perfectly well when it is given one).

The only direction with a live measurement behind it is a **better estimate of
the barrier distance itself**. Compensated voltage beats the whole first-passage
model at within-scenario ranking, and the compensation optimum sits exactly on
the independently measured 0.00463 V/degC. That says the barrier distance is the
binding quantity and it is currently measured with error. `docs/V11_TRANSFER_FINDINGS.md`
reached the same spec from a different direction ("estimate the V/degC
compensation per device instead of the global 0.00463"), and the naive per-device
version tested here is *worse* because `beta_30` carries HVAC duty cycle -- which
makes a clean per-device V/degC estimate, from the raw hourly channel rather than
from the within-day shape features, the one specific unexplored thing this
analysis points at.

---

## 9. Appendix: the public rows, decoded on this branch

`tools/public_row.py` is recovered from `claude/v13-pipeline` (`cfd8389`) so the
only ground-truth instrument this project has lives on the branch that cites it.
It inverts the emergency-queue late-cost curve in closed form and self-tests
against both recorded rows. All figures per scenario:

| row | swaps | misses | planned | caught | precision | recall | early per planned swap |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 2078.28 | 21.68 | 2.27 | 19.41 | 7.23 | 0.372 | 0.761 | 48.7 |
| V9 2137.22 | 22.69 | 2.28 | 20.41 | 7.22 | 0.354 | 0.760 | 51.8 |
| V19 2113.43 | 17.80 | 3.63 | 14.17 | 5.87 | 0.414 | 0.618 | 47.4 |
| **J2W 1077.72** | **14.48** | **2.13** | **12.35** | **7.37** | **0.597** | **0.776** | **22.6** |

J2W catches *marginally more* than V8 (7.37 against 7.23) with **seven fewer
planned swaps**. That is the whole gap, and it is precision.

One term does not close on precision alone. Solving the two rows jointly for a
shared cost per due swap and per wasted swap has no non-negative solution: at a
due swap's ~2.5, our wasted swap prices at 76.1 and J2W's at 52.4. Two things
can produce that, and they are not distinguishable from the published columns:
J2W places wasted swaps later in the 42-day window, or its wasted swaps land
disproportionately on batteries whose substitute end of life is near (the
closing scenarios), which is a selection effect rather than a timing one.

The timing branch is already measured and closed. `docs/HANDOVER.md` section 4:
an oracle-free swap-day policy has a naive headroom of 333 per scenario, and
fitted on three scenario blocks and scored on the other three it is **56** --
because moving a swap late costs 10 per day on the ones that really were due,
against 0.5 per day saved on the ones that were not, and at precision 0.31 that
trade is close to break-even. Day 1 is close to the asymmetric-loss optimum.
Nothing here reopens it.

---

## 10. Clean-checkout verification

A fresh detached worktree was cut from this branch's HEAD, given only a junction
to the dataset, and run through the official entry point with nothing else set:

```bash
git worktree add --detach <tmp> claude/final-j2w-precision
BATTERYSWAP_DATASET_PATH=dataset BATTERYSWAP_SPLITS=train python script.py
python tools/fj_check_submission.py --dataset dataset --split train
```

| check | result |
|---|---|
| artifact is a real file, not a Git-LFS pointer | **ok** — 2.03 MB, first bytes `\x80\x04\x95…bsai.wiener.WienerModel`; no `git lfs pull` needed |
| artifact loads through the submission's own loader | **ok** — `joblib.load` returns `bsai.wiener.WienerModel` |
| identity logged at INFO | **ok** — `version=bsai-wiener/v1 volatility_scale=1.0 calibration=[0.4134, 0.6563, 0.7955, 1.0965, 1.7123, 2.3348]` |
| `submission.csv` produced | **ok** — 751,349 bytes, 19,890 rows, columns `day, battery, split, scenario` |
| planner fallbacks | **none** — `planned=48 degraded=0 deferred=0` |
| NaN / Inf anywhere | **none** |
| dates parse, none before a scenario start | **ok** — 2025-09-02 .. 2026-09-08 |
| every active battery exactly once, per scenario | **ok** — 48 scenarios, no missing, no duplicates, no extras |
| runtime | **462 s for 48 scenarios** = 9.6 s/scenario |
| network | none required; the dataset is local and no model is fetched |
| GPU | none required |

**Runtime projection for the official run.** 96 scenarios plus a second dataset
load projects to roughly **16.6 minutes** against the 30-minute limit. That is
just under `bsai/runtime.py`'s 17-minute soft deadline, so the governor may
switch the last scenarios to the cheap search; that is the shipped V8 behaviour,
unchanged by this branch, and the 25-minute hard deadline is far away.

The CDF contract itself is not visible in `submission.csv` -- the file is a plan,
not a forecast -- so monotonicity in the horizon and the sum-to-one branch split
are covered where they live, in `tests/test_task1_forecast.py` and
`tests/test_rerank.py`. The whole suite is **85 tests, all passing**.
