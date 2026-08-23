# The sequence model: the state was the bottleneck, and it is not V8's fault

_Branch `claude/battery-device-segmentation-9cd2dd`. Lower cost is better, higher
concordance is better. Every number here is out of fold by building._

Seven engineered representations have now improved ordering at matched margin and
failed to improve it across margins — volatility ratio, trajectory templates,
survival frailty, three decision-focused losses, a 72-signal GBDT, analog cohorts,
and the cohort-4 observation score of `docs/FINAL_SEGMENT_EXPERIMENT.md` §10. Each
described the *state at the cutoff*, and V8's margin already described the state
better.

This one is not given the state. It reads 120 days of raw daily trajectory and
predicts the future trajectory, which is the supervision that made the Wiener law
work in the first place: **106,612 causal windows rather than 82 failure events**.

```
GATE 1 PASSED. GATE 2 PASSED. Order-only planner runs completed on both planners.
VERDICT: TCN REPRESENTATION PASSES.
```

---

## 1. What was built

| | |
|---|---|
| inputs | 5 channels × 120 days: margin, anchor-relative shape, temperature, observation mask, staleness |
| architecture | `Conv1d(5→24,k=1)` stem; 6 residual blocks, `k=3`, dilations 1/2/4/8/16/32, GroupNorm + GELU; last timestep → `Linear(24→32)` → GELU → `Linear(32→35)` |
| receptive field | 127 days |
| **parameters** | **12,899 per fold** |
| outputs | 7 quantiles (0.05…0.95) of the voltage *change* at 7, 14, 21, 28, 42 days |
| loss | pinball, masked per (window, horizon) so censoring is never imputed |
| training | Adam 3e-3, cosine schedule, batch 512, 8 epochs, 5 building-disjoint folds |
| windows | **106,612** from 456 devices, stride 3 |
| runtime | **2,398 s** (40 min) for all five folds, 16 CPU threads |

The 64 engineered features are deliberately absent. So is `v_now` as a separate
input — the network sees the trajectory that produced it.

### The invariant that had to be fixed first

The corpus originally anchored windows anywhere in a device's series, and a
series continues a **median 86 days past its barrier crossing**. That put 2,869 of
109,481 windows — 13.1 % of all windows belonging to the 82 EOL devices — at
origins at or after end of life.

The share understates it:

| | origins at/after EOL | anchor below 2.4 V | min anchor voltage |
|---|---:|---:|---:|
| training windows, unconstrained | 2,869 | **75.5 % of them** | 2.037 V |
| training windows, pre-EOL only | — | **0.0 %** | 2.400 V |
| **the 19,890 competition rows** | **0** | **0 %** | margin exactly 0.0000 |

A competition scenario only ever asks about an active battery, so those windows
were a voltage regime inference never visits, immediately adjacent to the decision
boundary, on a 12,899-parameter budget. `Corpus` now requires the anchor strictly
before the device's crossing and `train()` asserts it. **Targets are untouched**:
a window anchored before EOL whose 42-day horizon runs through the crossing is
the terminal decline the model exists to learn, and 887 windows carry one.
`tests/test_tcn.py::OriginPrecedesEolTest` pins both halves. The model reported
here was trained after the fix; nothing from the contaminated run is reported.

## 2. Gate 1 — forecast quality, against V8's own machinery

Both models answer the same question at the same 19,638 scenario cutoffs (98.7 %
of rows; the rest have under 120 days of history and fall back to V8). V8 answers
with the drift and scatter regressors the first-passage law is built on.
Device-weighted, held-out buildings.

| horizon | n | persistence MAE | V8 MAE | **TCN MAE** | V8 medAE | **TCN medAE** | V8 pinball | **TCN pinball** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 d | 15,203 | 0.01853 | 0.01813 | **0.01661** | 0.00983 | **0.00752** | 0.00527 | **0.00453** |
| 14 d | 14,844 | 0.02831 | 0.02477 | **0.02353** | 0.01314 | **0.01105** | 0.00704 | **0.00645** |
| 21 d | 14,480 | 0.03735 | 0.03102 | **0.02997** | 0.01650 | **0.01426** | 0.00871 | **0.00802** |
| 28 d | 14,118 | 0.04616 | 0.03687 | **0.03588** | 0.01977 | **0.01689** | 0.01035 | **0.00952** |
| 42 d | 13,374 | 0.06048 | 0.04652 | **0.04537** | 0.02794 | **0.02283** | 0.01351 | **0.01228** |

Better on every horizon and every metric. The median error improves by 18–23 %,
much more than the mean — the TCN is markedly better on the typical device and
roughly tied on the outliers.

**By margin band at 42 days**, which is the question the gate was written for:

| band | n | devices | persistence | V8 | **TCN** | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 0–0.03 V | 81 | 43 | 0.05274 | 0.04535 | **0.04336** | −0.00198 |
| **0.03–0.05 V** | 121 | 50 | 0.05697 | 0.05161 | **0.04364** | **−0.00797 (−15.4 %)** |
| **0.05–0.10 V** | 447 | 108 | 0.06841 | 0.05686 | **0.04806** | **−0.00880 (−15.5 %)** |
| 0.10–0.20 V | 1,414 | 179 | 0.07989 | 0.06285 | **0.06112** | −0.00173 |
| > 0.20 V | 11,311 | 422 | 0.06304 | 0.04488 | 0.04492 | +0.00004 |

The gain is concentrated exactly in the planner-relevant band and is a rounding
error where nothing is decided.

## 3. Gate 2 — cross-margin ordering

Same landmark population and metric as every other candidate on this branch: top
40 rows per scenario by V8 probability among rows whose 42-day fate is observed,
1,708 rows, 9,810 cross-margin pairs at a 0.01 V bin. The forecast is turned into
a crossing score and used as a **rank**; absolute levels are not trusted.

| | overall | **cross-margin** | same-margin | folds won |
|---|---:|---:|---:|---:|
| **V8** | 0.7280 | **0.7359** | 0.5846 | — |
| quantile p42, standalone | 0.7662 | 0.7704 | **0.6896** | 2/5 |
| z-score at 42 d | 0.7649 | 0.7696 | 0.6784 | 2/5 |
| z-score, first passage over all horizons | 0.7648 | 0.7700 | 0.6691 | 2/5 |
| rank blend with V8, w = 0.25 | 0.7601 | 0.7662 | 0.6487 | 5/5 |
| rank blend, w = 0.5 | 0.7691 | 0.7752 | 0.6589 | 5/5 |
| **rank blend, w = 1.0** | **0.7746** | **0.7802** | 0.6729 | **5/5** |

**+0.0443 cross-margin, device bootstrap +0.0446 [+0.0221, +0.0729], P(Δ>0) = 1.00.**
Past the 0.75 bar the gate was written against, and twenty times the general
segmentation result.

**Same-margin moves up as well**, 0.5846 → 0.6729. Every previous representation
on this project traded the two conditionings against each other. This is the
first to improve both.

## 4. The controls that killed the previous seven

An aggregate is not evidence here — §10 of `docs/FINAL_SEGMENT_EXPERIMENT.md`
produced +0.0347 with an interval excluding zero and 66.8 % reversal accuracy,
and one battery was 57 % of it.

**Routing, asserted rather than assumed.** The TCN's fold index comes from
`fold_of_device` and V8's from `v8_folds`; both derive from the same bundle by
different paths, and one disagreeing building would mean a model scoring devices
it trained on. **0 of 458 devices disagree**; the five folds hold 117 / 72 / 106 /
91 / 72 devices and are pairwise disjoint.

**Every building fold improves.**

| fold | V8 cross-margin | blend | Δ |
|---|---:|---:|---:|
| 0 | 0.9600 | 0.9840 | +0.0240 |
| 1 | 0.7955 | 0.8182 | +0.0227 |
| 2 | 0.6236 | 0.6854 | **+0.0618** |
| 3 | 0.7811 | 0.7965 | +0.0154 |
| 4 | 0.7039 | 0.7279 | +0.0240 |

**The cumulative device jackknife**, the instrument that took the cohort-4 result
negative by its fifth device:

| devices removed | cross-margin Δ | pairs left |
|---|---:|---:|
| — | **+0.0443** | 9,810 |
| top 1 | +0.0383 | 9,621 |
| top 2 | +0.0324 | 9,418 |
| top 3 | +0.0283 | 9,295 |
| top 5 | **+0.0222** | 8,996 |
| top 8 | +0.0157 | 8,606 |
| top 12 | **+0.0100** | 8,070 |

1,110 moved pairs over **73 due devices**, 47 helped and 23 hurt, top 5 carrying
52.5 % of the net rather than cohort-4's 104 %.

**The reversal test.** It overrides V8 on **10.3 % of cross-margin pairs** — not
the 0.7 % a heavily shrunk correction manages — and is right on **71.5 %** of
them. The general segment model was right on 47.3 %.

## 5. The planner, order-only

`bsai/rerank.SequenceScorer` looks the score up by `(device, remaining)`, which is
unique across all 19,890 rows and integral in days. It is *not* reconstructed from
`end_time − remaining`, which agrees with the authoritative scenario start on only
17,801 of them. `RankRemapModel` hands V8's own per-scenario CDF multiset out in
the blended order, so risk mass is unchanged by construction; 95 % of rows sit in
a single equal-`remaining` group of about 394, so the remap has nearly full
freedom. `tools/fj_compare.py` reproduces the committed deterministic-emergency
result exactly (−35.30, t −3.02, 31 W / 10 L) before being pointed at anything new.

| | V8 | **TCN order-only** | Δ | paired t | W/L | median Δ | 10 % trimmed | sign test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **shipped planner** (robust 4, 80/35) | 2126.53 | **2055.58** | **−70.96** | −1.23 | 31/17 | **−91.16** | −91.18 | p = 0.059 |
| _same, folds **averaged** as the container runs them_ | 2126.53 | **1967.10** | **−159.43** | **−3.21** | 31/17 | −105.22 | −161.27 | p = 0.0595 |
| deterministic emergency (robust 0) | 2091.23 | **2038.94** | **−52.29** | −0.94 | 27/21 | −59.51 | −62.57 | p = 0.47 |

The median and trimmed mean are both *stronger* than the mean on both planners:
the distribution is right-skewed by a few expensive scenarios, and the typical
scenario improves by about 91 points on the shipped planner.

**The decision signature is the one that was asked for, and better** — on the
shipped planner:

| | V8 | TCN | Δ |
|---|---:|---:|---:|
| early_swap | 764.18 | 743.48 | **−20.70** |
| late_swap | 1045.83 | 984.79 | **−61.04** |
| served / scenario | 17.458 | 17.396 | −0.062 |
| hits / scenario | 5.458 | 5.729 | **+0.271** |
| missed / scenario | 4.000 | 3.729 | **−0.271** |
| wasted swaps / scenario | 12.000 | 11.667 | −0.333 |
| **precision** | 0.313 | **0.329** | **+0.017** |
| **recall** | 0.577 | **0.606** | **+0.029** |

Early *and* late both fall, with slightly fewer swaps. Operational costs move
against it slightly (weekly_limit +10.42, overtime +2.36).

### Runtime

| configuration | s/scenario | projected for 96 | governor (soft 25, hard 27.5) |
|---|---:|---:|---|
| V8 shipped | 9.57 | 17.6 min | OK |
| V8 deterministic | 12.23 | 21.8 min | OK |
| **TCN + shipped planner** | 12.40 | **22.1 min** | **OK** |
| TCN + deterministic | 16.48 | **28.6 min** | **past the hard deadline** |

**The deterministic-emergency combination is not shippable as configured.** The
scorer itself is a dictionary lookup; the cost is that reordering changes which
batteries the search explores.

## 6. Where the gain is, and where it is not

The planner deltas split cleanly by the scenario's mean remaining-observation
window, and **the concordance metric predicts the split exactly**:

| scenario tercile | pairs | V8 cross-margin | blend | Δ concordance | **planner Δ** | wins |
|---|---:|---:|---:|---:|---:|---:|
| low remaining (49 d) | 1,800 | 0.7183 | 0.8100 | **+0.0917** | **−122.0** | 12/16 |
| mid remaining (161 d) | 2,925 | 0.6427 | 0.7352 | **+0.0925** | **−314.8** | 13/16 |
| **high remaining (274 d)** | 5,085 | **0.7957** | 0.7955 | **−0.0002** | **+223.9** | 6/16 |
| all | 9,810 | 0.7359 | 0.7802 | +0.0443 | −71.0 | 31/48 |

Two thirds of the scenarios get **+0.09 of cross-margin concordance** and give
back 120–315 points of cost. (Under the fold-averaged ensemble the container
actually runs, the third tercile stops being neutral: concordance +0.0231 and
cost +2.5. See `docs/SUBMISSION_TCN.md` §8.) In the last third V8 is already at its best (0.7957)
and the blend is exactly neutral — and a *concordance-neutral* reorder still
costs +224, because those are the opening scenarios where the substitute end of
life is far away and a wasted swap is worth roughly 182. Precision there moves
0.577 → 0.558 and three catches are lost.

**This is the single most actionable number in the study.** The blend is spending
half the ordering influence in the one regime where the incumbent has nothing to
learn from it. Gating the blend weight on the remaining-observation axis is the
obvious next move and is *not* made here: it is a fitted decision that needs its
own nested validation, and this branch has already watched an unvalidated
threshold turn into one battery.

## 7. Verdict

```
TCN REPRESENTATION PASSES -- candidate worth planner/public validation.
```

Both pre-registered gates passed by margins that are not noise-scale: forecast
error down on every horizon and 15 % down in the planner band; cross-margin
concordance 0.7359 → 0.7802 with a bootstrap interval far from zero, all five
building folds improving, survival of a twelve-device cumulative jackknife, and
71.5 % accuracy on the 10.3 % of pairs where it overrules V8. Same-margin
concordance rises at the same time, which nothing else on this project has done.

**Stated plainly, three caveats.** The planner delta is favourable in mean, median
and trimmed mean on both planners with the desired signature, but it is **not
statistically significant** (t −1.23 shipped, sign test p = 0.059; t −0.94
deterministic). It is **not uniform** — one tercile of scenarios costs +224. And
**TCN + deterministic planner is over the hard runtime deadline** at 28.6 min for
96; only TCN + shipped planner (22.1 min) is deployable as measured.

The representation hypothesis this branch opened with is answered. The
hand-engineered state *had* discarded temporal information that matters for
cross-margin ordering, and a 12,899-parameter model trained on the window
population recovers it. What remains before this is worth a public submission is
the remaining-axis gating in §6, and that is a new experiment with its own gate.
