# Analog cohorts: the regimes are real, and they do not order across margins

_Branch `claude/battery-device-segmentation-9cd2dd`, opened 2026-08-23 from
`origin/main` (`d38afe6`) with `claude/final-j2w-precision` (`5fe2dca`) merged in
at `76b0b59` for its instruments. Lower cost is better; higher concordance is
better throughout._

The hypothesis under test, from `docs/FINAL_TERMINALITY.md` section 8:

> There may be latent battery/device/environment regimes such that the **same**
> current voltage margin implies a **different** near-term failure risk. If so, a
> cohort model can legally move a cross-margin pair, which is the one thing four
> previous auxiliary representations could not do.

**It is half true, and the half that is true is the half that was already
priced.** The regimes exist and are large — one of the five carries an
eight-fold V8 under-prediction over 22 distinct failure devices in 10 buildings.
They improve out-of-fold likelihood on top of V8 by 8 %. They improve same-margin
ordering by 0.055. And they move cross-margin ordering by +0.0021 [−0.0007,
+0.0060], which a **signature shuffled among devices reproduces**: 1 of 8
permutations matches or beats it.

```
GATE FAILED. No planner run. See section 7 for the one live thread.
```

---

## 0. Git state, data and folds

| | |
|---|---|
| branch | `claude/battery-device-segmentation-9cd2dd` |
| base | `d38afe6` (`origin/main`) |
| merged for instruments | `5fe2dca` (`claude/final-j2w-precision`) at `76b0b59` |
| population | `outputs/v9_frame.npz` — 19,890 (scenario, battery) rows, 48 scenarios, 461 devices |
| model scored | `outputs/v7_folds.joblib`, V8 = V7 Wiener + `RemainingCalibration`, out of fold by building |
| folds | V8's own five building-disjoint groups, read off the fold bundle |
| new code | `tools/fj_signature.py`, `tools/fj_segment.py`, `tests/test_segment.py` |
| artefacts | `outputs/fj_segment{,_probe,_gate,_interpret}.json` |

**The landmark population and the metric are the committed ones, reproduced
exactly before anything was built.** Top 40 rows per scenario by V8 probability
among rows whose 42-day fate is *observed* — 1,708 rows, 376 due from 76 devices,
survivors from 133 devices, 10,348 within-scenario pairs of which **9,810 are
cross-margin** and 538 same-margin at a 0.01 V bin.

| | all | cross-margin | same-margin | f0 | f1 | f2 | f3 | f4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V8** | **0.7280** | **0.7359** | **0.5846** | 0.945 | 0.799 | 0.616 | 0.776 | 0.692 |

Every figure matches `docs/FINAL_FRAILTY.md` section 3 and `docs/FINAL_TEMPLATES.md`
section 3 to four decimals, including all five fold values. A fold's number is
measured on pairs whose **both** rows are held-out buildings; scenarios mix
buildings, so keying a fold on the due row alone scores a pair partly on training
buildings and gives a different (and wrong) table.

## 1. The signature: what kind of device, not where it sits

Seventeen features (`tools/fj_signature.py`), each an average, slope, coupling or
dispersion over the device's **whole observed life** up to the cutoff. `v_now` is
deliberately absent — it is the thing the signature is supposed to re-interpret.

| family | features |
|---|---|
| environment | `t_mean_life`, `t_std_life`, `t_p10_life`, `t_p90_life`, `t_amp_life`, `t_cold_frac`, `t_warm_frac` |
| baseline | `v_plateau` (own early-life level), `v_slope_life`, `v_curv_life` |
| thermal coupling | `beta_life` (V/°C over the whole life), `beta_r2_life` |
| dynamics | `v_std_detr_life`, `v_std_ratio_30_180` |
| observation | `age_days`, `obs_frac` |
| _margin-coupled, excluded from every fit_ | _`v_drop_life`_ |

Coverage 99.2 % of rows (87.6 % for the volatility ratio); 17,891 distinct
(device, cutoff) signatures behind 19,890 rows; build cost 21 s.

**Leakage discipline**, re-created independently inside every fold and pinned by
`tests/test_segment.py`:

* a signature at cutoff `c` reads `series[:c]` — tampering with `series[c:]`
  changes nothing, asserted feature by feature;
* fewer than 120 days of history returns NaN rather than a guess;
* no name may contain `building`, `room`, `device`, `eol`, `lifetime` or `due`;
* standardisation, centroids and neighbour pools are fitted on training-fold rows
  only. The decisive test inverts **every held-out row's 42-day outcome** and
  asserts the held-out rows' scores do not move by 1e-12 — with a control that
  the *other* building's scores do move, and a guard that the fold was scored at
  all. The control caught a first version of the test that was vacuous.

## 2. The probe: cohorts are worth something over margin, and nothing over V8

Nested out-of-fold fits on the landmarks, device-weighted (the 48 scenarios
overlap by ~85 %, so one device is up to 48 near-copies), margin entering as a
7-knot linear B-spline, cohorts as in-fold k-means on the standardised signature.

| anchored on **margin** | OOF logloss | all | cross | same |
|---|---:|---:|---:|---:|
| margin only | 0.59146 | 0.6703 | 0.6869 | 0.3680 |
| margin + 3 cohorts | 0.59593 | 0.6720 | 0.6802 | 0.5223 |
| **margin + 4 cohorts** | **0.58026** | 0.6794 | 0.6869 | 0.5428 |
| margin + 5 cohorts | 0.58210 | 0.6884 | **0.6955** | 0.5595 |
| margin + 5 cohorts × margin | 0.58221 | 0.6878 | 0.6947 | 0.5613 |

Against margin alone the cohort term is real: likelihood improves 1.9 %, and
cross-margin ordering improves **+0.0086**. That is the hypothesis working.

| anchored on **V8** | OOF logloss | all | cross | same |
|---|---:|---:|---:|---:|
| **V8 only** | 0.91178 | **0.7280** | **0.7359** | 0.5846 |
| V8 + 4 cohorts | **0.83679** | 0.7261 | 0.7322 | 0.6143 |
| V8 + 4 cohorts × margin | 0.83486 | 0.7243 | 0.7303 | 0.6152 |
| V8 + 5 cohorts | 0.84974 | 0.7298 | 0.7348 | **0.6394** |
| V8 + 5 cohorts × margin | 0.84762 | 0.7284 | 0.7330 | 0.6431 |

**This is the whole result in two lines.** Freely fitted on top of V8 the cohort
term buys a large likelihood gain (0.912 → 0.837, an 8 % reduction) and +0.055 of
same-margin concordance, and it **loses cross-margin concordance** — 0.7359 →
0.7348 / 0.7322. Fifth independent repetition of the pattern in
`docs/FINAL_TERMINALITY.md`: helps at fixed margin, hurts across margins. The
likelihood gain is a *level* gain, and level is the axis V9 proved does not
transfer (public 2078.28 → 2137.22 for one extra swap per scenario and zero extra
catches).

## 3. The gate: S1 and S2, order-only, against 0.7359

`logit(p) = logit(p_V8) + correction`, scored on the same landmarks.

**S1 — soft kNN analog cohort.** The query's signature retrieves the K nearest
*training-fold devices*; the correction is a difference of two censor-aware,
device-weighted kernel estimates at the query's own margin (0.02 V bandwidth):

    delta = logit P(due | margin, cohort) - logit P(due | margin, everyone)

so if cohort membership carries nothing the two curves coincide, delta is zero,
and no ordering moves by accident.

**S2 — in-fold segmentation.** Weighted k-means on training devices, held-out
devices assigned to the nearest centroid, one shrunk logit offset per segment
(`beta · n/(n+25)` in devices), optionally keyed on margin band.

| candidate | all | **cross-margin** | same-margin | cross folds won |
|---|---:|---:|---:|---:|
| **V8** | **0.7280** | **0.7359** | 0.5846 | — |
| S1 K=15 λ=0.125 | 0.7304 | 0.7374 | 0.6022 | 3/5 |
| S1 K=15 λ=0.25 | 0.7292 | 0.7366 | 0.5948 | 4/5 |
| S1 K=15 λ=1.0 | 0.7155 | 0.7229 | 0.5799 | 2/5 |
| S1 K=30 λ=0.125 | 0.7295 | 0.7371 | 0.5911 | 2/5 |
| **S1 K=50 λ=0.125** | 0.7300 | **0.7379** | 0.5855 | 3/5 |
| S1 K=50 λ=0.25 | 0.7295 | 0.7373 | 0.5874 | 3/5 |
| S1 K=50 λ=1.0 | 0.7254 | 0.7335 | 0.5762 | 2/5 |
| S2 k=3 flat | 0.7213 | 0.7270 | 0.6171 | 2/5 |
| S2 k=4 flat | 0.7289 | 0.7360 | 0.5994 | 3/5 |
| **S2 k=5 flat** | **0.7305** | 0.7369 | **0.6134** | 3/5 |
| S2 k=5 margin-banded 0.05 | 0.7237 | 0.7309 | 0.5911 | 3/5 |
| S2 k=5 margin-banded 0.03/0.08 | 0.6971 | 0.7039 | 0.5725 | 1/5 |

Twelve of the twenty-one settings are in the table; the full grid is in
`outputs/fj_segment_gate.json`. **This is the first direction on this project that
does not actively damage cross-margin ordering** — the frailty correction's
interval excluded zero on the negative side, this one straddles zero on the
positive. But:

* **every gain is monotone-decreasing in strength.** For K=50 the cross-margin
  number goes 0.7359 (λ=0) → 0.7379 → 0.7373 → 0.7368 → 0.7335 as λ rises through
  0.125 / 0.25 / 0.5 / 1.0. The correction is best when it is almost switched off.
* **keying the correction on margin band — the literal form of the hypothesis —
  is the worst arm in the table** (0.7039 at k=5, two bands).
* device bootstrap of the cross-margin delta, 300 draws, resampling *devices*:

| candidate | Δ cross-margin [95 % CI] | P(Δ > 0) |
|---|---|---:|
| S1 K=50 λ=0.125 | **+0.0021 [−0.0007, +0.0060]** | 0.90 |
| S1 K=15 λ=0.125 | +0.0016 [−0.0010, +0.0047] | 0.84 |
| S1 K=50 λ=0.25 | +0.0014 [−0.0028, +0.0067] | 0.67 |

Every interval contains zero.

## 4. The permutation control: the delta is not cohort information

The signature is shuffled among the devices present in **each scenario** — which
destroys the device-to-signature link and preserves the scenario composition, the
signature distribution and every other moving part — and S1 is re-run on it.

| | cross-margin Δ |
|---|---:|
| **real signature** | **+0.0020** |
| shuffled, 8 seeds | −0.0013, −0.0015, +0.0015, +0.0004, −0.0003, +0.0010, +0.0021, +0.0012 |
| shuffled mean / sd / max | +0.0004 / 0.0013 / **+0.0021** |

**One of eight permutations matches or beats the real signature** (p ≈ 0.13). The
machinery — a kernel-smoothed local recalibration of V8 at matched margin — makes
a small positive number on its own, whoever the neighbours are. The gate's best
result is that number.

## 5. What the regimes actually are

Five cohorts, refitted inside every fold; the profile below is the illustrative
whole-population fit (`--interpret`). Medians in raw units.

| cohort | rows | devices | EOL | censored | 42-d rate | t_mean | t_amp | v_plateau | v_slope (mV/d) | beta (V/°C) | v_std_detr | age (d) | **obs_frac** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 126 | 8 | 4 | 4 | 0.190 | 17.98 | 4.06 | **2.795** | −0.502 | 0.008 | 0.029 | 609 | 0.910 |
| 1 | 498 | 53 | 28 | 25 | 0.207 | 18.77 | 9.89 | 3.036 | **−0.709** | 0.009 | 0.037 | 728 | 0.870 |
| 2 | 253 | 27 | 15 | 12 | 0.253 | **22.65** | 8.47 | 3.051 | −0.590 | 0.006 | **0.067** | 830 | 0.930 |
| 3 | 608 | 53 | 16 | 37 | **0.137** | 20.24 | 3.98 | 3.048 | −0.528 | 0.010 | 0.057 | **974** | **0.998** |
| 4 | 223 | 38 | 24 | 14 | **0.457** | **17.30** | 9.62 | 3.008 | −0.562 | 0.011 | 0.029 | 753 | **0.780** |

They are recognisable physical regimes, not noise:

* **c3 — the quiet fleet.** Warm-stable (amplitude 4.0 °C), oldest (974 d),
  essentially complete telemetry (0.998), lowest failure rate (0.137).
* **c2 — the hot, noisy rooms.** 22.6 °C, the highest detrended voltage
  dispersion (0.067) and the weakest thermal coupling (0.006 V/°C).
* **c1 — the fast decliners.** The steepest lifetime slope (−0.709 mV/d).
* **c0 — the low-baseline cells.** `v_plateau` 2.795 V against ~3.04 elsewhere:
  these devices never had the headroom. Only 8 devices; treat as thin.
* **c4 — cold, seasonal, and badly observed.** 17.3 °C, 9.6 °C amplitude, and
  `obs_frac` **0.780** against 0.87–1.00 everywhere else. Highest failure rate by
  a wide margin (0.457) and the largest recent/long volatility ratio (0.375).

## 6. Does cohort change what a margin means? Yes — and it still does not rank

Realised 42-day failure rate by margin band and out-of-fold cohort. **Row rates
are reported beside device-weighted rates on purpose**: the six repeat false
positives of `docs/FINAL_FP_ANALYSIS.md` sit at small margin in up to 48
scenarios each, so a row-weighted low-margin cell is largely a statement about
them, and here the two weightings disagree enough to reverse the reading.

| band | c0 | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|---:|
| | _row / device_ | | | | |
| < 0.02 V | 0.591 / 0.833 | 0.929 / 0.833 | 0.217 / **0.694** | 0.147 / 0.750 | — |
| 0.02–0.05 V | 0.378 / 0.525 | 0.750 / 0.802 | 0.319 / 0.562 | 0.292 / **0.286** | 0.507 / 0.785 |
| 0.05–0.10 V | 0.229 / 0.373 | 0.250 / 0.433 | 0.228 / 0.269 | 0.089 / 0.165 | 0.267 / **0.599** |
| > 0.10 V | 0.134 / 0.110 | 0.069 / 0.117 | 0.070 / 0.106 | 0.000 / 0.000 | 0.258 / **0.366** |

**The hypothesis's own prediction is confirmed.** Device-weighted, c4 at
0.05–0.10 V (0.599) outranks c3 at 0.02–0.05 V (0.286), and c4 at 0.02–0.05 V
(0.785) is above c2 at < 0.02 V (0.694). A worse cohort at a wider margin really
does out-rank a better cohort at a tighter one.

And V8's calibration inside these cells, all device-weighted, says exactly what
kind of statement the cohorts are making:

| band | c0 | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|---:|
| | _realised ÷ V8 mean p_ | | | | |
| < 0.02 V | 1.19 | 1.25 | 0.92 | 0.98 | — |
| 0.02–0.05 V | 1.51 | 1.41 | 1.09 | 0.58 | **1.30** |
| 0.05–0.10 V | 1.58 | 2.02 | 1.44 | 0.99 | **2.10** |
| > 0.10 V | 2.40 | 2.85 | 2.19 | 0.00 | **7.89** |

**Read down the columns, not across.** At tight margin V8 is calibrated to within
±25 % in every cohort — which is `docs/FINAL_PROCESS.md`'s result (predicted
0.682 against realised 0.684 in the 0–0.03 V band) reproduced cohort by cohort.
The error grows monotonically with margin in every cohort, and c4 is the worst
cell of every band it appears in:

| cell | rows | devices | **due devices** | buildings | folds | realised (dev) | V8 p (dev) | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **c4, margin > 0.10 V** | 178 | 35 | **22** | 10 | 4 of 5 | **0.366** | 0.046 | **7.9×** |
| c1, margin > 0.10 V | 72 | 15 | 2 | — | — | 0.117 | 0.041 | 2.9× |
| c0, margin > 0.10 V | 127 | 26 | 5 | — | — | 0.110 | 0.046 | 2.4× |
| c2, margin > 0.10 V | 214 | 45 | 8 | — | — | 0.106 | 0.048 | 2.2× |

22 distinct failure devices across 10 buildings and 4 of the 5 folds — not the
repeated-device artefact that killed `weeks_observed` in `docs/FINAL_FRAILTY.md`.
V8's own shipped gap features are blind to it: `staleness` and `gap_fraction_90`
are **median 0 in every cohort**, which `docs/FINAL_J2W_RESULTS.md` already
recorded for the decision population. `obs_frac` is a whole-life quantity and
separates c4 cleanly; the shipped features are 90-day and do not.

**So why does none of this rank?** Because it is a level statement, not an order
statement. The miscalibration is *shared* — every cohort's ratio rises with
margin, tracking the tail-thinness of the Gaussian passage law that is already on
record. A cohort correction mostly re-learns that common curve, which is a change
to risk *mass*, which `docs/FINAL_J2W_RESULTS.md` section 4 forbids and V9
measured the cost of (public 2078.28 → 2137.22). What is left once the shared
level is removed — the genuinely cohort-specific part, which is the only part
that can move an ordering — is what the permutation control in section 4 priced
at zero.

## 7. Falsification: does the model know when to override margin?

Restricted to the 9,810 cross-margin pairs, a reversal is a **strict**
disagreement — both models order the pair and they order it opposite ways, so
"cohort right" and "V8 right" sum to one.

| candidate | reversals | rate | **cohort model correct** | V8 correct |
|---|---:|---:|---:|---:|
| S1 K=50 λ=0.125 | 64 | 0.7 % | 60.9 % | 39.1 % |
| **S1 K=50 λ=1.0** | **529** | **5.4 %** | **47.3 %** | **52.7 %** |

**No.** At the strength where the model actually overrides V8 it is right on
**47.3 %** of its own overrides — worse than a coin flip and worse than the
incumbent it is overruling. Stratified, it stays wrong everywhere and moves
without structure:

| stratum (λ = 1.0) | correct |
|---|---:|
| margin gap < 0.01 V (n=55) | 0.44 |
| margin gap 0.01–0.03 V (n=162) | 0.47 |
| margin gap 0.03–0.08 V (n=223) | 0.47 |
| margin gap > 0.08 V (n=89) | 0.52 |
| lifetime-temperature gap < 1 °C (n=108) | 0.54 |
| lifetime-temperature gap > 3 °C (n=239) | 0.45 |
| device-age gap > 300 d (n=174) | 0.45 |
| plateau gap < 0.01 V (n=61) | 0.41 |

The shrunk arm's 60.9 % rests on 64 pairs (SE 6.1 %, 1.8σ) and is internally
incoherent — 0.41 at one margin gap and 0.65 at another. The temperature-gap row
points the wrong way: the model is *least* accurate exactly where the cohorts are
most different.

## 8. Runtime

| stage | cost |
|---|---:|
| `--build` (17,891 signatures over 461 cached series) | 21 s |
| `--probe` (nested fits, both anchors, 5 folds) | 18 s |
| `--gate` (21 settings, 3 bootstraps, 2 reversal tables) | 31 s |
| `--interpret` (regimes, cohort table, 8 permutations) | 59 s |
| `python -m unittest discover -s tests` | 167 s, **144 tests, OK** |

No planner run: the gate did not pass, and `docs/FINAL_TEMPLATES.md` section 5
records what spending one anyway is worth.

## 9. Verdict

```
SEGMENT MODEL FAILS -- no transferable cross-margin information found.
```

The cohorts are real, physically interpretable, and out-of-fold they carry an 8 %
likelihood gain and +0.055 of same-margin concordance on top of V8. They are
worth +0.0086 of cross-margin concordance *over margin alone* and **negative**
cross-margin concordance over V8. The best shrunk arm's +0.0021 has a bootstrap
interval containing zero and is matched by 1 of 8 permutations of the signature.
When it overrides V8 it is right 47.3 % of the time.

Sixth construction to show the same thing, now with the mechanism named: **V8's
64 features already contain the regime information that bears on ordering, and
the part the cohorts add on top is level, not order.**

**The one live thread, and it is Task-1 calibration rather than Task-1 ranking:**
cohort 4 — cold, high seasonal amplitude, `obs_frac` 0.78 — is under-priced by V8
**7.9×** at margins above 0.10 V, over 22 distinct failure devices in 10 buildings,
and the shipped `staleness` / `gap_fraction_90` features cannot see it because
they are median 0 on the decision population. That is a probability-*level*
finding. `docs/FINAL_J2W_RESULTS.md` and the V9 public result both say level
changes do not transfer, so it must not be spent as one without a transfer
argument this branch has not built. Recorded, not deployed.

Per the brief's own instruction, the next representation experiment on the list —
a small causal-TCN self-supervised future-trajectory model on the increment
window population — was **not** started.
