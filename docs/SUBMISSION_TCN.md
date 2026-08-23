# Submission candidate: V8 levels, sequence-model order, shipped planner

_Candidate prepared on `claude/battery-device-segmentation-9cd2dd`. This file is
the reproducibility record required by `docs/SUBMISSION_CHECKLIST.md` §6._

**What ships.** V8's first-passage forecast, unchanged, with its per-scenario
CDF multiset handed out in the order a small causal TCN prefers, planned by the
**four-sample** search that V8 itself shipped.

| | |
|---|---|
| Task 1 levels | `models/v7_wiener.joblib` — V8 phase 1, `bsai-wiener/v1`, public **2078.28** |
| Task 1 ordering | `models/sequence_tcn.json` — `bsai-sequence/v1`, 5 folds × 12,899 parameters |
| Task 2 | `batteryswap_solution` planner, `robust_emergency_samples=4`, 80/35 search, solver 0.5 s |
| entry point | `script.py`, unchanged interface |
| evidence | `docs/FINAL_TCN_REPRESENTATION.md` |

---

## 1. Why this configuration and not the faster-scoring one

The deterministic-emergency planner (`robust_emergency_samples=0`) is worth
−35.30 on V8's own ordering and is what `script.py` defaulted to before this
commit. **It is not used here.** Measured through this entry point:

| configuration | s/scenario | projected for 96 | governor (soft 25 / hard 27.5 min) |
|---|---:|---:|---|
| V8 + deterministic (previous default) | 12.23 | 21.8 min | OK |
| **sequence remap + 4-sample search (this candidate)** | **12.40** | **22.1 min** | **OK** |
| sequence remap + deterministic | 16.48 | **28.6 min** | past the hard deadline |

The combination that scores best locally is the one that does not fit. Runtime
headroom decides it.

## 2. What the ordering is worth

Out of fold by building, 48 train scenarios, order-only:

| | V8 | candidate | Δ |
|---|---:|---:|---:|
| mean total cost | 2126.53 | **2055.58** | **−70.96** |
| median scenario | — | — | **−91.16** |
| paired t / wins | — | — | −1.23, 31 W / 17 L |
| early swap | 764.18 | 743.48 | −20.70 |
| late swap | 1045.83 | 984.79 | −61.04 |
| precision | 0.313 | **0.329** | +0.017 |
| recall | 0.577 | **0.606** | +0.029 |
| swaps / scenario | 17.458 | 17.396 | −0.062 |

Ranking quality, on the standard landmark population:

| | overall | cross-margin | same-margin |
|---|---:|---:|---:|
| V8 | 0.7280 | 0.7359 | 0.5846 |
| **as deployed (folds averaged)** | **0.7870** | **0.7920** | **0.6952** |

Device bootstrap of the cross-margin delta: **+0.0564 [+0.0350, +0.0775]**,
P(Δ>0) = 1.00, **5 of 5 building folds improved**.

**The deployed configuration is measured, not assumed.** Gate 2 routed each row
to the single fold model that never saw its building; a container cannot do that,
because a test building belongs to no fold, so it averages all five. The number
above is the honest leave-one-fold-out proxy for that ensemble
(`tools/fj_tcn_ensemble.py`) and is *better* than the routed 0.7802.

## 3. The artifact, and the three ways it could have failed silently

`models/sequence_tcn.json`, 1.40 MB, 5 folds × 12,899 parameters.

* **Git-LFS.** `.gitattributes` routes `*.pt`, `*.npz` and `*.npy` through LFS,
  so the training artifact `outputs/fj_tcn.pt` **is a pointer in the repository**.
  A checkout without the smudge filter would have handed the model 132 bytes of
  text. The shipped artifact is plain JSON, which is tracked normally;
  `SequenceModel.load` raises on a pointer rather than parsing one, and
  `tests/test_sequence.py` asserts both halves.
* **The Dockerfile never copied it.** `COPY models/ ./models` is in the image;
  `outputs/` is not. An artifact under `outputs/` would simply not exist at
  inference. It now lives in `models/`.
* **A lookup table cannot generalise.** The gate measurements used a per-row
  score keyed on `(device, remaining)` precomputed for the 48 cached training
  scenarios. On unseen public/private buildings **every lookup would miss**, the
  scorer would degrade to the identity, and the submission would silently be
  plain V8 at 2126.53. The container rebuilds the 120-day window from the
  forecaster's own smoothing cache and scores it live.

## 4. No Torch at inference

`torch>=2.13.0` is in the competition-provided `requirements.txt`, so the
dependency would be allowed. It is still not used: `bsai/sequence.py` implements
the forward pass — six dilated causal convolutions, GroupNorm, exact GELU, two
linear layers — in NumPy. `tools/fj_export_tcn.py` refuses to write the artifact
unless NumPy and Torch agree, and measured **1.05e-6** worst-case absolute
difference across all five folds. Torch is needed only to *train*, never to run.

This also settles determinism: there is no RNG anywhere in the inference path,
so the output is bit-stable across runs (asserted in `tests/test_sequence.py`).

## 5. The order-only invariant

`bsai.rerank.RankRemapModel` permutes whole CDF rows within each equal-`remaining`
group, so the multiset of curves — and therefore the per-scenario risk mass — is
preserved by construction rather than by a tolerance. Asserted directly:

* `sort(p_candidate) == sort(p_V8)` per scenario, and equal sums, to 1e-12;
* the whole 24-point curve travels with its battery, not just the 42-day column;
* `BATTERYSWAP_SEQUENCE_WEIGHT=0` reproduces the incumbent exactly;
* a row the sequence model cannot score (fewer than 120 days of grid, no observed
  voltage at the cutoff, device absent from the cache) keeps V8's own rank, so a
  cold start degrades to V8 rather than to noise.

95 % of rows sit in a single equal-`remaining` group of about 394 batteries, so
the remap has nearly full freedom despite the grouping.

## 6. Reproducing the artifact from scratch

```bash
python tools/fj_tcn.py --train --epochs 8 --stride 3 --threads 16 \
                       --model outputs/fj_tcn.pt          # ~40 min, 5 folds
python tools/fj_export_tcn.py --model outputs/fj_tcn.pt \
                              --out models/sequence_tcn.json
python tools/fj_tcn_ensemble.py                            # deployed-config gate
python -m unittest discover -s tests
```

Training reads only `outputs/fj_series.npz` (rebuildable with
`python tools/fj_terminality.py --rebuild-series`) and `dataset/train`. Windows
are **106,612** from 456 devices; every origin is strictly before its device's
EOL while targets are allowed to run through it — the invariant is asserted in
`train()` and pinned by `tests/test_tcn.py::OriginPrecedesEolTest`.

## 7. Environment switches

| variable | default | effect |
|---|---|---|
| `BATTERYSWAP_SEQUENCE_PATH` | `models/sequence_tcn.json` | `""` ships plain V8 |
| `BATTERYSWAP_SEQUENCE_WEIGHT` | `1.0` | `0.0` is exactly the incumbent order |
| `BATTERYSWAP_ROBUST_SAMPLES` | `4` | `0` is the deterministic path — **too slow with the remap** |

Every failure path logs at ERROR and falls back to plain V8. The identity of what
was loaded is logged on every run, so a silent downgrade is visible in the
transcript rather than inferred from the score.

## 8. Known limitations, stated plainly

* The planner delta is favourable in mean, median and trimmed mean but **is not
  statistically significant** (t = −1.23; sign test p = 0.059).
* It is **not uniform across scenarios**. Split by the scenario's mean
  remaining-observation window: concordance +0.0917 / +0.0925 / **−0.0002** and
  planner −122 / −315 / **+224** across the low, mid and high terciles. One third
  of scenarios is made worse. Gating the blend weight on that axis is the
  obvious next move and is deliberately **not** fitted here.
* Local rank has been a poor guide to public transfer three times on this project
  (V9, V10, V19). This candidate is offered as the next public measurement, not
  as a proven improvement.
