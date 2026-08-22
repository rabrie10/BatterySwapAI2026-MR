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

