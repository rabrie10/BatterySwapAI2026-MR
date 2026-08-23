# EOL-aligned trajectory templates: shape matching against the 82 deaths

_Branch `claude/final-j2w-precision`, 2026-08-23. Code: `bsai/templates.py`,
`tools/fj_templates.py`, `tests/test_templates.py` (11 tests). Measured out of
fold by building on the 19,890 cached scenario rows._

## 1. Why this is not another feature

Every earlier attempt on this branch reduced a trajectory to a scalar -- a
slope, a dwell, a volatility ratio -- and asked a model to combine scalars. This
one does not summarise. The 82 devices that crossed 2.4 V each leave a *curve*
running into their own end of life; the question asked of every active battery
at every cutoff is

> does the last W days of this battery's history look like the segment of a
> dying battery's curve that ended L days before it crossed?

and the output is **a predicted lead time**, estimated by k-nearest-neighbour
regression over template segments. Not a class label, not a fitted combination
of engineered features. It also sidesteps the sample-size wall differently: 82
crossings give 82 *labels* but roughly 2,600 distinguishable pre-EOL segments.

* windows **30 / 60 / 90 / 180** days;
* channels **voltage** and **temperature-adjusted voltage**
  (`v - 0.00463 (T - 20)`);
* normalisations **`anchored`** (subtract the window's last value, so only the
  recent *shape* survives and the level is discarded -- the genuinely new axis)
  and **`level`** (no subtraction, which partly re-encodes margin and is kept as
  the control);
* leads sampled every 7 days from 0 to 365 before the crossing;
* k = 25, softmin-weighted so a close single match is not dragged to the bulk.

**Fold discipline.** For each of V8's five building folds the bank is built only
from crossing devices whose building is in the other four, and only held-out
rows are scored against it. `tools/fj_templates.py` asserts on every fold that
no allowed device belongs to a held-out building, and
`tests/test_templates.py::FoldDisciplineTest` pins the primitive. Bank sizes:
2,212-2,615 segments per fold.

Coverage 77.7-81.8 % of landmark rows; the rest have too gappy a window.

## 2. Standalone: below V8, as expected

Within-scenario concordance on the standard landmark population (top 40 by V8
probability per scenario, censored rows excluded, **V8 = 0.7280**):

| | 30 d | 60 d | 90 d | 180 d |
|---|---:|---:|---:|---:|
| anchored, voltage | **0.5963** | 0.5900 | 0.5668 | 0.5582 |
| anchored, adjusted | 0.5799 | 0.5659 | 0.5555 | 0.5530 |
| level, voltage | 0.6530 | 0.6291 | 0.6217 | 0.5568 |
| level, adjusted | **0.6594** | 0.6415 | 0.6255 | 0.5685 |

Short windows beat long ones on every row, and `level` beats `anchored`
everywhere -- which is the expected reading: most of what a template match knows
is *where the battery sits*, and V8 already knows that better.

The pure-shape variants are still clearly above chance (0.55-0.60), so trajectory
shape is not nothing. The question is whether it is anything V8 does not have.

## 3. Combined with V8: worse overall, worse across margins

Order-only rank blend, `centred_rank(p_V8) + w x centred_rank(-lead)`, computed
within each scenario, missing rows neutral:

| template | w | all pairs | **cross-margin** | same-bin | folds beating V8 |
|---|---:|---:|---:|---:|---:|
| **V8 alone** | — | **0.7280** | **0.7359** | **0.5846** | — |
| anchored_voltage_30 | 0.25 | 0.7053 | 0.7099 | 0.6208 | 1/5 |
| anchored_voltage_30 | 0.50 | 0.6720 | 0.6750 | 0.6171 | 1/5 |
| anchored_voltage_60 | 0.25 | 0.6972 | 0.6998 | **0.6506** | 1/5 |
| anchored_voltage_60 | 1.00 | 0.6484 | 0.6493 | 0.6320 | 1/5 |
| anchored_adjusted_30 | 0.25 | 0.6842 | 0.6875 | 0.6245 | 2/5 |
| level_adjusted_30 | 0.50 | 0.6945 | 0.6998 | 0.5985 | 1/5 |
| level_voltage_30 | 0.50 | 0.6888 | 0.6945 | 0.5855 | 0/5 |

**All fifteen configurations lose overall concordance**, and every one loses
cross-margin concordance -- the criterion this experiment was set up to test.
At best 2 of 5 building folds beat V8.

```
GATE FAILED on its stated criterion. One planner run was spent anyway; see
section 5 for why, and for the fact that the extrapolation it replaced had the
sign wrong.
```

## 4. The one result worth recording

Splitting the pairs by whether the case and the control sit in the same 0.01 V
margin bin separates two very different stories:

| | V8 | best template blend | change |
|---|---:|---:|---:|
| **cross-margin pairs** | **0.7359** | 0.7099 | **−0.026** |
| **same-margin pairs** | **0.5846** | **0.6506** | **+0.066** |

At matched margin V8 is close to uninformative (0.585) and the pure-shape
template score adds a great deal -- **+0.066, five times the +0.013 the
volatility ratio gave at the same conditioning**. Across margins it is a
straight loss.

That is the third independent arrival at the same wall, and it is worth stating
plainly because it is now a pattern rather than a result:

1. matched volatility (`docs/FINAL_TERMINALITY.md`) -- helps at matched margin,
   harmful across;
2. trajectory templates (this document) -- helps *more* at matched margin,
   harmful across;
3. twelve independent channels in `docs/task1_investigation_findings.md` --
   "each attempted second axis collapses".

**Margin so dominates the cross-margin ordering that every other axis tested
degrades it, and every other axis only shows its value once margin is held
fixed -- which is about a third of the decisions the planner actually makes.**

## 5. The planner run, spent anyway -- and the extrapolation was wrong

The pre-registered rule was to stop before planner experimentation if OOF
concordance did not improve materially. It does not improve at all: every
configuration is below V8 overall and below V8 cross-margin, so the gate failed.

One run was spent regardless, because the alternative was to close the line on
an extrapolation. The reasoning would have been: the margin-conditioned
deployment -- reorder only within 0.01 V bins, the only way to spend a signal
validated at matched margin -- was already measured with the volatility ratio at
**-11.32 (t = -0.45)** and **-7.94 (t = -0.42)**, and the template score's
same-bin edge is five times larger, so a linear extrapolation lands near -50.

**Measured, it is +5.54 (t = +0.73, 6 wins / 11 losses).**

| | V8 | template rerank | delta |
|---|---:|---:|---:|
| mean total cost (deterministic planner) | **2091.23** | 2096.77 | **+5.54** |
| early / late | 765.7 / 1030.8 | 766.7 / 1034.4 | +1.0 / +3.6 |
| served / misses | 17.56 / 3.96 | 17.60 / 3.96 | +0.04 / 0.00 |
| precision / recall | 0.313 / 0.581 | 0.312 / 0.581 | -0.001 / 0.000 |
| runtime | 12.23 s/scen | 13.68 s/scen | +1.45 |

The extrapolation was not merely optimistic, it had the **sign wrong**: five
times the same-bin concordance edge bought slightly *less* than the weaker
signal did. A same-bin ordering gain does not scale into cost, because the
number of decisions it can reach is fixed at about a third and the rows it
exchanges sit at a true rate near 16 % either way.

That is worth more than the concordance table above it. It is the third and
cleanest demonstration on this branch that **ordering metrics do not convert
linearly into planner cost**, after the top-k screen over-reporting a
permutation by 100 points and the matched metric moving opposite to cost.
