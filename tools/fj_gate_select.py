"""One threshold on the remaining-observation axis, chosen without touching cost.

`docs/FINAL_TCN_REPRESENTATION.md` section 6: the remap is worth +0.09 of
cross-margin concordance in the two lower terciles of scenario remaining window
and **-0.0002** in the top one, where V8 is already at 0.7957 -- and a
concordance-neutral reorder still costs +224 there, because those are the opening
scenarios where a wasted swap is worth about 182.

So: turn the remap off above a threshold. One parameter, gated per *scenario*
rather than per row, because `own` and `own + other` live on different scales and
mixing them inside one sort would rank the ungated rows by an artefact. 95 % of a
scenario's rows share a single remaining value, so the scenario median is the
axis.

**The threshold is selected on held-out concordance, never on cost.** The
scenarios are split temporally: whatever maximises cross-margin concordance on
the earlier half is applied unchanged to the later half, and the reverse. Only if
both halves pick the same threshold *and* both held-out halves improve is a
single planner run spent.

    python tools/fj_gate_select.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.sequence import HORIZONS, SequenceModel, crossing_probability  # noqa: E402
from tools.fj_fit import standardise_within  # noqa: E402
from tools.fj_segment import context, logit  # noqa: E402
from tools.fj_tcn_gates import _positions, _view  # noqa: E402

THRESHOLDS = (120.0, 160.0, 200.0, 240.0, 280.0, 1e9)


def leave_one_fold_out_probability(ctx, args):
    """The deployed ensemble's 42-day crossing probability, honestly out of fold."""
    from tools.fj_tcn import Corpus
    from tools.fj_templates import crossing_index
    from tools.fj_terminality import load_series

    frame, fold, margin = ctx["frame"], ctx["fold"], ctx["margin"]
    series = load_series(args.series)
    corpus = Corpus(series, stride=3, stop=crossing_index(series, args.dataset))
    position = _positions(corpus, frame, args.dataset)
    model = SequenceModel.load(args.artifact)

    rows = np.flatnonzero(position >= 0)
    view = _view(corpus, position[rows])
    index = view.anchor[:, None] - 120 + 1 + np.arange(120)[None, :]
    voltage = corpus.filled[index]
    anchor = corpus.filled[view.anchor][:, None]
    windows = np.empty((rows.size, 5, 120))
    windows[:, 0] = (voltage - 2.4) / 0.5
    windows[:, 1] = (voltage - anchor) / 0.1
    windows[:, 2] = 0.0
    windows[:, 3] = corpus.mask[index]
    windows[:, 4] = np.minimum(corpus.stale[index], 30.0) / 30.0
    temperature = corpus.temperature[index]

    column = HORIZONS.index(42)
    probability = np.zeros(frame.scenario.size)
    for group in sorted(set(fold.tolist())):
        inside = fold[rows] == group
        if not inside.any():
            continue
        subset = SequenceModel(
            folds=[f for i, f in enumerate(model.folds) if i != group])
        predicted = subset.predict(windows[inside], temperature[inside])
        probability[rows[inside]] = crossing_probability(
            predicted[:, column, :], -margin[rows[inside]])
    scored = np.zeros(frame.scenario.size, bool)
    scored[rows] = True
    return probability, scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_segment.npz"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--artifact", type=Path,
                        default=Path("models/sequence_tcn.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/fj_gate_select.json"))
    args = parser.parse_args()

    ctx = context(args)
    frame, base, lab = ctx["frame"], ctx["base"], ctx["lab"]
    probability, scored = leave_one_fold_out_probability(ctx, args)

    standalone = logit(base).copy()
    standalone[scored] = logit(np.clip(probability, 1e-9, 1 - 1e-9))[scored]
    blend_all = (standardise_within(frame.scenario, logit(base))
                 + standardise_within(frame.scenario, standalone))
    plain = standardise_within(frame.scenario, logit(base))

    scenario_remaining = np.array(
        [np.median(frame.remaining[frame.scenario == s]) for s in range(48)])
    print("scenario median remaining: min %.0f, median %.0f, max %.0f"
          % (scenario_remaining.min(), np.median(scenario_remaining),
             scenario_remaining.max()))

    def gated(threshold: float) -> np.ndarray:
        keep = scenario_remaining[frame.scenario] <= threshold
        return np.where(keep, blend_all, plain)

    def cross_on(score: np.ndarray, scenarios: np.ndarray) -> float:
        keep = lab.split["cross"] & np.isin(frame.scenario[lab.positive], scenarios)
        return lab.concordance(score, keep)

    early, late = np.arange(24), np.arange(24, 48)
    print(f"\n{'threshold':>13} {'gated':>7} {'cross(all)':>11} "
          f"{'cross(0-23)':>12} {'cross(24-47)':>13}")
    table = {}
    for threshold in THRESHOLDS:
        score = gated(threshold)
        count = int((scenario_remaining <= threshold).sum())
        entry = {
            "scenarios_remapped": count,
            "cross_all": round(cross_on(score, np.arange(48)), 4),
            "cross_early": round(cross_on(score, early), 4),
            "cross_late": round(cross_on(score, late), 4),
        }
        table[f"{threshold:g}"] = entry
        label = "inf (ungated)" if threshold > 1e8 else f"{threshold:.0f}"
        print(f"{label:>13} {count:>7} {entry['cross_all']:>11.4f} "
              f"{entry['cross_early']:>12.4f} {entry['cross_late']:>13.4f}")

    def pick(train: np.ndarray) -> float:
        return max(THRESHOLDS, key=lambda t: cross_on(gated(t), train))

    picked_early, picked_late = pick(early), pick(late)
    held_late = cross_on(gated(picked_early), late)
    held_early = cross_on(gated(picked_late), early)
    ungated_late = cross_on(gated(1e9), late)
    ungated_early = cross_on(gated(1e9), early)

    print("\nnested temporal selection (threshold never sees the half it is read on)")
    print(f"  picked on 0-23  -> {picked_early:g};  held-out 24-47 cross "
          f"{held_late:.4f} vs ungated {ungated_late:.4f} "
          f"({held_late - ungated_late:+.4f})")
    print(f"  picked on 24-47 -> {picked_late:g};  held-out 0-23  cross "
          f"{held_early:.4f} vs ungated {ungated_early:.4f} "
          f"({held_early - ungated_early:+.4f})")
    agree = picked_early == picked_late
    improves = held_late > ungated_late and held_early > ungated_early
    print(f"  halves agree on the threshold: {agree}")
    print(f"  both held-out halves improve:  {improves}")
    verdict = "PROCEED to one planner run" if (agree and improves) else "ABANDON"
    print(f"\n  {verdict}")

    args.report.write_text(json.dumps({
        "thresholds": table,
        "picked_on_early": picked_early, "picked_on_late": picked_late,
        "held_out_early": round(held_early, 4), "held_out_late": round(held_late, 4),
        "ungated_early": round(ungated_early, 4), "ungated_late": round(ungated_late, 4),
        "agree": bool(agree), "both_improve": bool(improves), "verdict": verdict,
    }, indent=1))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
