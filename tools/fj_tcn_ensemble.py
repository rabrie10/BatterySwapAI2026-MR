"""Measure the configuration that actually ships: folds averaged, not routed.

Gate 2 sent each row to the one fold model that never saw its building. The
container cannot do that -- a test building belongs to no fold -- so it averages
all five. The honest local proxy is leave-one-fold-out: score a row with the
average of the four models that never saw its building, which is the deployed
ensemble minus the one member that would have been legal anyway.

    python tools/fj_tcn_ensemble.py
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
from tools.fj_segment import context, device_bootstrap, logit  # noqa: E402
from tools.fj_tcn_gates import _positions, _view  # noqa: E402


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
                        default=Path("outputs/fj_tcn_ensemble.json"))
    args = parser.parse_args()

    ctx = context(args)
    frame, fold, base, margin = ctx["frame"], ctx["fold"], ctx["base"], ctx["margin"]
    lab = ctx["lab"]

    from tools.fj_templates import crossing_index
    from tools.fj_terminality import load_series
    from tools.fj_tcn import Corpus

    series = load_series(args.series)
    corpus = Corpus(series, stride=3, stop=crossing_index(series, args.dataset))
    position = _positions(corpus, frame, args.dataset)
    model = SequenceModel.load(args.artifact)
    print(f"{len(model.folds)} folds loaded, "
          f"{sum(t.size for t in model.folds[0].tensors.values())} parameters each")

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
        # Leave-one-fold-out: the four models that never saw this building.
        subset = SequenceModel(folds=[f for i, f in enumerate(model.folds) if i != group])
        predicted = subset.predict(windows[inside], temperature[inside])
        probability[rows[inside]] = crossing_probability(
            predicted[:, column, :], -margin[rows[inside]])

    scored = np.zeros(frame.scenario.size, bool)
    scored[rows] = True
    standalone = logit(base).copy()
    standalone[scored] = logit(np.clip(probability, 1e-9, 1 - 1e-9))[scored]
    blend = (standardise_within(frame.scenario, logit(base))
             + standardise_within(frame.scenario, standalone))

    reference = lab.report(base, fold)
    table = lab.report(blend, fold)
    won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
    boot = device_bootstrap(lab, blend, base, "cross")
    print(f"\n{'':<26}{'all':>8}{'cross':>9}{'same':>8}")
    print(f"  {'V8':<24}{reference['all']:>8.4f}{reference['cross']:>9.4f}"
          f"{reference['same']:>8.4f}")
    print(f"  {'routed folds (gate 2)':<24}{0.7746:>8.4f}{0.7802:>9.4f}{0.6729:>8.4f}")
    print(f"  {'averaged folds (ships)':<24}{table['all']:>8.4f}{table['cross']:>9.4f}"
          f"{table['same']:>8.4f}")
    print(f"\n  folds won on cross-margin: {won}/5  "
          + " ".join(f"f{i} {table[f'x{i}']:.3f}/{reference[f'x{i}']:.3f}"
                     for i in range(5)))
    print(f"  device bootstrap: {boot['delta']:+.4f} "
          f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}], P(d>0) = {boot['p_positive']:.2f}")
    args.report.write_text(json.dumps(
        {"V8": reference, "ensemble": {**table, "folds_won_cross": won},
         "bootstrap": boot, "rows_scored": int(rows.size)}, indent=1))
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
