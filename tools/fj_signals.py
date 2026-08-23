"""Residual ranking signal: what orders a death above a survivor that V8 already
scores the same?

The decision this project actually makes is *within one scenario*: given ~414
alive batteries on one date, which fifteen do we touch. So a feature that
separates deaths from survivors across the whole pooled frame can be worthless
here -- ``season_sin`` reaches AUC 0.655 inside the candidate band and is almost
constant within a scenario, which means it moves volume between scenarios (what
``bsai/calibrate.py`` already corrects) and cannot reorder anything.

The metric here is therefore a *conditional pairwise accuracy*: inside one
scenario, take a due battery A and a surviving battery B whose V8 logits are
within ``--delta``, and ask how often the candidate signal puts A above B.
0.500 is no information. This is the quantity an order-only reranker converts
directly into precision, and it is blind by construction to anything constant
within a scenario.

    python tools/fj_signals.py --report outputs/fj_signals.json
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

from bsai.features import FEATURE_NAMES  # noqa: E402
from tools.fj_derived import derived, room_of  # noqa: E402
from tools.fj_frame import grid_for, load_frame  # noqa: E402

EPS = 1e-9


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def pair_index(
    scenario: np.ndarray,
    due: np.ndarray,
    base_logit: np.ndarray,
    *,
    delta: float,
    floor: float,
    rng: np.random.Generator,
    block: np.ndarray | None = None,
    max_pairs_per_scenario: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (positive, negative) of comparable within-scenario pairs.

    ``block`` restricts a pair to rows sharing a building or a room, which is
    how a device-level effect is told apart from a site-level one. Building
    identity must never become a *feature* (v3 learned it and collapsed on
    public), but conditioning a comparison on it is the standard control.
    """
    keys = scenario if block is None else np.char.add(
        np.char.add(scenario.astype(str), "|"), block.astype(str)
    )
    pos_out: list[np.ndarray] = []
    neg_out: list[np.ndarray] = []
    for index in np.unique(keys):
        rows = np.flatnonzero((keys == index) & (base_logit >= floor))
        if rows.size < 2:
            continue
        positives = rows[due[rows]]
        negatives = rows[~due[rows]]
        if positives.size == 0 or negatives.size == 0:
            continue
        gap = np.abs(base_logit[positives][:, None] - base_logit[negatives][None, :])
        pi, ni = np.nonzero(gap <= delta)
        if pi.size == 0:
            continue
        if pi.size > max_pairs_per_scenario:
            keep = rng.choice(pi.size, max_pairs_per_scenario, replace=False)
            pi, ni = pi[keep], ni[keep]
        pos_out.append(positives[pi])
        neg_out.append(negatives[ni])
    if not pos_out:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(pos_out), np.concatenate(neg_out)


def pairwise_accuracy(
    score: np.ndarray, positives: np.ndarray, negatives: np.ndarray
) -> tuple[float, int]:
    a, b = score[positives], score[negatives]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() == 0:
        return float("nan"), 0
    wins = (a[good] > b[good]).astype(float) + 0.5 * (a[good] == b[good])
    return float(wins.mean()), int(good.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--delta", type=float, default=0.75,
                        help="max V8 logit gap for two rows to count as comparable")
    parser.add_argument("--floor", type=float, default=-6.0,
                        help="ignore rows V8 scores below this logit (p ~ 0.0025)")
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--block", choices=("none", "bldg", "room"), default="none",
                        help="restrict pairs to rows sharing a building or a room")
    parser.add_argument("--min-pairs", type=int, default=400)
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_signals.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    extra = derived(frame, grid, room_of())
    base = logit(frame.probability)
    rng = np.random.default_rng(20260823)
    block = None
    if args.block == "bldg":
        block = frame.building
    elif args.block == "room":
        block = np.asarray([room_of().get(b, "?") for b in frame.battery])
    positives, negatives = pair_index(
        frame.scenario, frame.due, base,
        delta=args.delta, floor=args.floor, rng=rng, block=block,
    )
    print(f"{positives.size} comparable within-scenario pairs "
          f"(|logit gap| <= {args.delta}, logit >= {args.floor})")
    print(f"distinct due rows {np.unique(positives).size}, "
          f"distinct survivor rows {np.unique(negatives).size}")
    print(f"distinct due devices {np.unique(frame.battery[positives]).size}, "
          f"buildings {np.unique(frame.building[positives]).size}")
    print()

    control, n = pairwise_accuracy(base, positives, negatives)
    print(f"control -- V8 logit itself: {control:.4f} on {n} pairs "
          f"(must be ~0.5 by construction)")
    print()

    candidates = {name: frame.features[:, index].astype(float)
                  for index, name in enumerate(FEATURE_NAMES)}
    candidates.update(extra)
    rows = []
    for name, column in candidates.items():
        accuracy, count = pairwise_accuracy(column, positives, negatives)
        if not np.isfinite(accuracy) or count < args.min_pairs:
            continue
        rows.append({"signal": name, "accuracy": round(accuracy, 4), "pairs": count})
    rows.sort(key=lambda r: -abs(r["accuracy"] - 0.5))
    print(f"{'signal':>26} {'pair acc':>9} {'|edge|':>8} {'pairs':>8}")
    for row in rows[:28]:
        print(f"{row['signal']:>26} {row['accuracy']:>9.4f} "
              f"{abs(row['accuracy']-0.5):>8.4f} {row['pairs']:>8d}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"delta": args.delta, "floor": args.floor, "block": args.block,
         "pairs": int(positives.size), "control": round(control, 4),
         "signals": rows}, indent=2))


if __name__ == "__main__":
    main()
