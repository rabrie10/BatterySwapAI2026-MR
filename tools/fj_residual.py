"""Fit the 42-day residual ranking score under three objectives, one fold set.

Everything except the loss is held fixed: the same landmarks, the same eight
within-scenario ranked signals, the same five building-disjoint folds, the same
L2, the same order-only deployment. So the comparison prices the objective and
nothing else.

    (1) cost   -- 42-day binary log-loss, each landmark weighted by |service
                  value|, the official cost model's own answer to what getting
                  this row wrong is worth
    (2) focal  -- the same log-loss with a (1-p_t)^gamma modulator and a class
                  prior, and no cost weight, so the two differ in exactly one
                  respect
    (3) pair   -- weighted pairwise logistic ranking over within-scenario
                  (due, survivor) pairs that V8 already scores within `--delta`
                  logits of each other, each pair weighted by the service-value
                  gap it would get wrong

    python tools/fj_residual.py --report outputs/fj_residual.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.rerank import DECISION_HORIZON, centred_rank  # noqa: E402
from bsai.residual import (  # noqa: E402
    SIGNALS,
    OofResidualScorer,
    ResidualScorer,
    build_pairs,
    fit_pairwise,
    fit_pointwise,
    landmark_mask,
    service_value,
    signal_columns,
)
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402


def room_of(dataset: Path) -> dict[str, str]:
    devices = pd.read_csv(dataset / "devices.csv")
    return dict(zip(devices["device_id"].astype(str), devices["room_id"].astype(str)))


def per_scenario_design(frame, grid, rooms) -> np.ndarray:
    """Ranks are taken inside each scenario, exactly as the scorer will do."""
    matrix = np.zeros((frame.features.shape[0], len(SIGNALS)))
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        columns = signal_columns(
            frame.features[rows], grid[rows], frame.remaining[rows],
            frame.battery[rows], rooms,
        )
        matrix[rows] = np.column_stack(
            [centred_rank(columns[name]) for name in SIGNALS]
        )
    return matrix


def v8_folds(folds_path: Path) -> dict[str, int]:
    bundle = joblib.load(folds_path)
    seen: dict[int, int] = {}
    return {
        str(name): seen.setdefault(id(model), len(seen))
        for name, model in bundle["by_building"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--candidates", type=int, default=40,
                        help="rows per scenario kept as landmarks, by V8 rank")
    parser.add_argument("--delta", type=float, default=2.0,
                        help="max V8 logit gap for a pair to count as ambiguous")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal exponent")
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--sweep", action="store_true",
                        help="report out-of-fold and in-sample concordance across "
                             "L2 instead of writing scorers; separates 'no signal' "
                             "from 'not enough devices to fit one'")
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_residual_scorers.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_residual.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    anchor = np.log(np.clip(base, 1e-9, 1 - 1e-9) / (1 - np.clip(base, 1e-9, 1 - 1e-9)))
    rooms = room_of(args.dataset)
    matrix = per_scenario_design(frame, grid, rooms)

    effective = np.where(
        np.isfinite(frame.days_to_eol), frame.days_to_eol, frame.substitute_eol
    )
    value = service_value(effective)
    usable = landmark_mask(frame.days_to_eol, frame.remaining, frame.due)

    candidates = np.zeros(base.size, dtype=bool)
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        candidates[rows[np.argsort(-base[rows], kind="stable")][: args.candidates]] = True
    landmarks = usable & candidates

    fold_of = v8_folds(args.folds)
    fold = np.asarray([fold_of.get(b, -1) for b in frame.building])
    label = frame.due.astype(float)

    print(f"landmarks: {int(landmarks.sum())} rows "
          f"({landmarks.sum() / frame.n_scenarios:.1f} per scenario), "
          f"{int(frame.due[landmarks].sum())} positives from "
          f"{np.unique(frame.battery[landmarks & frame.due]).size} devices")
    print(f"excluded as censored before {DECISION_HORIZON} d: "
          f"{int((candidates & ~usable).sum())} rows "
          f"(of {int(candidates.sum())} in the candidate pool)")
    print(f"service value: positives median {np.median(value[landmarks & frame.due]):+.1f}, "
          f"negatives median {np.median(value[landmarks & ~frame.due]):+.1f}")
    rng = np.random.default_rng(20260823)
    pos_all, neg_all, weight_all = build_pairs(
        frame.scenario, frame.battery, frame.due, anchor, value, landmarks,
        delta=args.delta, rng=rng,
    )
    print(f"ambiguous pairs: {pos_all.size} from "
          f"{np.unique(frame.battery[pos_all]).size} due devices")
    print()

    def fit_fold(name: str, train: np.ndarray, l2: float, seed: int) -> np.ndarray:
        if name == "pair":
            pos, neg, weight = build_pairs(
                frame.scenario, frame.battery, frame.due, anchor, value, train,
                delta=args.delta, rng=np.random.default_rng(seed),
            )
            return fit_pairwise(matrix, anchor, pos, neg, weight, l2=l2)
        rows = np.flatnonzero(train)
        if name == "cost":
            weight = np.abs(value[rows])
        else:
            prior = label[rows].mean()
            weight = np.where(label[rows] > 0.5, 1.0 - prior, prior)
        return fit_pointwise(
            matrix[rows], anchor[rows], label[rows], weight,
            l2=l2, focal_gamma=args.gamma if name == "focal" else 0.0,
        )

    def concordance_of(score: np.ndarray) -> float:
        good = ties = total = 0.0
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero((frame.scenario == index) & landmarks)
            y = frame.due[rows]
            if y.sum() == 0 or y.all():
                continue
            gap = score[rows][y][:, None] - score[rows][~y][None, :]
            good += (gap > 0).sum()
            ties += (gap == 0).sum()
            total += gap.size
        return (good + 0.5 * ties) / total

    if args.sweep:
        print(f"V8 anchor concordance on the landmarks: {concordance_of(anchor):.4f}")
        print(f"{'objective':>10} {'L2':>8} {'sum|w|':>8} {'OOF':>8} {'in-sample':>10}")
        table = []
        for l2 in (0.2, 0.05, 0.02, 0.005, 0.001, 0.0):
            for name in ("cost", "focal", "pair"):
                oof = anchor.copy()
                norms = []
                for index in sorted(set(fold_of.values())):
                    held = fold == index
                    w = fit_fold(name, landmarks & ~held, l2, 20260823 + index)
                    norms.append(float(np.abs(w).sum()))
                    oof[held] = anchor[held] + matrix[held] @ w
                whole = fit_fold(name, landmarks, l2, 20260823)
                row = {
                    "objective": name, "l2": l2,
                    "sum_abs_w": round(float(np.mean(norms)), 3),
                    "oof": round(concordance_of(oof), 4),
                    "in_sample": round(concordance_of(anchor + matrix @ whole), 4),
                }
                table.append(row)
                print(f"{name:>10} {l2:>8.4f} {row['sum_abs_w']:>8.2f} "
                      f"{row['oof']:>8.4f} {row['in_sample']:>10.4f}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"concordance_v8": round(concordance_of(anchor), 4), "sweep": table},
            indent=1))
        return

    results: dict[str, dict] = {}
    scores: dict[str, np.ndarray] = {}
    for name in ("cost", "focal", "pair"):
        by_building: dict[str, np.ndarray] = {}
        score = anchor.copy()
        per_fold = []
        for index in sorted(set(fold_of.values())):
            held = fold == index
            train = landmarks & ~held
            if name == "pair":
                pos, neg, weight = build_pairs(
                    frame.scenario, frame.battery, frame.due, anchor, value,
                    train, delta=args.delta, rng=np.random.default_rng(20260823 + index),
                )
                w = fit_pairwise(matrix, anchor, pos, neg, weight, l2=args.l2)
            else:
                rows = np.flatnonzero(train)
                if name == "cost":
                    weight = np.abs(value[rows])
                else:
                    prior = label[rows].mean()
                    weight = np.where(label[rows] > 0.5, 1.0 - prior, prior)
                w = fit_pointwise(
                    matrix[rows], anchor[rows], label[rows], weight,
                    l2=args.l2, focal_gamma=args.gamma if name == "focal" else 0.0,
                )
            per_fold.append(w)
            score[held] = anchor[held] + matrix[held] @ w
            for building, value_index in fold_of.items():
                if value_index == index:
                    by_building[building] = w
        scores[name] = score
        mean = np.mean(per_fold, axis=0)
        spread = np.std(per_fold, axis=0)
        results[name] = {
            "mean_weights": dict(zip(SIGNALS, np.round(mean, 3).tolist())),
            "fold_sd": dict(zip(SIGNALS, np.round(spread, 3).tolist())),
        }
        print(f"{name:>6}  " + "  ".join(
            f"{s.split('_')[0][:6]}={m:+.2f}" for s, m in zip(SIGNALS, mean)))
        joblib.dump(
            {
                "by_building": by_building,
                "room_of": rooms,
                "names": SIGNALS,
                "objective": name,
            },
            args.out.with_name(f"{args.out.stem}_{name}.joblib"),
        )

    # Within-scenario concordance among the landmarks: does the order improve at
    # all before a planner run is spent on it?
    def concordance(score: np.ndarray) -> float:
        good = ties = total = 0.0
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero((frame.scenario == index) & landmarks)
            y = frame.due[rows]
            if y.sum() == 0 or y.all():
                continue
            gap = score[rows][y][:, None] - score[rows][~y][None, :]
            good += (gap > 0).sum()
            ties += (gap == 0).sum()
            total += gap.size
        return (good + 0.5 * ties) / total

    print()
    print(f"within-scenario concordance among landmarks (V8 = "
          f"{concordance(anchor):.4f}, out of fold by building):")
    for name, score in scores.items():
        results[name]["concordance"] = round(concordance(score), 4)
        print(f"  {name:>6}  {results[name]['concordance']:.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "landmarks": int(landmarks.sum()),
        "positives": int(frame.due[landmarks].sum()),
        "excluded_censored": int((candidates & ~usable).sum()),
        "pairs": int(pos_all.size),
        "concordance_v8": round(concordance(anchor), 4),
        "objectives": results,
        "settings": {"candidates": args.candidates, "delta": args.delta,
                     "gamma": args.gamma, "l2": args.l2},
    }, indent=1))
    print(f"\nwrote {args.out.with_name(args.out.stem + '_{cost,focal,pair}.joblib')}")


if __name__ == "__main__":
    main()
