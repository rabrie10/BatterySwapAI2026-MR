"""The candidate league on the cached frame: fit, remap, score, transfer.

Each arm is a set of residual signals. The weights are fitted with the pairwise
ranking objective of ``tools/fj_fit.py`` and the result is deployed the only way
this branch allows -- as an order-only remap of V8's own curves
(``bsai/rerank.py``), so every arm has exactly V8's predicted mass.

Two independent holdouts are reported, and a candidate has to survive both:

* **pooled OOF** -- weights fitted on four of V8's five building folds and
  applied to the fifth, so no row is ever ranked by weights that saw its own
  building;
* **hard transfer** -- weights refitted with an entire adversarial building
  group excluded, then applied to that group. These are the five groups of
  ``docs/TRANSFER_STRESS.md``, the closest thing this project has to a fresh
  building.

    python tools/fj_lab.py
    python tools/fj_lab.py --arms warm,dwell,warm+dwell --l2 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fj_derived import derived, room_of  # noqa: E402
from tools.fj_fit import (  # noqa: E402
    anchor_rank,
    build_pairs,
    design_matrix,
    fit_weights,
    label_and_mask,
    logit,
    standardise_within,
)
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402
from tools.fj_screen import apply_remap, hard_groups, price, report, show  # noqa: E402

# Signal sets, smallest first. Nothing here is a building identity and nothing
# reads a future value.
ARMS: dict[str, list[str]] = {
    # The physical state itself, temperature-compensated. The passage law uses
    # the *measured* margin as its barrier distance, and the measured voltage
    # moves +0.00463 V per degree; near the knee 0.02 V is about two weeks of
    # life. So this is a correction to which battery is closest to the barrier,
    # not to how much risk exists.
    "vcomp": ["voltage_compensated"],
    # Control: the same shrinkage toward the state variable with the
    # temperature term removed, so the compensation can be priced on its own.
    "vraw": ["voltage"],
    "vmin": ["voltage_min"],
    # The residual signals the false-positive profile named.
    "knee": ["beta_rise"],
    "slope": ["slope_comp_30"],
    "peer": ["rel_margin_room"],
    "warm": ["rel_temp_room"],
    "dwell": ["dwell_45_log"],
    "stale": ["staleness"],
    "sat": ["p07_over_p42"],
    "vcomp+peer": ["voltage_compensated", "rel_margin_room"],
    "vcomp+knee": ["voltage_compensated", "beta_rise"],
    "vcomp+slope": ["voltage_compensated", "slope_comp_30"],
    "vcomp+sat": ["voltage_compensated", "p07_over_p42"],
    "vcomp4": [
        "voltage_compensated", "beta_rise", "slope_comp_30", "rel_margin_room",
    ],
}


def v8_folds(frame, folds_path: Path) -> list[tuple[str, ...]]:
    """V8's own five building partitions, recovered by object identity."""
    bundle = joblib.load(folds_path)
    seen: dict[int, list[str]] = {}
    for name, model in bundle["by_building"].items():
        seen.setdefault(id(model), []).append(str(name))
    return [tuple(names) for names in seen.values()]


def fit_and_score(
    frame, anchor, design, usable, partitions, *, top_k, delta, l2
) -> tuple[np.ndarray, list[np.ndarray]]:
    out = anchor.copy()
    weights: list[np.ndarray] = []
    for held in partitions:
        inside = np.isin(frame.building, held)
        if not inside.any():
            continue
        sub = _Sub(frame, ~inside)
        pos, neg, weight = build_pairs(
            sub, anchor[~inside], usable[~inside], top_k=top_k, delta=delta
        )
        w = fit_weights(
            design[~inside], pos, neg, weight, l2=l2, anchor=anchor[~inside]
        )
        weights.append(w)
        out[inside] = anchor[inside] + design[inside] @ w
    return out, weights


class _Sub:
    def __init__(self, frame, mask: np.ndarray) -> None:
        self.scenario = frame.scenario[mask]
        self.battery = frame.battery[mask]
        self.due = frame.due[mask]
        self.building = frame.building[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--candidates", type=int, default=40,
                        help="rows per scenario the residual ranks are taken over")
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_lab.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    signals = derived(frame, grid, room_of())
    # Any shipped feature may be named as a residual signal too. It is already
    # inside the passage model, but the passage law is a fixed functional form
    # -- separability there does not mean the ordering uses it (HANDOVER trap 6
    # is about a free-form model; this is not one).
    from bsai.features import FEATURE_NAMES

    for index, name in enumerate(FEATURE_NAMES):
        signals.setdefault(name, frame.features[:, index].astype(float))
    groups = hard_groups(frame)
    partitions = v8_folds(frame, args.folds)
    anchor = logit(base)
    _, usable = label_and_mask(frame)
    # Rank the residual signals among the candidates only: the swaps all come
    # from the riskiest handful, and a rank taken over all 414 alive devices
    # spends its resolution on the ones that are never touched.
    candidates = np.zeros(base.size, dtype=bool)
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        candidates[rows[np.argsort(-base[rows], kind="stable")][: args.candidates]] = True
    print(f"candidate pool {candidates.sum() / frame.n_scenarios:.0f} rows/scenario, "
          f"{int(frame.due[candidates].sum())} of {int(frame.due.sum())} dues inside")
    anchor = anchor_rank(frame.scenario, base, candidates)

    entries = [report("V8 (base)", frame, base, groups)]
    oracle = frame.due.astype(float)
    entries.append(report("oracle order", frame, apply_remap(frame, grid, oracle), groups))

    detail: dict[str, dict] = {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        names = ARMS[name]
        design = design_matrix(frame, signals, names, subset=candidates)
        score, weights = fit_and_score(
            frame, anchor, design, usable, partitions,
            top_k=args.top_k, delta=args.delta, l2=args.l2,
        )
        level = apply_remap(frame, grid, score)
        entry = report(name, frame, level, groups)
        # Hard transfer: refit with the whole group out, then rank that group.
        for group, buildings in groups.items():
            held = np.isin(frame.building, buildings)
            sub = _Sub(frame, ~held)
            pos, neg, weight = build_pairs(
                sub, anchor[~held], usable[~held], top_k=args.top_k, delta=args.delta
            )
            w = fit_weights(
                design[~held], pos, neg, weight, l2=args.l2, anchor=anchor[~held]
            )
            transfer_score = anchor.copy()
            transfer_score[held] = anchor[held] + design[held] @ w
            transfer_level = apply_remap(frame, grid, transfer_score)
            rows = np.flatnonzero(held)
            from tools.fj_screen import average_precision

            entry["transfer"][group] = {
                "rows": int(rows.size),
                "due": int(frame.due[rows].sum()),
                "ap": round(average_precision(frame.due[rows], transfer_level[rows]), 4),
                **price(frame, transfer_level, rows, counts=(5, 10)),
            }
        entries.append(entry)
        detail[name] = {
            "signals": names,
            "weights": [[round(float(v), 4) for v in w] for w in weights],
            "mean_weight": [round(float(v), 4) for v in np.mean(weights, axis=0)],
        }
        print(f"  fitted {name:>12}: mean weights "
              f"{dict(zip(names, np.round(np.mean(weights, axis=0), 3)))}", flush=True)

    print()
    show(entries)
    print()
    names = list(groups)
    print("hard-transfer AP (weights refitted with the whole group held out):")
    print(f"{'arm':>22} " + " ".join(f"{n.replace('hard_', ''):>12}" for n in names)
          + f" {'mean':>8}")
    for entry in entries:
        values = [entry["transfer"][n]["ap"] for n in names]
        print(f"{entry['arm']:>22} " + " ".join(f"{v:>12.4f}" for v in values)
              + f" {np.mean(values):>8.4f}")
    print()
    print("temporal blocks, timing at k=15:")
    print(f"{'arm':>22} " + " ".join(f"{'b' + str(i):>9}" for i in range(6)))
    for entry in entries:
        print(f"{entry['arm']:>22} " + " ".join(
            f"{entry['blocks'][i]['timing']:>9.1f}" for i in range(6)))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"entries": entries, "fits": detail}, indent=1))


if __name__ == "__main__":
    main()
