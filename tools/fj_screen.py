"""Screen an order-only reranker on the cached scenario frame in seconds.

Every arm here is the same transformation: keep V8's per-scenario multiset of
CDF curves and reassign them by a candidate score (``bsai/rerank.py``). So the
predicted mass, the expected-due budget and the set of probability levels the
planner ever sees are identical across arms by construction, and the only thing
that moves is *which battery* gets which curve. That is the whole point -- V9
moved the mass up and V19 moved it down, and both lost.

Pricing reuses ``tools/rank_lab.py``'s arithmetic: a planned swap on day 1 of the
window, a missed due battery at the first emergency slot, 0.5 per early day and
10 per late day. On the shipped model that reproduces the out-of-fold planner
split to within a few percent, which is what licenses screening here before
paying ten minutes for a planner run.

Three views are reported for every arm, because a pooled win has misled this
project three times:

* **pooled** -- all 19,890 rows, out of fold by building;
* **hard transfer** -- the five adversarial building groups of
  ``docs/TRANSFER_STRESS.md``, with any fitted scorer refit on the complement;
* **temporal** -- six non-overlapping scenario blocks.

    python tools/fj_screen.py --arms base,oracle,warm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.hazard import HORIZON_GRID  # noqa: E402
from bsai.rerank import DECISION_HORIZON, remap  # noqa: E402
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402

DECISION_COLUMN = list(HORIZON_GRID).index(DECISION_HORIZON)
SWAP_COUNTS = (12, 15, 18, 21)
SWAP_DAY, EMERGENCY_DAY = 1.0, 48.0
EARLY_RATE, LATE_RATE = 0.5, 10.0


# --------------------------------------------------------------------------
# hard building groups, reconstructed from the dataset metadata
# --------------------------------------------------------------------------

def hard_groups(frame, dataset: Path = Path("dataset/train")) -> dict[str, tuple[str, ...]]:
    """The five adversarial holdouts of docs/TRANSFER_STRESS.md.

    Rebuilt from ``devices.csv``/``eol_times.csv`` and the frame's own beta_30
    medians rather than from that harness's stride frame, which no longer
    exists. The membership is reported on every run so a shift is visible.
    """
    from bsai.features import FEATURE_NAMES

    devices = pd.read_csv(dataset / "devices.csv")
    eol = pd.read_csv(dataset / "eol_times.csv")
    building_of = dict(zip(devices["device_id"].astype(str), devices["building_id"].astype(str)))
    sizes = devices.groupby("building_id")["device_id"].nunique()
    id_column = "device_id" if "device_id" in eol else eol.columns[0]
    recorded = eol[eol["end_time"].notna()]
    eol_building = pd.Series(
        [building_of.get(str(d), "") for d in recorded[id_column]]
    ).value_counts()
    counts = eol_building.reindex(sizes.index).fillna(0)
    rate = (counts / sizes).sort_values(ascending=False)
    sizes_desc = sizes.sort_values(ascending=False)

    beta = frame.features[:, FEATURE_NAMES.index("beta_30")].astype(float)
    global_median = np.nanmedian(beta)
    shift = {}
    for building in np.unique(frame.building):
        values = beta[frame.building == building]
        values = values[np.isfinite(values)]
        if values.size < 40:
            continue
        shift[building] = abs(np.log(max(np.median(values), 1e-6) / global_median))
    ranked = sorted(shift, key=lambda b: -shift[b])

    return {
        "hard_large5": tuple(sizes_desc.index[:5]),
        "hard_small10": tuple(sizes.sort_values(kind="stable").index[:10]),
        "hard_mosteol5": tuple(counts.sort_values(ascending=False, kind="stable").index[:5]),
        "hard_hirate6": tuple(rate.index[:6]),
        "hard_betashift5": tuple(ranked[:5]),
    }


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------

def price(frame, level: np.ndarray, rows: np.ndarray, counts=SWAP_COUNTS) -> dict:
    """Precision, recall and the evaluator's timing arithmetic at top-k."""
    effective = np.where(
        np.isfinite(frame.days_to_eol), frame.days_to_eol, frame.substitute_eol
    )
    scenarios = frame.scenario[rows]
    n = int(np.unique(scenarios).size)
    due = frame.due[rows]
    out = {}
    for k in counts:
        chosen = np.zeros(rows.size, dtype=bool)
        for index in np.unique(scenarios):
            local = np.flatnonzero(scenarios == index)
            order = local[np.argsort(-level[rows][local], kind="stable")]
            chosen[order[:k]] = True
        hits = int((chosen & due).sum())
        delta = effective[rows][chosen] - SWAP_DAY
        early = EARLY_RATE * np.maximum(delta, 0.0).sum()
        late_planned = LATE_RATE * np.maximum(-delta, 0.0).sum()
        missed = due & ~chosen
        late_missed = LATE_RATE * np.maximum(
            EMERGENCY_DAY - effective[rows][missed], 0.0
        ).sum()
        out[k] = {
            "precision": round(hits / max(int(chosen.sum()), 1), 4),
            "recall": round(hits / max(int(due.sum()), 1), 4),
            "early": round(float(early) / n, 1),
            "late": round(float(late_planned + late_missed) / n, 1),
            "timing": round(float(early + late_planned + late_missed) / n, 1),
        }
    return out


def average_precision(label: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if label.sum() == 0 or label.all():
        return float("nan")
    return float(average_precision_score(label, score))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def apply_remap(frame, grid: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Per-scenario order-only remap; returns the new decision probability."""
    out = np.empty(grid.shape[0])
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        moved = remap(grid[rows], frame.remaining[rows], score[rows], DECISION_COLUMN)
        out[rows] = decision_probability(moved, frame.remaining[rows])
    return out


def report(name: str, frame, level: np.ndarray, groups: dict, blocks: int = 6) -> dict:
    everything = np.arange(frame.features.shape[0])
    entry = {
        "arm": name,
        "pooled": price(frame, level, everything),
        "pooled_ap": round(average_precision(frame.due, level), 4),
        "sum_p_per_scenario": round(float(level.sum()) / frame.n_scenarios, 3),
        "transfer": {},
        "blocks": {},
    }
    for group, buildings in groups.items():
        rows = np.flatnonzero(np.isin(frame.building, buildings))
        entry["transfer"][group] = {
            "rows": int(rows.size),
            "due": int(frame.due[rows].sum()),
            "ap": round(average_precision(frame.due[rows], level[rows]), 4),
            **price(frame, level, rows, counts=(5, 10)),
        }
    edges = np.array_split(np.arange(frame.n_scenarios), blocks)
    for index, block in enumerate(edges):
        rows = np.flatnonzero(np.isin(frame.scenario, block))
        entry["blocks"][index] = price(frame, level, rows, counts=(15,))[15]
    return entry


def show(entries: list[dict], counts=SWAP_COUNTS) -> None:
    header = f"{'arm':>22} {'AP':>7} " + " ".join(
        f"{'k=' + str(k):>26}" for k in counts
    )
    print(header)
    print(f"{'':>22} {'':>7} " + " ".join(
        f"{'prec':>7}{'rec':>7}{'timing':>9}   " for _ in counts
    ))
    for entry in entries:
        line = f"{entry['arm']:>22} {entry['pooled_ap']:>7.4f} "
        for k in counts:
            cell = entry["pooled"][k]
            line += f"{cell['precision']:>7.3f}{cell['recall']:>7.3f}{cell['timing']:>9.1f}   "
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_screen.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    groups = hard_groups(frame)
    print("hard transfer groups:")
    for name, buildings in groups.items():
        rows = np.isin(frame.building, buildings)
        print(f"  {name:>16}: {len(buildings):2d} buildings, {int(rows.sum()):5d} rows, "
              f"{int(frame.due[rows].sum()):3d} due")
    print()

    entries = [report("V8 (base)", frame, base, groups)]

    # The ceiling: the same multiset of curves, handed out with hindsight.
    oracle_score = frame.due.astype(float) * 1e6 - np.arange(frame.due.size) * 0.0
    entries.append(
        report("oracle order", frame, apply_remap(frame, grid, oracle_score), groups)
    )

    show(entries)
    print()
    print("transfer (AP on held-out building group rows):")
    names = list(groups)
    print(f"{'arm':>22} " + " ".join(f"{n.replace('hard_',''):>12}" for n in names))
    for entry in entries:
        print(f"{entry['arm']:>22} " + " ".join(
            f"{entry['transfer'][n]['ap']:>12.4f}" for n in names))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(entries, indent=1))


if __name__ == "__main__":
    main()
