"""Score any candidate ranker against the shipped one in seconds, not minutes.

Reads the cached scenario frame from ``tools/build_scenario_frame.py`` and
reports the only numbers that decide anything: precision and recall at the swap
counts the leaderboard charges, and a timing estimate built from the evaluator's
own arithmetic.

The timing estimate is grounded rather than assumed. ``tools/swap_ledger.py``
measured where the planner actually puts swaps -- the median offset is **day 1**
of a 42-day window -- so a swap is priced at day 1 here, and a missed due battery
at the emergency queue's first slot. On the shipped model this reproduces the
out-of-fold ``validate_v6`` split to within a few percent, which is what licenses
using it to screen ideas before paying for a planner run.

Folds are the ones the shipped model was trained under, recovered from the
bundle by object identity, so a candidate fitted here never sees its own
building.

    python tools/rank_lab.py                     # baselines only
    python tools/rank_lab.py --band-probe        # what separates deaths at low margin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES

EOL_THRESHOLD = 2.4
SWAP_COUNTS = (10, 12, 15, 18, 21, 25)
# The planner's realised behaviour, from tools/swap_ledger.py.
SWAP_DAY = 1.0
HORIZON = 42.0
# First emergency slot: the window end rounded up to Sunday, plus queue position.
EMERGENCY_DAY = 48.0
EARLY_RATE, LATE_RATE = 0.5, 10.0


class Frame:
    def __init__(self, path: Path, folds: Path) -> None:
        z = np.load(path, allow_pickle=True)
        self.features = z["features"]
        self.scenario = z["scenario_index"]
        self.building = np.asarray([str(b) for b in z["building"]])
        self.probability = z["probability"]
        self.remaining = z["remaining"]
        self.due = z["due"].astype(bool)
        self.days_to_eol = z["days_to_eol"]
        self.substitute = z["substitute_eol"]
        self.margin = (
            self.features[:, FEATURE_NAMES.index("voltage")].astype(float)
            - EOL_THRESHOLD
        )
        # Effective EOL the evaluator prices: the record if there is one inside
        # the observation window, else last data + unobserved_eol_days.
        self.effective = np.where(
            np.isfinite(self.days_to_eol), self.days_to_eol, self.substitute
        )
        bundle = joblib.load(folds)
        seen: dict[int, int] = {}
        fold_of: dict[str, int] = {}
        for name, model in bundle["by_building"].items():
            key = id(model)
            seen.setdefault(key, len(seen))
            fold_of[str(name)] = seen[key]
        self.fold = np.asarray([fold_of.get(b, -1) for b in self.building])
        self.n_scenarios = int(len(np.unique(self.scenario)))

    def __len__(self) -> int:
        return int(self.features.shape[0])


def operating_points(frame: Frame, score: np.ndarray, label: str) -> list[dict]:
    """Top-k per scenario, priced with the evaluator's timing arithmetic."""
    order = np.lexsort((-score, frame.scenario))
    rank = np.empty(len(frame), dtype=int)
    start = 0
    scen_sorted = frame.scenario[order]
    for index in range(len(order)):
        if index and scen_sorted[index] != scen_sorted[index - 1]:
            start = index
        rank[order[index]] = index - start

    out = []
    total_due = int(frame.due.sum())
    for k in SWAP_COUNTS:
        chosen = rank < k
        hits = int((chosen & frame.due).sum())
        delta = frame.effective[chosen] - SWAP_DAY
        early = EARLY_RATE * np.maximum(delta, 0.0).sum()
        late_planned = LATE_RATE * np.maximum(-delta, 0.0).sum()
        missed = frame.due & ~chosen
        late_missed = LATE_RATE * np.maximum(
            EMERGENCY_DAY - frame.effective[missed], 0.0
        ).sum()
        n = frame.n_scenarios
        out.append(
            {
                "label": label,
                "k": k,
                "precision": round(hits / max(int(chosen.sum()), 1), 4),
                "recall": round(hits / max(total_due, 1), 4),
                "early": round(float(early) / n, 1),
                "late": round(float(late_planned + late_missed) / n, 1),
                "timing": round(float(early + late_planned + late_missed) / n, 1),
            }
        )
    return out


def show(rows: list[dict]) -> None:
    print(f"{'label':>18} {'k':>4} {'precis':>8} {'recall':>8} {'early':>9} "
          f"{'late':>9} {'timing':>9}")
    for r in rows:
        print(f"{r['label']:>18} {r['k']:>4} {r['precision']:>8.3f} {r['recall']:>8.3f} "
              f"{r['early']:>9.1f} {r['late']:>9.1f} {r['timing']:>9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--band-probe", action="store_true")
    parser.add_argument("--band", type=float, default=0.12,
                        help="margin below which a device is a plausible candidate")
    parser.add_argument("--report", type=Path, default=Path("outputs/v9_rank_lab.json"))
    args = parser.parse_args()

    frame = Frame(args.frame, args.folds)
    n = frame.n_scenarios
    print(f"{len(frame)} rows, {n} scenarios ({len(frame)/n:.1f} per scenario), "
          f"{int(frame.due.sum())} due ({frame.due.sum()/n:.2f} per scenario)")
    print(f"predicted mass {frame.probability.sum()/n:.2f} per scenario")
    print()

    results = []
    results += operating_points(frame, frame.probability, "wiener (shipped)")
    results += operating_points(frame, -frame.margin, "margin only")
    # The two-line physics control from HANDOVER trap 1, rebuilt on this frame.
    slope = frame.features[:, FEATURE_NAMES.index("slope_30")].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        physics = np.where(slope < 0, frame.margin / -slope, 1e9)
    results += operating_points(frame, -physics, "margin / -slope")
    show(results)

    print()
    print("=== the candidate band: how much is even reachable? ===")
    for lo, hi in [(-9, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.12),
                   (0.12, 0.20), (0.20, 9)]:
        m = (frame.margin >= lo) & (frame.margin < hi)
        if not m.any():
            continue
        print(f"  margin [{lo:6.2f},{hi:5.2f})  rows/scen {m.sum()/n:6.2f}  "
              f"deaths/scen {frame.due[m].sum()/n:5.2f}  due rate {frame.due[m].mean():.4f}  "
              f"model mass/scen {frame.probability[m].sum()/n:5.2f}")

    band = frame.margin < args.band
    print(f"\n  band margin < {args.band}: {band.sum()/n:.2f} rows/scenario, "
          f"{frame.due[band].sum()/n:.2f} deaths/scenario "
          f"({frame.due[band].sum()/max(frame.due.sum(),1):.1%} of all deaths), "
          f"due rate {frame.due[band].mean():.3f}")

    if args.band_probe:
        print()
        print(f"=== inside the band, what separates the deaths? (one feature at a time) ===")
        rows = []
        for index, name in enumerate(FEATURE_NAMES):
            column = frame.features[band, index].astype(float)
            finite = np.isfinite(column)
            target = frame.due[band][finite]
            if finite.sum() < 300 or target.sum() < 20 or target.all():
                continue
            auc = float(roc_auc_score(target, column[finite]))
            rows.append({"feature": name, "auc": round(max(auc, 1 - auc), 4),
                         "direction": "higher" if auc > 0.5 else "lower"})
        table = pd.DataFrame(rows).sort_values("auc", ascending=False)
        print(table.head(20).to_string(index=False))
        model_auc = roc_auc_score(frame.due[band], frame.probability[band])
        print(f"\n  the shipped model inside the band: AUC {model_auc:.4f}")
        print(f"  margin alone inside the band:       "
              f"{roc_auc_score(frame.due[band], -frame.margin[band]):.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
