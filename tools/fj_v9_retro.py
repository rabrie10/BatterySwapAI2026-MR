"""Retrospective: would the hard building-holdout gate have rejected V9?

``docs/PIHYBRID_HANDOFF.md`` calls the leave-building-out harness "the
instrument that CORRECTLY predicted the V19 public failure -- trust it", and
`docs/ENSEMBLE_FINDINGS.md` used its 0.428 bar to award a GO. Neither claim was
ever tested against V9, whose public row came back 59 points *worse* than V8
from a local out-of-fold score 373 points *better*. A gate that cannot see that
is not a gate.

The test isolates the component V9 actually added. V8's Wiener probability is
held fixed at its five-fold out-of-building value for every row, and only the
gradient-boosted head is refitted with an entire adversarial building group
excluded, then applied to that group. So any degradation measured here belongs
to the head, which is the piece the transfer argument was about.

The head is refitted from the cached scenario frame -- the same rows and the
same construction ``tools/train_blend.py`` uses -- so no smoothing pass is
needed and a group costs a couple of minutes rather than a day.

    python tools/fj_v9_retro.py --report outputs/fj_v9_retro.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.blend import DECISION_HORIZON, BlendedModel  # noqa: E402
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402
from tools.fj_screen import average_precision, hard_groups, price  # noqa: E402

WEIGHT = 0.5  # V9 ships the geometric mean, i.e. exponent 0.5 on each side.


def head_probability(heads: list, features, remaining) -> np.ndarray:
    design = BlendedModel.head_design(features, remaining, DECISION_HORIZON)
    total = np.zeros(features.shape[0])
    for head in heads:
        total += np.log(np.clip(head.predict_proba(design)[:, 1], 1e-12, 1.0))
    return np.exp(total / len(heads))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--seeds", type=int, default=3,
                        help="heads per fit; V9 ships five, three is the same signal")
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_v9_retro.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    groups = hard_groups(frame)
    seeds = tuple(range(20260822, 20260822 + args.seeds))
    started = time.time()

    results = {}
    for name, buildings in groups.items():
        held = np.isin(frame.building, buildings)
        heads = BlendedModel.fit_heads(
            frame.features[~held], frame.remaining[~held],
            frame.days_to_eol[~held], seeds=seeds,
        )
        head = head_probability(heads, frame.features[held], frame.remaining[held])
        blend = np.zeros(frame.due.size)
        blend[held] = (
            np.clip(base[held], 1e-12, 1.0) ** (1.0 - WEIGHT)
            * np.clip(head, 1e-12, 1.0) ** WEIGHT
        )
        rows = np.flatnonzero(held)
        head_full = np.zeros(frame.due.size)
        head_full[held] = head
        entry = {
            "rows": int(rows.size),
            "due": int(frame.due[rows].sum()),
            "ap_v8": round(average_precision(frame.due[rows], base[rows]), 4),
            "ap_head": round(average_precision(frame.due[rows], head_full[rows]), 4),
            "ap_blend": round(average_precision(frame.due[rows], blend[rows]), 4),
            "sum_p_v8": round(float(base[rows].sum()), 2),
            "sum_p_blend": round(float(blend[rows].sum()), 2),
            "realised": int(frame.due[rows].sum()),
            "timing_v8": price(frame, base, rows, counts=(10,))[10],
            "timing_blend": price(frame, blend, rows, counts=(10,))[10],
        }
        # Rank disagreement: where does the head move batteries the passage law
        # is unsure about?
        order_v8 = np.argsort(np.argsort(-base[rows]))
        order_bl = np.argsort(np.argsort(-blend[rows]))
        entry["mean_abs_rank_shift"] = round(float(np.abs(order_v8 - order_bl).mean()), 1)
        results[name] = entry
        print(f"  {name:>16}  AP v8 {entry['ap_v8']:.4f}  head {entry['ap_head']:.4f}  "
              f"blend {entry['ap_blend']:.4f}   mass {entry['sum_p_v8']:.1f} -> "
              f"{entry['sum_p_blend']:.1f} vs {entry['realised']} realised   "
              f"({time.time() - started:.0f}s)", flush=True)

    mean_v8 = float(np.mean([e["ap_v8"] for e in results.values()]))
    mean_blend = float(np.mean([e["ap_blend"] for e in results.values()]))
    wins = sum(1 for e in results.values() if e["ap_blend"] > e["ap_v8"])
    print()
    print(f"hard-holdout mean PR-AUC: V8 {mean_v8:.4f}   V9 blend {mean_blend:.4f}   "
          f"blend wins {wins}/5")
    print(f"the gate's own bar was 0.428; V9 blend reads "
          f"{'PASS' if mean_blend >= 0.428 else 'FAIL'}")
    mass_v8 = sum(e["sum_p_v8"] for e in results.values())
    mass_blend = sum(e["sum_p_blend"] for e in results.values())
    realised = sum(e["realised"] for e in results.values())
    print(f"predicted mass over all held-out rows: V8 {mass_v8:.1f}, "
          f"V9 blend {mass_blend:.1f}, realised {realised}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "seeds": args.seeds,
        "mean_ap_v8": round(mean_v8, 4),
        "mean_ap_blend": round(mean_blend, 4),
        "blend_wins": wins,
        "groups": results,
    }, indent=1))


if __name__ == "__main__":
    main()
