"""Fit the discriminative head and assemble the blended model.

The head is trained on the cached scenario frame -- one row per (scenario,
device) the forecaster is actually asked about -- because that is the population
the model is deployed on, and the strided training grid is not a substitute for
it (``HANDOVER.md`` trap 2).

Folds are read back from the Wiener bundle by object identity, so a head never
sees its own building and the two halves of the blend share a fold partition.

    python tools/build_scenario_frame.py          # once, ~13 min
    python tools/train_blend.py                   # ~1 min
    python tools/fit_calibration.py --folds outputs/v9_blend_folds.joblib \\
        --model models/v9_blend.joblib --volatility-scale 1.0
    python tools/validate_v6.py --folds outputs/v9_blend_folds.joblib \\
        --model models/v9_blend.joblib --volatility-scale 1.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.blend import DECISION_HORIZON, BlendedModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--wiener-folds", type=Path,
                        default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--wiener-model", type=Path,
                        default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--folds-out", type=Path,
                        default=Path("outputs/v9_blend_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("models/v9_blend.joblib"))
    parser.add_argument("--report", type=Path,
                        default=Path("docs/v9_blend_report.json"))
    parser.add_argument("--weight", type=float, default=0.5,
                        help="exponent on the head in the geometric mean")
    args = parser.parse_args()

    started = time.time()
    z = np.load(args.frame, allow_pickle=True)
    features = z["features"]
    remaining = z["remaining"]
    due = z["due"].astype(int)
    days_to_eol = z["days_to_eol"]
    building = np.asarray([str(b) for b in z["building"]])
    wiener_probability = z["probability"]

    bundle = joblib.load(args.wiener_folds)
    seen: dict[int, int] = {}
    fold_of: dict[str, int] = {}
    for name, model in bundle["by_building"].items():
        key = id(model)
        seen.setdefault(key, len(seen))
        fold_of[str(name)] = seen[key]
    fold = np.asarray([fold_of[b] for b in building])
    print(f"{len(features)} rows, {int(due.sum())} due at 42d, {len(seen)} folds")
    for horizon in (14, 42, 91, 126):
        label = (days_to_eol <= horizon) & (days_to_eol <= remaining)
        print(f"  positives at h={horizon:3d}: {int(label.sum())}")

    # Out-of-fold heads, so the number below is the one validate_v6 reproduces.
    oof_head = np.zeros(len(features))
    heads_by_fold: dict[int, list] = {}
    for index in sorted(seen.values()):
        held = fold == index
        heads = BlendedModel.fit_heads(
            features[~held], remaining[~held], days_to_eol[~held]
        )
        heads_by_fold[index] = heads
        design = BlendedModel.head_design(
            features[held], remaining[held], DECISION_HORIZON
        )
        total = np.zeros(int(held.sum()))
        for head in heads:
            total += np.log(np.clip(head.predict_proba(design)[:, 1], 1e-12, 1.0))
        oof_head[held] = np.exp(total / len(heads))
        print(f"  fold {index}: {len(heads)} heads fitted, "
              f"{time.time() - started:.0f}s", flush=True)

    blended = (
        np.clip(wiener_probability, 1e-12, 1.0) ** (1.0 - args.weight)
        * np.clip(oof_head, 1e-12, 1.0) ** args.weight
    )
    metrics = {
        name: {
            "auc": round(float(roc_auc_score(due, value)), 4),
            "pr_auc": round(float(average_precision_score(due, value)), 4),
        }
        for name, value in [
            ("wiener", wiener_probability), ("head", oof_head), ("blend", blended)
        ]
    }
    print(json.dumps(metrics, indent=2))

    by_building: dict[str, BlendedModel] = {}
    for index in sorted(seen.values()):
        names = [b for b, f in fold_of.items() if f == index]
        model = BlendedModel(
            wiener=copy.deepcopy(bundle["by_building"][names[0]]),
            heads=heads_by_fold[index],
            weight=args.weight,
        )
        for name in names:
            by_building[name] = model
    args.folds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"by_building": by_building, "climatology": bundle["climatology"]},
        args.folds_out,
    )

    production = BlendedModel(
        wiener=joblib.load(args.wiener_model),
        heads=BlendedModel.fit_heads(features, remaining, days_to_eol),
        weight=args.weight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production, args.out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"weight": args.weight, "metrics": metrics}, indent=2)
    )
    print(f"wrote {args.folds_out} and {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
