"""What does the drift model actually use?

The within-day ablation crashed when it was first attempted, so the split
between "within-day features" and "Wiener structure" in the V7 gain was never
measured. Permutation importance answers it more cheaply than a retrain, and it
also settles a second question the seasonal finding raised: the model
under-predicts failures in the winter scenarios by a factor of two, and it
already has seasonal features -- so is it using them?

Permutation is done on a held-out subsample, feature by feature, against the
drift model's own squared error. Grouped variants are reported too, because
features within a group are correlated and permuting one at a time understates
a group that carries the signal jointly.

    python tools/importance.py
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

from bsai.features import FEATURE_NAMES

GROUPS = {
    "within-day shape": lambda n: n.startswith(("beta_", "v_std_", "v_range_", "t_range_")),
    "seasonal outlook": lambda n: n.startswith(("temp_outlook", "voltage_outlook", "season_")),
    "level": lambda n: n in {"voltage", "voltage_compensated", "voltage_max", "voltage_min", "drawdown", "range_90"},
    "slope": lambda n: n.startswith(("slope_", "curvature", "slope_ratio")),
    "threshold history": lambda n: n.startswith(("days_below_", "crossing_")),
    "knee": lambda n: n.startswith("knee_"),
    "temperature": lambda n: n.startswith("temp_") and not n.startswith("temp_outlook"),
    "housekeeping": lambda n: n in {"staleness", "observations", "gap_fraction_90", "age_days"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--design", type=Path, default=Path("outputs/v7_design.npz"))
    parser.add_argument("--sample", type=int, default=60000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_importance.json"))
    args = parser.parse_args()

    if not args.design.exists():
        raise SystemExit(
            f"{args.design} not found -- rerun tools/train_wiener.py, which now saves "
            "a held-out design block for exactly this measurement"
        )
    blob = np.load(args.design)
    design, target = blob["design"], blob["target"]

    model = joblib.load(args.model)
    rng = np.random.default_rng(20260821)
    if design.shape[0] > args.sample:
        pick = rng.choice(design.shape[0], args.sample, replace=False)
        design, target = design[pick], target[pick]

    def error(matrix: np.ndarray) -> float:
        return float(np.mean((model.drift.predict(matrix) - target) ** 2))

    base = error(design)
    names = list(FEATURE_NAMES) + ["effective_horizon"]
    single: dict[str, float] = {}
    for index, name in enumerate(names):
        losses = []
        for _ in range(args.repeats):
            shuffled = design.copy()
            shuffled[:, index] = shuffled[rng.permutation(design.shape[0]), index]
            losses.append(error(shuffled))
        single[name] = float(np.mean(losses) / base - 1.0)

    grouped: dict[str, float] = {}
    for label, predicate in GROUPS.items():
        columns = [i for i, n in enumerate(names) if predicate(n)]
        if not columns:
            continue
        losses = []
        for _ in range(args.repeats):
            shuffled = design.copy()
            order = rng.permutation(design.shape[0])
            for column in columns:
                shuffled[:, column] = shuffled[order, column]
            losses.append(error(shuffled))
        grouped[label] = float(np.mean(losses) / base - 1.0)

    print(f"baseline drift MSE {base:.6e} on {design.shape[0]} rows\n")
    print("=== grouped (permuted together) ===")
    for label, value in sorted(grouped.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(max(0.0, value) * 200)
        print(f"  {label:20s} {value:+8.4f}  {bar}")
    print("\n=== top 20 individual ===")
    for name, value in sorted(single.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {name:34s} {value:+8.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"baseline_mse": base, "grouped": grouped, "single": single}, indent=2)
    )


if __name__ == "__main__":
    main()
