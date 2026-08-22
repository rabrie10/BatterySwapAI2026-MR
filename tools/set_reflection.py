"""Write a folds/model variant with a chosen reflection weight.

    python tools/set_reflection.py --weight 0.0 \
        --folds outputs/v7_folds.joblib --folds-out outputs/v7_folds_w0.joblib \
        --model models/v7_wiener.joblib --model-out models/v7_wiener_w0.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--folds-out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    bundle = joblib.load(args.folds)
    for model in bundle["by_building"].values():
        model.reflection_weight = args.weight
    joblib.dump(bundle, args.folds_out)
    print(f"wrote {args.folds_out} with reflection_weight={args.weight}")

    if args.model is not None and args.model_out is not None:
        shipped = joblib.load(args.model)
        shipped.reflection_weight = args.weight
        joblib.dump(shipped, args.model_out)
        print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
