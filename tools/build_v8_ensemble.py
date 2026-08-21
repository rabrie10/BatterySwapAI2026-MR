"""Package the fitted V8 hybrid and V7 Wiener models for submission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.ensemble import ProbabilityBlendModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", type=Path, default=Path("models/v8_hybrid.joblib"))
    parser.add_argument("--wiener", type=Path, default=Path("models/v7_wiener.joblib"))
    parser.add_argument("--out", type=Path, default=Path("models/v8_ensemble.joblib"))
    parser.add_argument("--hybrid-weight", type=float, default=0.25)
    parser.add_argument("--wiener-volatility-scale", type=float, default=1.4)
    args = parser.parse_args()

    hybrid = joblib.load(args.hybrid)
    wiener = joblib.load(args.wiener)
    wiener.volatility_scale = float(args.wiener_volatility_scale)
    model = ProbabilityBlendModel(
        left=hybrid,
        right=wiener,
        left_weight=float(args.hybrid_weight),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(
        f"wrote {args.out} (hybrid_weight={model.left_weight}, "
        f"wiener_volatility_scale={wiener.volatility_scale})"
    )


if __name__ == "__main__":
    main()
