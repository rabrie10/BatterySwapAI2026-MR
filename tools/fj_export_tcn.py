"""Convert the trained Torch folds into the artifact the container actually loads.

Three things change on the way out, each for a reason `bsai/sequence.py`
documents: plain JSON instead of `.pt` because `.gitattributes` sends `*.pt`
through Git LFS; every fold in one file because fold routing is meaningless on
an unseen building; and the temperature statistics travel with their own fold.

The exporter refuses to write a file whose NumPy forward pass disagrees with
Torch, so the artifact cannot drift from the measured model.

    python tools/fj_export_tcn.py --model outputs/fj_tcn.pt \
                                  --out models/sequence_tcn.json
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

from bsai.sequence import SequenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/fj_tcn.pt"))
    parser.add_argument("--out", type=Path, default=Path("models/sequence_tcn.json"))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    import torch

    from tools.fj_tcn import make_model

    payload = torch.load(args.model, weights_only=False)
    width = payload["width"]
    groups = sorted(payload["states"])
    folds = []
    for group in groups:
        state = payload["states"][group]
        centre, spread = payload["stats"][group]
        folds.append({
            "fold": int(group),
            "temperature_centre": float(centre),
            "temperature_spread": float(spread),
            "tensors": {
                name: {"shape": list(tensor.shape),
                       "data": tensor.detach().numpy().astype(np.float64).ravel().tolist()}
                for name, tensor in state.items()
            },
        })
    per_fold = sum(t.numel() for t in payload["states"][groups[0]].values())
    document = {
        "format": SequenceModel.version,
        "history": payload["history"],
        "horizons": payload["horizons"],
        "quantiles": payload["quantiles"],
        "width": width,
        "stride_trained_at": payload["stride"],
        "folds": folds,
        "parameters_per_fold": int(per_fold),
        "source": str(args.model),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document))
    size = args.out.stat().st_size / 1e6
    print(f"{len(folds)} folds, {per_fold} parameters each -> {args.out} ({size:.2f} MB)")

    # The artifact is only valid if NumPy reproduces Torch on real-shaped input.
    model = SequenceModel.load(args.out)
    TCN = make_model(torch)
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(8, 5, payload["history"])).astype(np.float64)
    worst = 0.0
    for index, group in enumerate(groups):
        torch_model = TCN(width=width)
        torch_model.load_state_dict(payload["states"][group])
        torch_model.eval()
        with torch.no_grad():
            expected = torch_model(torch.from_numpy(windows.astype(np.float32))).numpy()
        got = model._forward_one(windows, model.folds[index])
        worst = max(worst, float(np.abs(expected - got).max()))
    print(f"worst |numpy - torch| over {len(groups)} folds: {worst:.3e}")
    if worst > args.tolerance:
        args.out.unlink()
        raise SystemExit(f"forward passes disagree by {worst:.3e}; artifact removed")
    print("numpy and torch agree; artifact kept")


if __name__ == "__main__":
    main()
