"""Extract trajectory dynamics at every scenario cutoff, once, and cache them.

The 64 shipped features and the Wiener state are both functions of *where* the
device is. This adds signals about *how it got there and whether it is still
moving* -- the distinction between a stable low-voltage plateau and an
irreversible terminal transition, which `bsai/terminality.py` explains.

Two passes are cached so nothing after this needs the 8.5 M-row timeseries:

* ``outputs/fj_series.npz``      -- every device's smoothed daily voltage and
  temperature grid, which is the expensive part (one full smoothing pass);
* ``outputs/fj_terminality.npz`` -- the twenty trajectory signals at each of the
  19,890 (scenario, battery) rows, aligned to ``outputs/v9_frame.npz`` row for
  row so the two can be concatenated.

    python tools/fj_terminality.py --rebuild-series
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_dataset  # noqa: E402

from bsai.smoothing import SmoothingCache  # noqa: E402
from bsai.terminality import NAMES, features_at  # noqa: E402
from tools.fj_frame import load_frame  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")


def build_series(dataset: Path, out: Path) -> dict:
    started = time.time()
    locations, timeseries, eol_times, scenarios = load_dataset(dataset)
    cache = SmoothingCache()
    cache.update(timeseries)
    print(f"smoothed {len(cache.devices)} devices in {time.time() - started:.0f}s")
    payload: dict[str, np.ndarray] = {}
    for device, series in cache.devices.items():
        payload[f"v::{device}"] = np.asarray(series.smooth_voltage, dtype=np.float32)
        payload[f"t::{device}"] = np.asarray(series.smooth_temperature, dtype=np.float32)
        payload[f"o::{device}"] = np.asarray([series.origin], dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return payload


def load_series(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    devices = {key[3:] for key in data.files if key.startswith("v::")}
    return {
        device: (
            data[f"v::{device}"].astype(float),
            data[f"t::{device}"].astype(float),
            int(data[f"o::{device}"][0]),
        )
        for device in devices
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_terminality.npz"))
    parser.add_argument("--rebuild-series", action="store_true")
    args = parser.parse_args()

    if args.rebuild_series or not args.series.exists():
        build_series(args.dataset, args.series)
    series = load_series(args.series)
    print(f"loaded {len(series)} smoothed device series")

    frame = load_frame(args.frame)
    # The scenario start ordinal, recovered from the dataset the frame was built
    # from. The cutoff index is the same one HazardForecaster uses.
    _, _, _, scenarios = load_dataset(args.dataset)
    starts = np.asarray(
        [
            int((pd.Timestamp(s["start_time"]).normalize() - _EPOCH) / pd.Timedelta(days=1))
            for s in scenarios
        ]
    )

    started = time.time()
    out = np.full((frame.features.shape[0], len(NAMES)), np.nan)
    misses = 0
    for row in range(frame.features.shape[0]):
        entry = series.get(frame.battery[row])
        if entry is None:
            misses += 1
            continue
        voltage, temperature, origin = entry
        index = int(starts[frame.scenario[row]] - origin)
        if index < 0:
            misses += 1
            continue
        index = min(index, voltage.size - 1)
        out[row] = features_at(voltage, temperature, index)
        if row and row % 4000 == 0:
            print(f"  {row}/{frame.features.shape[0]} ({time.time() - started:.0f}s)",
                  flush=True)

    finite = np.isfinite(out).mean(axis=0)
    print(f"\n{misses} rows without a usable series")
    print("coverage per signal:")
    for name, share in zip(NAMES, finite):
        print(f"  {name:>28} {share:6.1%}")
    np.savez_compressed(args.out, features=out.astype(np.float32),
                        names=np.asarray(NAMES))
    print(f"\nwrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
