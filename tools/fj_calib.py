"""Diagnostics 3 and 4: is V8's sigma right, and is its passage law calibrated?

Diagnostic 1 found the 0-0.03 V band sub-diffusive against a smoothed-Brownian
null (variance ratio 0.67 [0.49, 0.88] against the null's 1.64 [1.26, 2.11],
non-overlapping) while every band above 0.05 V is Brownian-consistent. That is a
statement about the *process*. It only matters if V8 has not already absorbed it.

It might have. V8 does not assume `sigma ~ sqrt(h)`: `scatter` is a regression
with the horizon as a monotone input, fitted on the observed absolute residual at
each horizon, so a sub-diffusive band is in principle learnable. What it *is*
fitted on matters though -- windows required to end before the crossing, which
near the threshold keeps only the devices that did not cross.

So:

* **(3)** V8's predicted 42-day sigma against the empirical 42-day dispersion in
  the same state, by margin band, and separately on the six repeat false
  positives.
* **(4)** V8's first-passage probability against the realised 42-day crossing
  rate, out of fold by building, on historical cutoffs rather than only the
  19,890 scenario rows -- roughly an order of magnitude more evidence about the
  law itself.

    python tools/fj_calib.py --stride 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES  # noqa: E402
from bsai.wiener import MIN_SIGMA, first_passage_probability  # noqa: E402
from tools.fj_increments import build  # noqa: E402

VOLTAGE = FEATURE_NAMES.index("voltage")
EOL_THRESHOLD = 2.4
HORIZON = 42
BANDS = ((0.0, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9.0))
ZOMBIES = (
    "d_b5b678a3f79f", "d_3d26e12378f1", "d_c9a2ce794b68",
    "d_a85bae19463d", "d_d9d695df1683", "d_d4b4272d5229",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_calib.json"))
    args = parser.parse_args()

    started = time.time()
    frame, cache, _ratio = build(args.dataset, args.stride)
    margin = frame.features[:, VOLTAGE].astype(float) - EOL_THRESHOLD

    # The realised 42-day fate of every cutoff, censor-correct: known only when
    # the device crosses inside the window or is observed all the way through.
    crossed = np.zeros(frame.features.shape[0], dtype=bool)
    known = np.zeros(frame.features.shape[0], dtype=bool)
    realised = np.full(frame.features.shape[0], np.nan)
    for device, series in cache.devices.items():
        rows = np.flatnonzero(frame.device == device)
        if rows.size == 0:
            continue
        voltage = series.smooth_voltage
        valid = np.flatnonzero(~np.isnan(voltage))
        if valid.size == 0:
            continue
        last = int(valid[-1])
        cross = int(frame.crossing[rows[0]])
        for row in rows:
            t = int(frame.cutoff[row])
            if cross >= 0 and t < cross <= t + HORIZON:
                crossed[row] = True
                known[row] = True
            elif (cross < 0 or cross > t + HORIZON) and t + HORIZON <= last:
                known[row] = True
            if t + HORIZON <= last and np.isfinite(voltage[t + HORIZON]):
                realised[row] = (voltage[t] - voltage[t + HORIZON])

    # V8's own drift, sigma and passage probability, out of fold by building.
    bundle = joblib.load(args.folds)
    drop = np.zeros(frame.features.shape[0])
    sigma = np.zeros(frame.features.shape[0])
    for building in np.unique(frame.building):
        model = bundle["by_building"].get(building)
        if model is None:
            continue
        mask = frame.building == building
        design = np.hstack([
            frame.features[mask],
            np.full((int(mask.sum()), 1), float(HORIZON), dtype=np.float32),
        ])
        drop[mask] = np.maximum(model.drift.predict(design), 0.0)
        sigma[mask] = (
            np.maximum(model.scatter.predict(design), MIN_SIGMA)
            * np.sqrt(np.pi / 2.0)
            * model.volatility_scale
        )
    probability = first_passage_probability(margin, drop, sigma)
    print(f"  {frame.features.shape[0]} cutoffs, {int(known.sum())} with a known "
          f"42-day fate, {int(crossed.sum())} crossings ({time.time() - started:.0f}s)")

    report: dict = {}

    print()
    print("=== 3. Is V8's sigma right?  std((realised - drop)/sigma), 1.00 = correct ===")
    print("    Comparing an unconditional band standard deviation against a")
    print("    conditional sigma inflates the ratio by Jensen -- HANDOVER trap 5,")
    print("    which cost a whole build cycle once already. This is the statistic")
    print("    that avoids it.")
    standardised = (realised - drop) / np.maximum(sigma, MIN_SIGMA)
    print(f"{'band':>12} {'n':>7} {'dev':>5} {'std(z)':>8} {'mean(z)':>9} "
          f"{'rms sigma':>10} {'rms resid':>10}")
    rows = []
    for low, high in BANDS:
        inside = (margin >= low) & (margin < high) & np.isfinite(realised)
        if inside.sum() < 50:
            continue
        entry = {
            "band": f"{low}-{high}", "n": int(inside.sum()),
            "devices": int(np.unique(frame.device[inside]).size),
            "std_z": round(float(standardised[inside].std()), 3),
            "mean_z": round(float(standardised[inside].mean()), 3),
            "rms_sigma": round(float(np.sqrt((sigma[inside] ** 2).mean())), 5),
            "rms_residual": round(
                float(np.sqrt(((realised[inside] - drop[inside]) ** 2).mean())), 5
            ),
        }
        rows.append(entry)
        print(f"{entry['band']:>12} {entry['n']:>7} {entry['devices']:>5} "
              f"{entry['std_z']:>8.3f} {entry['mean_z']:>9.3f} "
              f"{entry['rms_sigma']:>10.4f} {entry['rms_residual']:>10.4f}")
    report["sigma"] = rows

    zombie = np.isin(frame.device, ZOMBIES) & np.isfinite(realised)
    if zombie.sum() > 10:
        print(f"{'ZOMBIES':>12} {int(zombie.sum()):>7} "
              f"{int(np.unique(frame.device[zombie]).size):>5} "
              f"{standardised[zombie].std():>8.3f} {standardised[zombie].mean():>9.3f} "
              f"{np.sqrt((sigma[zombie] ** 2).mean()):>10.4f} "
              f"{np.sqrt(((realised[zombie] - drop[zombie]) ** 2).mean()):>10.4f}")
        report["zombies"] = {
            "n": int(zombie.sum()),
            "std_z": round(float(standardised[zombie].std()), 3),
            "mean_z": round(float(standardised[zombie].mean()), 3),
        }

    print()
    print("=== 4. First-passage probability against the realised crossing rate ===")
    print(f"{'band':>12} {'n':>7} {'dev':>5} {'V8 mean p':>10} {'realised':>9} {'ratio':>7}")
    rows = []
    for low, high in BANDS:
        inside = (margin >= low) & (margin < high) & known
        if inside.sum() < 50:
            continue
        predicted = float(probability[inside].mean())
        observed = float(crossed[inside].mean())
        entry = {
            "band": f"{low}-{high}", "n": int(inside.sum()),
            "devices": int(np.unique(frame.device[inside]).size),
            "predicted": round(predicted, 4), "realised": round(observed, 4),
            "ratio": round(predicted / max(observed, 1e-6), 2),
        }
        rows.append(entry)
        print(f"{entry['band']:>12} {entry['n']:>7} {entry['devices']:>5} "
              f"{entry['predicted']:>10.4f} {entry['realised']:>9.4f} "
              f"{entry['ratio']:>7.2f}")
    report["passage"] = rows

    print()
    print("by predicted probability, all margins:")
    print(f"{'bucket':>12} {'n':>7} {'dev':>5} {'V8 mean p':>10} {'realised':>9} {'ratio':>7}")
    buckets = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2),
               (0.2, 0.4), (0.4, 0.7), (0.7, 1.01)]
    rows = []
    for low, high in buckets:
        inside = (probability >= low) & (probability < high) & known
        if inside.sum() < 30:
            continue
        predicted = float(probability[inside].mean())
        observed = float(crossed[inside].mean())
        entry = {
            "bucket": f"{low}-{high}", "n": int(inside.sum()),
            "devices": int(np.unique(frame.device[inside]).size),
            "predicted": round(predicted, 4), "realised": round(observed, 4),
            "ratio": round(predicted / max(observed, 1e-6), 2),
        }
        rows.append(entry)
        print(f"{entry['bucket']:>12} {entry['n']:>7} {entry['devices']:>5} "
              f"{entry['predicted']:>10.4f} {entry['realised']:>9.4f} "
              f"{entry['ratio']:>7.2f}")
    report["reliability"] = rows

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
