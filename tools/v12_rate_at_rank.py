"""Rate-at-rank: realized due rate by within-scenario probability rank, OOF.

The number that matters for the mid-year ranking failure: the shipped cens
model's top-12 realized rate is 0.59-0.60 in the opening block, 0.214 in the
mid block (scenarios 16-31) and 0.26-0.29 late (docs/V11_TRANSFER_FINDINGS.md).
This reproduces that table for any fold bundle so V12 can be compared on the
same measurement.

    set BSAI_FEATURE_VARIANT if the bundle needs it, or pass --feature-variant
    python tools/v12_rate_at_rank.py --folds outputs/v12_folds.joblib \
        --feature-variant invariant --report outputs/v12_rate_at_rank.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai import features as feature_lib
from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel

BLOCKS = ((0, 16, "early"), (16, 32, "mid"), (32, 48, "late"))
TOP_KS = (5, 8, 12, 15)


def collect(dataset: Path, folds: Path) -> pd.DataFrame:
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(folds)
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    locations, timeseries, eol_times, scenarios = load_dataset(dataset)
    rows: list[pd.DataFrame] = []
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        start = pd.Timestamp(scenario["start_time"])
        horizon = int(scenario["settings"].planning_window_days)
        forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        probability = forecaster.last_probabilities
        battery_ids = locs["battery"].astype(str).to_numpy()
        due = not_dead.reindex(battery_ids)
        is_due = (due.notna() & (due <= start + pd.Timedelta(days=horizon))).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "scenario_index": index,
                    "battery": battery_ids,
                    "predicted": probability.reindex(battery_ids).to_numpy(),
                    "due": is_due.astype(float),
                }
            )
        )
        print(f"  {scenario['name']:>5}", flush=True)
    return pd.concat(rows, ignore_index=True)


def rate_table(frame: pd.DataFrame, max_rank: int = 20) -> dict:
    """Per-rank and cumulative realized rates, per block of scenario thirds."""
    per_scenario: dict[int, np.ndarray] = {}
    for s, group in frame.groupby("scenario_index"):
        order = np.argsort(-group["predicted"].to_numpy(), kind="stable")
        per_scenario[int(s)] = group["due"].to_numpy()[order]

    out: dict = {"blocks": {}}
    for lo, hi, label in BLOCKS:
        members = [per_scenario[s] for s in per_scenario if lo <= s < hi]
        if not members:
            continue
        block: dict = {"n_scenarios": len(members)}
        block["rate_at_rank"] = [
            round(
                float(np.mean([d[r] for d in members if d.size > r])), 3
            )
            for r in range(max_rank)
        ]
        for k in TOP_KS:
            block[f"top_{k}_realized"] = round(
                float(np.mean([d[:k].mean() for d in members if d.size >= k])), 3
            )
        block["realized_per_scenario"] = round(
            float(np.mean([d.sum() for d in members])), 2
        )
        out["blocks"][label] = block
    all_members = list(per_scenario.values())
    out["pooled"] = {
        f"top_{k}_realized": round(
            float(np.mean([d[:k].mean() for d in all_members if d.size >= k])), 3
        )
        for k in TOP_KS
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v12_folds.joblib"))
    parser.add_argument(
        "--feature-variant",
        choices=sorted(feature_lib.FEATURE_VARIANTS),
        default=None,
        help="feature set the fold models were trained on",
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/v12_rate_at_rank.json"))
    args = parser.parse_args()

    if args.feature_variant is not None:
        feature_lib.set_feature_variant(args.feature_variant)

    frame = collect(args.dataset, args.folds)
    table = rate_table(frame)
    table["folds"] = str(args.folds)
    table["feature_variant"] = feature_lib.active_feature_variant()

    print(json.dumps(table, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(table, indent=2))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
