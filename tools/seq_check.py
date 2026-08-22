"""Integration check: the SeqModel wrapper inside the real forecaster path.

Builds a throwaway bundle (untrained net, real deployment snapshot), runs the
actual ``HazardForecaster`` + ``OofHazardModel`` over the first N scenarios,
and verifies every row the forecaster produces is identified by the feature
fingerprint (a KeyError would abort). This is the only novel plumbing in the
gate-(d) path; everything downstream is the shipped planner.

    python tools/seq_check.py --limit 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.forecaster import HazardForecaster
from bsai.seq_head import SeqModel, SeqQuantileNet
from bsai.validation import OofHazardModel

DEFAULT_WORK = Path(
    os.environ.get(
        "SEQ_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\seq",
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_WORK / "seq_pack.joblib")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset/train")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    torch.set_num_threads(3)
    pack = joblib.load(args.pack)
    train, deploy = pack["train"], pack["deploy"]

    net = SeqQuantileNet()
    net.eval()
    shim = SeqModel(
        net=net,
        windows=deploy["windows"],
        margins=deploy["margin"].astype(np.float32),
        key_index={key: i for i, key in enumerate(deploy["keys"])},
        climatology=train["climatology"],
    )
    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    by_building = {str(b): shim for b in devices["building_id"].unique()}
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=by_building,
            building_of=building_of,
            climatology=train["climatology"],
        )
    )

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    started = time.time()
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"])
        horizon = int(scenario["settings"].planning_window_days)
        forecast = forecaster.predict(
            cut,
            locs,
            prediction_origin=start,
            horizon_days=horizon,
            evaluation_observation_end=pd.to_datetime(locs["end_time"]).max(),
        )
        p = forecaster.last_probabilities
        print(
            f"  {scenario['name']:>5}  rows={len(p)}  cold={forecaster.last_cold_start}  "
            f"sum_p42={float(forecaster.last_expected_due):.2f}  "
            f"curves={len(forecast.curves)}  {time.time()-started:.0f}s",
            flush=True,
        )
    print("integration check PASSED: every forecaster row identified")


if __name__ == "__main__":
    main()
