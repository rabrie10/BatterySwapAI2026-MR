"""Pack everything the sequence head needs: training windows + deployment rows.

Two populations, built by two different code paths on purpose:

* TRAINING side (full-history caches, causal by construction): the stride-4
  cutoff frame from ``bsai.hazard.build_training_frame`` -- the exact frame the
  incumbent censored GBDT was gated on (88k cutoffs, PR-AUC 0.4706) -- plus a
  concatenated per-device channel bank for window extraction and the
  censor-aware increment targets of ``bsai.wiener.build_increment_targets``,
  reproduced per (row, horizon) so each window knows its cutoff row.

* DEPLOYMENT side (incremental caches, one scenario at a time): the planner's
  forecaster evaluates each device at the last day its *incremental* smoothing
  grid reaches, which is not always reproducible from the full-history grid
  (the scenario cut can split a calendar day's readings). So this tool runs
  the real incremental loop over the 48 scenarios, replicates the forecaster's
  row logic exactly (index clamp, feature_row eligibility), and captures for
  every (scenario, alive device) row: the feature fingerprint the wrapper will
  see at plan time, the 120-day channel window, and the margin. Gate (c) and
  the planner gate (d) both run off this snapshot.

    python tools/seq_pack.py
"""

from __future__ import annotations

import argparse
import json
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

from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from bsai.features import (
    FEATURE_NAMES,
    DeviceView,
    FeatureContext,
    feature_row,
    fleet_climatology,
)
from bsai.hazard import build_training_frame
from bsai.margin import EOL_THRESHOLD
from bsai.rawdaily import RawDailyCache
from bsai.shape import ShapeCache, align_to
from bsai.smoothing import SmoothingCache
from bsai.seq_head import (
    KEY_COLUMNS,
    SEQ_FIT_HORIZONS,
    SEQ_WINDOW,
    TARGET_SCALE,
    build_channels,
    pad_channels,
    window_from_channels,
)

_EPOCH = pd.Timestamp("1970-01-01")
DECISION_HORIZON = 42

DEFAULT_WORK = Path(
    os.environ.get(
        "SEQ_WORK",
        r"C:\Users\MAHDIN\AppData\Local\Temp\claude"
        r"\C--Users-MAHDIN--vscode-BSAI-challenge-BatterySwapAI2026-MR"
        r"\2356bc38-2fa7-45f2-9b8b-168cd02b20e7\scratchpad\seq",
    )
)

VOLTAGE_COL = FEATURE_NAMES.index("voltage")
BETA30_COL = FEATURE_NAMES.index("beta_30")


def _ordinal(value) -> int:
    return int((pd.Timestamp(value).normalize() - _EPOCH) / pd.Timedelta(days=1))


def _aligned(series_origin: int, size: int, origin: int, values: np.ndarray) -> np.ndarray:
    """Place ``values`` (own grid at ``origin``) onto the smoothing grid."""
    out = np.full(size, np.nan)
    offset = origin - series_origin
    source_start = max(0, -offset)
    target_start = max(0, offset)
    span = min(values.shape[0] - source_start, size - target_start)
    if span > 0:
        out[target_start : target_start + span] = values[
            source_start : source_start + span
        ]
    return out


def device_channels(cache, shape_cache, raw_cache, device_id: str) -> np.ndarray:
    """Unscaled channel matrix for one device on its smoothing grid."""
    series = cache.devices[device_id]
    size = len(series)
    shape = shape_cache.devices.get(device_id) if shape_cache is not None else None
    beta = (
        _aligned(series.origin, size, shape.origin, shape.beta)
        if shape is not None
        else None
    )
    raw = raw_cache.devices.get(device_id) if raw_cache is not None else None
    raw_daily = (
        _aligned(series.origin, size, raw.origin, raw.median)
        if raw is not None
        else None
    )
    return build_channels(
        series.smooth_voltage, series.smooth_temperature, beta, raw_daily
    )


# ---------------------------------------------------------------------------
# training side
# ---------------------------------------------------------------------------

def build_train_side(dataset: Path, stride: int) -> dict:
    started = time.time()
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol = pd.to_datetime(
        pd.read_csv(dataset / "eol_times.csv").set_index("device_id")["end_time"]
    )
    observation_end = devices.set_index("device_id")["end_time"]

    print("full-history caches...", flush=True)
    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    shape_cache = ShapeCache()
    shape_cache.update(raw)
    raw_cache = RawDailyCache()
    raw_cache.update(raw)
    del raw
    print(f"  {len(cache.devices)} devices, {time.time()-started:.0f}s", flush=True)

    eol_index: dict[str, int | None] = {}
    observation_index: dict[str, int] = {}
    for device_id, series in cache.devices.items():
        moment = eol.get(device_id)
        eol_index[device_id] = (
            None if pd.isna(moment) else _ordinal(moment) - series.origin
        )
        end = observation_end.get(device_id)
        observation_index[device_id] = (
            (series.origin + len(series) - 1)
            if pd.isna(end)
            else _ordinal(end) - series.origin
        )

    print(f"stride-{stride} frame...", flush=True)
    frame = build_training_frame(
        cache,
        eol_index,
        building_of,
        observation_index,
        shape_cache=shape_cache,
        stride=stride,
    )
    print(f"  {len(frame)} cutoffs, {time.time()-started:.0f}s", flush=True)

    print("channel bank...", flush=True)
    device_ids = sorted(cache.devices)
    device_of = {d: i for i, d in enumerate(device_ids)}
    blocks, offsets = [], np.zeros(len(device_ids), dtype=np.int64)
    total = 0
    for i, device_id in enumerate(device_ids):
        padded = pad_channels(device_channels(cache, shape_cache, raw_cache, device_id))
        offsets[i] = total
        total += padded.shape[0]
        blocks.append(padded.astype(np.float16))
    bank_data = np.concatenate(blocks, axis=0)
    del blocks
    print(
        f"  bank {bank_data.shape[0]} rows x {bank_data.shape[1]} channels "
        f"({bank_data.nbytes/1e6:.0f} MB f16), {time.time()-started:.0f}s",
        flush=True,
    )

    # Row-level arrays for training and the OOF gate.
    frame_device_index = np.asarray([device_of[d] for d in frame.device], dtype=np.int32)
    margins = frame.features[:, VOLTAGE_COL].astype(np.float64) - EOL_THRESHOLD
    truth = (
        (frame.crossing >= 0)
        & (frame.crossing > frame.cutoff)
        & ((frame.crossing - frame.cutoff) <= DECISION_HORIZON)
        & (frame.crossing <= frame.observation_end)
    ).astype(np.int8)
    decision_horizon = np.clip(
        np.minimum(DECISION_HORIZON, frame.observation_end - frame.cutoff), 0.0, None
    ).astype(np.float32)

    # Consistency: the bank's forward-filled margin at each cutoff must equal
    # the feature row's voltage (float32) to within float16 storage error.
    probe = np.random.default_rng(0).choice(len(frame), size=min(4000, len(frame)), replace=False)
    bank_margin = bank_data[
        offsets[frame_device_index[probe]] + frame.cutoff[probe] + SEQ_WINDOW - 1, 0
    ].astype(np.float64)
    gap = np.abs(bank_margin - margins[probe])
    print(f"  bank-vs-feature margin max gap {gap.max():.5f} (f16 quantum ~5e-4)", flush=True)
    assert gap.max() < 3e-3, "channel bank misaligned with the training frame"

    print("censor-aware increment targets...", flush=True)
    window_row, window_horizon, window_target = [], [], []
    order = np.argsort(frame.device, kind="stable")
    margins_by_device = {
        d: cache.devices[d].smooth_voltage - EOL_THRESHOLD for d in cache.devices
    }
    for h_id, horizon in enumerate(SEQ_FIT_HORIZONS):
        start = 0
        while start < order.size:
            stop = start
            device_id = frame.device[order[start]]
            while stop < order.size and frame.device[order[stop]] == device_id:
                stop += 1
            block = order[start:stop]
            start = stop
            margin = margins_by_device.get(device_id)
            if margin is None:
                continue
            crossing = int(frame.crossing[block[0]])
            last = int(frame.last_observed[block[0]])
            cutoffs = frame.cutoff[block]
            ends = cutoffs + horizon
            usable = (ends <= last) & (cutoffs >= 0)
            if not usable.any():
                continue
            chosen = block[usable]
            here = margin[cutoffs[usable]]
            there = margin[ends[usable]]
            finite = np.isfinite(here) & np.isfinite(there)
            if not finite.any():
                continue
            drop = here[finite] - there[finite]
            if crossing >= 0:
                crossed = (
                    (cutoffs[usable][finite] < crossing)
                    & (ends[usable][finite] >= crossing)
                )
                drop = np.where(crossed, np.maximum(drop, here[finite]), drop)
            window_row.append(chosen[finite].astype(np.int32))
            window_horizon.append(np.full(int(finite.sum()), h_id, dtype=np.int8))
            window_target.append((drop * TARGET_SCALE).astype(np.float32))
    window_row = np.concatenate(window_row)
    window_horizon = np.concatenate(window_horizon)
    window_target = np.concatenate(window_target)
    print(
        f"  {window_row.size} windows, mean fall {window_target.mean()/TARGET_SCALE:.5f} V, "
        f"{time.time()-started:.0f}s",
        flush=True,
    )

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    return {
        "frame": frame,
        "device_ids": device_ids,
        "bank_data": bank_data,
        "bank_offsets": offsets,
        "frame_device_index": frame_device_index,
        "frame_margin": margins.astype(np.float32),
        "truth": truth,
        "decision_horizon": decision_horizon,
        "window_row": window_row,
        "window_horizon": window_horizon,
        "window_target": window_target,
        "climatology": climatology,
        "building_sizes": devices.groupby("building_id")["device_id"].count().to_dict(),
        "building_eol": (
            devices.assign(has=devices["device_id"].map(lambda d: pd.notna(eol.get(d))))
            .groupby("building_id")["has"]
            .sum()
            .astype(int)
            .to_dict()
        ),
        "stride": stride,
    }


# ---------------------------------------------------------------------------
# deployment side: the incremental loop the forecaster actually runs
# ---------------------------------------------------------------------------

def build_deploy_side(dataset: Path, climatology: np.ndarray) -> dict:
    started = time.time()
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    locations, timeseries, eol_times, scenarios = load_dataset(dataset)

    cache = SmoothingCache()
    shape_cache = ShapeCache()
    raw_cache = RawDailyCache()
    context = FeatureContext(climatology=climatology)

    columns: dict[str, list] = {
        k: []
        for k in (
            "scenario_index",
            "device",
            "building",
            "remaining",
            "due",
            "margin",
            "staleness",
            "beta30",
        )
    }
    keys: list[bytes] = []
    windows: list[np.ndarray] = []

    for s_index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        start = pd.Timestamp(scenario["start_time"])
        origin_ordinal = _ordinal(start)
        horizon = int(scenario["settings"].planning_window_days)
        horizon_end = start + pd.Timedelta(days=horizon)

        cache.update(cut)
        shape_cache.update(cut)
        raw_cache.update(cut)

        end_time = pd.to_datetime(locs["end_time"])
        if getattr(end_time.dt, "tz", None) is not None:
            end_time = end_time.dt.tz_localize(None)
        remaining_days = (
            (end_time.dt.normalize() - start.normalize()) / pd.Timedelta(days=1)
        ).to_numpy(dtype=float)
        battery_ids = locs["battery"].astype(str).to_numpy()

        kept = 0
        for position, device_id in enumerate(battery_ids):
            series = cache.devices.get(device_id)
            if series is None:
                continue
            index = series.index_of(origin_ordinal)
            if index < 0:
                continue
            index = min(index, len(series) - 1)  # the forecaster's clamp
            view = DeviceView(series.smooth_voltage, series.smooth_temperature)
            shape_view = align_to(
                shape_cache.devices.get(device_id), series.origin, len(series)
            )
            row = feature_row(view, index, series.origin + index, context, shape_view)
            if row is None:
                continue
            row32 = np.asarray(row, dtype=np.float32)
            remaining = float(remaining_days[position])
            key = (
                np.ascontiguousarray(row32[list(KEY_COLUMNS)]).tobytes()
                + np.int32(round(remaining)).tobytes()
            )
            channels = device_channels(cache, shape_cache, raw_cache, device_id)
            window = window_from_channels(channels, index)

            moment = not_dead.get(device_id)
            due = int((not pd.isna(moment)) and moment <= horizon_end)
            value, stale = view.value_at_or_before(index)

            columns["scenario_index"].append(s_index)
            columns["device"].append(device_id)
            columns["building"].append(building_of.get(device_id, ""))
            columns["remaining"].append(remaining)
            columns["due"].append(due)
            columns["margin"].append(float(value) - EOL_THRESHOLD)
            columns["staleness"].append(float(stale))
            columns["beta30"].append(float(row[BETA30_COL]))
            keys.append(key)
            windows.append(window.astype(np.float16))
            kept += 1
        print(
            f"  {scenario['name']:>5}  rows={kept:3d}  "
            f"due={int(np.sum([d for i, d in zip(columns['scenario_index'], columns['due']) if i == s_index])):3d}  "
            f"{time.time()-started:5.0f}s",
            flush=True,
        )

    out = {
        "scenario_index": np.asarray(columns["scenario_index"], dtype=np.int32),
        "device": np.asarray(columns["device"]),
        "building": np.asarray(columns["building"]),
        "remaining": np.asarray(columns["remaining"], dtype=np.float64),
        "due": np.asarray(columns["due"], dtype=np.int8),
        "margin": np.asarray(columns["margin"], dtype=np.float64),
        "staleness": np.asarray(columns["staleness"], dtype=np.float64),
        "beta30": np.asarray(columns["beta30"], dtype=np.float64),
        "keys": keys,
        "windows": np.stack(windows).astype(np.float16),
        "n_scenarios": len(scenarios),
    }

    # The wrapper's whole correctness rests on fingerprint uniqueness. A
    # collision is tolerable only when the colliding rows carry identical
    # windows and margins (byte-identical device histories): the wrapper then
    # returns the right prediction regardless of which row it resolves to.
    seen: dict[bytes, int] = {}
    benign, fatal = 0, 0
    for i, key in enumerate(keys):
        j = seen.get(key)
        if j is None:
            seen[key] = i
            continue
        same = (
            np.array_equal(out["windows"][i], out["windows"][j])
            and abs(out["margin"][i] - out["margin"][j]) < 1e-9
        )
        pair = (
            f"s{out['scenario_index'][j]}/{out['device'][j]} vs "
            f"s{out['scenario_index'][i]}/{out['device'][i]}"
        )
        if same:
            benign += 1
            print(f"  benign fingerprint collision (identical windows): {pair}")
        else:
            fatal += 1
            print(f"  FATAL fingerprint collision (different windows): {pair}")
    if fatal:
        raise SystemExit(f"{fatal} unresolvable fingerprint collisions")
    print(
        f"deploy snapshot: {len(keys)} rows, {int(out['due'].sum())} due, "
        f"windows {out['windows'].nbytes/1e6:.0f} MB f16, "
        f"keys unique ({benign} benign twin collisions)",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset/train")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "outputs/seq_pack_summary.json")
    args = parser.parse_args()

    started = time.time()
    args.work.mkdir(parents=True, exist_ok=True)

    train_side = build_train_side(args.dataset, args.stride)
    print("deployment snapshot (incremental caches, 48 scenarios)...", flush=True)
    deploy = build_deploy_side(args.dataset, train_side["climatology"])

    pack = {"train": train_side, "deploy": deploy}
    out = args.work / "seq_pack.joblib"
    joblib.dump(pack, out, compress=0)

    summary = {
        "stride": args.stride,
        "n_cutoffs": int(len(train_side["frame"])),
        "n_windows": int(train_side["window_row"].size),
        "n_due_stride": int(train_side["truth"].sum()),
        "base_rate_stride": round(float(train_side["truth"].mean()), 5),
        "deploy_rows": int(len(deploy["keys"])),
        "deploy_due": int(deploy["due"].sum()),
        "deploy_due_per_scenario": round(float(deploy["due"].sum() / deploy["n_scenarios"]), 3),
        "bank_mb": round(train_side["bank_data"].nbytes / 1e6, 1),
        "windows_mb": round(deploy["windows"].nbytes / 1e6, 1),
        "seconds": round(time.time() - started, 1),
        "pack_path": str(out),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out} in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
