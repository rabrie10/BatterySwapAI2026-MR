"""Ship the pi-hybrid: production artifact + incremental-filter equivalence.

Builds the all-buildings production GBDT (66 features: base + pi from the
production filter), fits the production RemainingCalibration on the 5-fold
OOF scenario predictions (the fit_calibration discipline), and wraps it with
a live ``PiFilterCache`` so the artifact loads and runs through ``script.py``
unchanged. Proves the incremental filter equals the batch pipeline:

  t1  single-pass cache vs batch (make_tracks/causal_scales/forward_pi)
  t2  chunked updates (three time slices) vs single-pass

    python tools/twophase_ship.py

Planner smoke and the pytest suite run separately (see pihybrid_ship.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bsai.calibrate import RemainingCalibration
from bsai.features import FEATURE_NAMES
from bsai.smoothing import SmoothingCache
from bsai.twophase import (
    PI_FEATURE_NAMES,
    DeviceTrack,
    PiFilterCache,
    ProductionPiHybrid,
    causal_scales,
    forward_pi,
    trailing_drift,
)
from bsai.wiener import WienerModel

from twophase_fit import make_tracks  # noqa: E402
from twophase_pihybrid import (  # noqa: E402
    DEFAULT_WORK,
    filter_tables_for,
    pi_columns,
)

DECISION = 42.0


def batch_reference_tables(bundle, params, fleet_scale):
    """Non-truncated batch tracks + filter, for the equivalence check."""
    tables = {}
    for device_id, entry in bundle["series"].items():
        days, margin = entry["days"], entry["margin"]
        if days.size < 2:
            continue
        moved = np.concatenate([[True], np.diff(margin) != 0.0])
        days, margin = days[moved], margin[moved]
        if days.size < 2:
            continue
        track = DeviceTrack(device=device_id, building=entry["building"], days=days, margin=margin)
        scales = causal_scales(track.dm, track.dt)
        track.scale = np.where(np.isfinite(scales), scales, fleet_scale)
        tables[device_id] = (
            days,
            forward_pi(track, params),
            trailing_drift(track.days, track.margin, params.mu1),
        )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--series", type=Path, default=Path("outputs/twophase_series.joblib"))
    parser.add_argument("--filter", type=Path, default=Path("outputs/twophase_model_oof.joblib"))
    parser.add_argument("--oof-model", type=Path, default=Path("outputs/twophase_pihybrid_model.joblib"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--gates", type=Path, default=Path("outputs/pihybrid_gates.json"))
    parser.add_argument("--out", type=Path, default=Path("models/pihybrid.joblib"))
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--skip-fit", action="store_true", help="tests only")
    args = parser.parse_args()
    started = time.time()

    def stamp(msg):
        print(f"[{time.time()-started:6.0f}s] {msg}", flush=True)

    bundle = joblib.load(args.series)
    filter_model = joblib.load(args.filter)
    production_params = filter_model.production_params
    oof = joblib.load(args.oof_model)

    # Fleet observation-scale constant: median of the finite causal scales over
    # the training (pre-crossing) tracks -- the exact constant the batch
    # pipeline used, frozen into the deployable cache like the climatology.
    tracks = make_tracks(bundle)
    pool = np.concatenate(
        [causal_scales(t.dm, t.dt) for t in tracks if t.dm.shape[0]]
    )
    fleet_scale = float(np.median(pool[np.isfinite(pool)]))
    stamp(f"fleet observation scale {fleet_scale*1e3:.3f} mV/sqrt-day")

    # ---- equivalence: batch vs incremental --------------------------------------
    stamp("building smoothing cache for equivalence tests...")
    raw = pd.read_parquet(args.dataset / "battery_metrics.parquet", engine="fastparquet")
    full_cache = SmoothingCache()
    full_cache.update(raw)

    reference = batch_reference_tables(bundle, production_params, fleet_scale)

    cache_t1 = PiFilterCache(production_params, fleet_scale)
    cache_t1.update_from(full_cache)
    cache_t1.update_from(full_cache)  # idempotency under repeated update

    worst_pi, worst_len, checked = 0.0, 0, 0
    for device_id, (days_ref, pi_ref, _) in reference.items():
        state = cache_t1.devices.get(device_id)
        if state is None or not state["days"]:
            continue
        days_inc = np.asarray(state["days"])
        n = days_inc.size
        # the cache defers the final grid day; its kept days must be a prefix
        if not np.array_equal(days_inc, days_ref[:n]):
            worst_len += 1
            continue
        # compare the causal pi at the last common boundary via a batch prefix
        checked += 1
        worst_pi = max(worst_pi, abs(float(state["pi"]) - float(pi_ref[n - 1])))
    t1_pass = worst_pi < 1e-6 and worst_len == 0 and checked >= 450
    stamp(
        f"t1 single-pass equivalence: devices checked {checked}, day-sequence "
        f"mismatches {worst_len}, max |pi diff| {worst_pi:.2e} -> "
        f"{'PASS' if t1_pass else 'FAIL'}"
    )

    # t2: chunked updates through a growing smoothing cache. Deployment feeds
    # CUMULATIVE battery_data every scenario (make_submissions hands all data
    # up to the scenario start; the smoothing cache's watermark re-reads the
    # provisional boundary day from that full frame), so the test must too --
    # disjoint slices would starve the boundary day's earlier measurements,
    # which no deployment path ever does.
    end_time = pd.to_datetime(raw["end_time"])
    if getattr(end_time.dt, "tz", None) is not None:
        end_time = end_time.dt.tz_localize(None)
    cuts = end_time.quantile([0.4, 0.75]).tolist()
    order = [raw[end_time <= cuts[0]], raw[end_time <= cuts[1]], raw]
    grow_cache = SmoothingCache()
    cache_t2 = PiFilterCache(production_params, fleet_scale)
    for chunk in order:
        grow_cache.update(chunk)
        cache_t2.update_from(grow_cache)
    del raw

    worst2, mismatch2, compared = 0.0, 0, 0
    for device_id, s1 in cache_t1.devices.items():
        s2 = cache_t2.devices.get(device_id)
        if s2 is None:
            mismatch2 += 1
            continue
        if s1["days"] != s2["days"]:
            mismatch2 += 1
            continue
        compared += 1
        worst2 = max(worst2, abs(s1["pi"] - s2["pi"]))
    t2_pass = worst2 < 1e-9 and mismatch2 == 0 and compared >= 450
    stamp(
        f"t2 chunked equivalence: devices compared {compared}, mismatches "
        f"{mismatch2}, max |pi diff| {worst2:.2e} -> {'PASS' if t2_pass else 'FAIL'}"
    )

    if args.skip_fit:
        return

    # ---- production fit -----------------------------------------------------------
    bank = joblib.load(args.work / "bank_stride4.joblib")
    frame, rows, horizon, drop = bank["frame"], bank["rows"], bank["horizon"], bank["drop"]
    scen = joblib.load(args.work / "scen_frame.joblib")
    sframe, s_day = scen["frame"], scen["day_ord"]
    gates = json.load(open(args.gates))
    volatility_scale = float(gates["volatility_scale"])

    prod_tables = filter_tables_for(production_params, tracks)
    cols = pi_columns(frame.device, bank["day_ord"], prod_tables)
    features_prod = np.hstack([frame.features, cols]).astype(np.float32)
    climatology = oof.climatology

    stamp("fitting production 66-feature GBDT on all buildings...")
    design = np.hstack([features_prod[rows], horizon[:, None]]).astype(np.float32)
    production = WienerModel.fit(design, drop, climatology, params={"max_iter": args.max_iter})
    del design
    production.feature_names = tuple(FEATURE_NAMES) + PI_FEATURE_NAMES
    production.model_version = "bsai-wiener/v1+pi"
    production.volatility_scale = volatility_scale
    stamp("production GBDT fitted")

    # Production calibration on the 5-fold OOF scenario predictions (raw).
    rowfeat = pd.read_parquet("outputs/research_rowfeat.parquet")
    key = pd.DataFrame(
        {"battery": sframe.device, "cutoff_ord": s_day, "row": np.arange(len(sframe))}
    )
    joined = key.merge(rowfeat, on=["battery", "cutoff_ord"], how="inner")
    s_rows = joined.row.to_numpy()
    s_remaining = joined.remaining.to_numpy(dtype=float)
    s_due = joined.due.to_numpy(dtype=float)
    s_heff = np.clip(np.minimum(DECISION, s_remaining), 0.0, None).astype(np.float32)
    oof_cols = pi_columns(sframe.device[s_rows], s_day[s_rows], oof.pi_tables)
    s_feats = np.hstack([sframe.features[s_rows], oof_cols]).astype(np.float32)
    buildings = sframe.building[s_rows]
    p_raw = np.zeros(len(joined))
    for building in np.unique(buildings):
        model = oof.by_building[str(building)]
        mask = buildings == building
        p_raw[mask] = model.probabilities(s_feats[mask], s_heff[mask])
    p_raw = np.where(s_heff <= 0.0, 0.0, p_raw)
    production.calibration = RemainingCalibration.fit(s_remaining, p_raw, s_due)
    stamp(f"production calibration: {production.calibration.describe()}")

    artifact = ProductionPiHybrid(
        inner=production,
        pi_cache=PiFilterCache(production_params, fleet_scale),
        climatology=climatology,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out)
    stamp(f"wrote {args.out}")
    print(
        json.dumps(
            {
                "t1_pass": t1_pass,
                "t2_pass": t2_pass,
                "fleet_scale": fleet_scale,
                "volatility_scale": volatility_scale,
                "calibration_factors": list(production.calibration.factors),
                "artifact": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
