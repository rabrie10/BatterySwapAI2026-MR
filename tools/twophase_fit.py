"""Fit the two-phase Wiener changepoint model and run its control gates.

Everything before the planner: EM fits (5 folds grouped by building, exactly
the train_wiener discipline, plus a production fit), per-device causal onset
posteriors with each device's out-of-fold parameters, p42 on the scenario
frame, and the four ordered control gates from the P3-1 spec:

    g1  dwell table: among margin<0.02 rows, predicted p42 falls with dwell
        like the empirical 0.80 / 0.29 / 0.18
    g2  zombie five median p42 < 0.3 while open-block genuine top rows keep >0.5
    g3  frame PR-AUC >= 0.45 (incumbent ~0.43-0.47); mid-block top-12 rate
    g4  sum-p per block within 0.8-1.25 of realized (RemainingCalibration may
        be bolted on if the shape is right but the level drifts)

    python tools/twophase_fit.py --rebuild-series
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.utils import load_devices

from bsai.calibrate import RemainingCalibration
from bsai.features import fleet_climatology
from bsai.smoothing import SmoothingCache
from bsai.twophase import (
    DeviceTrack,
    TwoPhaseModel,
    TwoPhaseParams,
    causal_scales,
    fit_em,
    forward_pi,
    mixture_crossing,
    onset_hazard_bins,
    plateau_weights,
    single_phase_loglik,
    trailing_drift,
)
from bsai.wiener import first_passage_probability

_EPOCH = pd.Timestamp("1970-01-01")

# The five documented floor-zombies (docs/AUDIT_OPERATING_POINT.md 2b).
ZOMBIES = (
    "d_b5b678a3f79f",
    "d_3d26e12378f1",
    "d_c9a2ce794b68",
    "d_d9d695df1683",
    "d_d4b4272d5229",
)

DWELL_EDGES = (0.0, 14.0, 42.0, 90.0, 1e9)
DWELL_LABELS = ("0-14", "15-42", "43-90", "90+")
EMPIRICAL_DWELL = (0.80, 0.29, 0.18)


def _ordinal(value) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return int((stamp.normalize() - _EPOCH) / pd.Timedelta(days=1))


def build_series(dataset: Path, out: Path) -> dict:
    started = time.time()
    devices = load_devices(dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    end_ordinal = {
        row.device_id: _ordinal(row.end_time) for row in devices.itertuples()
    }
    eol = pd.read_csv(dataset / "eol_times.csv")
    eol["end_time"] = pd.to_datetime(eol["end_time"], format="ISO8601")
    eol = eol.dropna(subset=["end_time"])
    eol_ordinal = {
        row.device_id: _ordinal(row.end_time) for row in eol.itertuples()
    }

    print("smoothing...", flush=True)
    raw = pd.read_parquet(dataset / "battery_metrics.parquet", engine="fastparquet")
    cache = SmoothingCache()
    cache.update(raw)
    del raw
    print(f"  {len(cache.devices)} devices, {time.time() - started:.0f}s", flush=True)

    climatology = fleet_climatology(
        {d: (s.origin, s.smooth_temperature) for d, s in cache.devices.items()}
    )

    series: dict[str, dict] = {}
    for device_id, dev in cache.devices.items():
        margin = dev.smooth_voltage - 2.4
        valid = np.flatnonzero(np.isfinite(margin))
        if valid.size == 0:
            continue
        days = (dev.origin + valid).astype(np.int64)
        below = margin[valid] < 0.05  # smooth_v < 2.45
        first_below = int(days[np.argmax(below)]) if below.any() else -1
        series[device_id] = {
            "days": days,
            "margin": margin[valid].astype(np.float64),
            "building": str(building_of.get(device_id, "")),
            "crossing": int(eol_ordinal.get(device_id, -1)),
            "end": int(end_ordinal.get(device_id, int(days[-1]))),
            "first_below": first_below,
        }

    bundle = {
        "series": series,
        "climatology": climatology,
        "end_ordinal": end_ordinal,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"wrote {out} ({len(series)} devices, {time.time() - started:.0f}s)")
    return bundle


def make_tracks(bundle: dict) -> list[DeviceTrack]:
    """Pre-crossing tracks with median holds merged.

    The trajectory of the battery that actually died ends at its recorded
    crossing; afterwards the series belongs to the replacement and its
    recovery jump would corrupt the plunge estimate.

    Exact repeats of the seven-day rolling median (35% of raw daily
    increments) are a discrete atom at zero that any Gaussian mixture
    collapses onto (measured: sigma_core -> floor, lambda -> cap). A hold is
    "the level did not move", so runs are compressed to their first day and
    the eventual move becomes one increment (dm != 0 over dt days) -- proper
    Wiener increments, and a completed hold run still scores as strong
    plateau evidence: a tiny dm over a long dt is far likelier under the
    plateau law than under a plunge that should have fallen mu2*dt.
    """
    tracks = []
    for device_id, entry in bundle["series"].items():
        days, margin = entry["days"], entry["margin"]
        if entry["crossing"] >= 0:
            keep = days <= entry["crossing"]
            days, margin = days[keep], margin[keep]
        if days.size < 2:
            continue
        moved = np.concatenate([[True], np.diff(margin) != 0.0])
        days, margin = days[moved], margin[moved]
        if days.size < 2:
            continue
        track = DeviceTrack(
            device=device_id,
            building=entry["building"],
            days=days,
            margin=margin,
        )
        track.scale = causal_scales(track.dm, track.dt)
        tracks.append(track)
    # Early increments (before the causal window has enough history) take the
    # fleet median scale -- the CMVN fallback for a fresh device.
    pool = np.concatenate([t.scale[np.isfinite(t.scale)] for t in tracks])
    fleet_scale = float(np.median(pool)) if pool.size else 1.0
    for track in tracks:
        track.scale = np.where(np.isfinite(track.scale), track.scale, fleet_scale)
    return tracks


def degradation_check(params: TwoPhaseParams) -> float:
    """mu1=mu2, sigma1=sigma2 must telescope to the single-phase law."""
    collapsed = TwoPhaseParams(
        mu1=params.mu1, sigma1=params.sigma1, mu2=params.mu1, sigma2=params.sigma1,
        rho=params.rho, pi0=params.pi0, scale_ref=params.scale_ref,
    )
    # Margins above FLOOR_MARGIN: below it the deliberate floor taper of the
    # plateau drift breaks the exact correspondence by design.
    margins = np.array([0.05, 0.1, 0.2, 0.3])
    worst = 0.0
    for h in (7.0, 42.0, 126.0):
        mixed = mixture_crossing(margins, np.zeros_like(margins), h, collapsed)
        plain = first_passage_probability(
            margins, np.full_like(margins, -collapsed.mu1 * h),
            np.full_like(margins, collapsed.sigma1_passage * np.sqrt(h)),
        )
        worst = max(worst, float(np.max(np.abs(mixed - plain))))
    return worst


def block_ratios(frame: pd.DataFrame, column: str, blocks: list[tuple[int, int]]):
    out = []
    for low, high in blocks:
        rows = frame[(frame.scenario >= low) & (frame.scenario <= high)]
        predicted = float(rows[column].sum())
        actual = float(rows.due.sum())
        out.append(
            {
                "scenarios": f"{low}-{high}",
                "predicted_per_scen": round(predicted / rows.scenario.nunique(), 2),
                "actual_per_scen": round(actual / rows.scenario.nunique(), 2),
                "ratio": round(predicted / max(actual, 1e-9), 3),
            }
        )
    return out


def top_k_rate(frame: pd.DataFrame, column: str, low: int, high: int, k: int = 12) -> float:
    rates = []
    for _, rows in frame[(frame.scenario >= low) & (frame.scenario <= high)].groupby("scenario"):
        top = rows.nlargest(k, column)
        rates.append(float(top.due.mean()))
    return float(np.mean(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--series", type=Path, default=Path("outputs/twophase_series.joblib"))
    parser.add_argument("--rebuild-series", action="store_true")
    parser.add_argument("--frame", type=Path, default=Path("outputs/research_rowfeat.parquet"))
    parser.add_argument("--report", type=Path, default=Path("outputs/twophase_gates.json"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/twophase_model_oof.joblib"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--reflection-weight",
        type=float,
        default=-1.0,
        help="weight of the touch-and-recover term; negative fits it on the "
        "floor touch rows together with the floor sigma",
    )
    parser.add_argument(
        "--barrier-shift",
        type=float,
        default=0.5826,
        help="Broadie-Glasserman-Kou daily-monitoring barrier offset, in units "
        "of the segment sigma; 0 disables",
    )
    parser.add_argument(
        "--mu2-ceiling",
        type=float,
        default=-1.5e-3,
        help="upper bound on the plunge drift (V/day; physics prior from the "
        "crossed devices' final-42d drift q90 of -1.27 mV/d); 0 disables",
    )
    parser.add_argument(
        "--sigma2-cap",
        type=float,
        default=1.5,
        help="cap sigma2 at this multiple of sigma1; 0 disables",
    )
    parser.add_argument("--force", action="store_true", help="write the artifact even on gate failure")
    args = parser.parse_args()

    started = time.time()
    if args.rebuild_series or not args.series.exists():
        bundle = build_series(args.dataset, args.series)
    else:
        bundle = joblib.load(args.series)
        print(f"loaded {args.series} ({len(bundle['series'])} devices)")

    tracks = make_tracks(bundle)
    crossings = sum(1 for t in tracks if bundle["series"][t.device]["crossing"] >= 0)
    increments = np.concatenate([t.dm for t in tracks])
    gaps = np.concatenate([t.dt for t in tracks])
    end_margin = [
        bundle["series"][t.device]["margin"][
            bundle["series"][t.device]["days"] <= bundle["series"][t.device]["crossing"]
        ][-1]
        for t in tracks
        if bundle["series"][t.device]["crossing"] >= 0
        and (bundle["series"][t.device]["days"] <= bundle["series"][t.device]["crossing"]).any()
    ]
    print(
        f"tracks {len(tracks)} (crossed {crossings}), increments {increments.size}, "
        f"gap>1 share {float((gaps > 1).mean()):.3f}, zero-increment share "
        f"{float((increments == 0).mean()):.3f}, median margin at crossing "
        f"{float(np.median(end_margin)):.4f}"
    )

    # ---- grouped folds, train_wiener discipline -------------------------------
    groups = np.concatenate([[t.building] * t.dm.shape[0] for t in tracks])
    splitter = GroupKFold(n_splits=args.folds)
    all_buildings = set(np.unique(groups))
    fold_of_building: dict[str, int] = {}
    fold_params: dict[int, TwoPhaseParams] = {}
    placeholder = np.zeros((groups.shape[0], 1))
    for fold, (train_rows, _) in enumerate(splitter.split(placeholder, None, groups)):
        train_buildings = set(np.unique(groups[train_rows]))
        held_out = all_buildings - train_buildings
        train_tracks = [t for t in tracks if t.building in train_buildings]
        ceiling = args.mu2_ceiling if args.mu2_ceiling < 0 else None
        cap = args.sigma2_cap if args.sigma2_cap > 0 else None
        params = fit_em(
            train_tracks, max_iter=args.max_iter, mu2_ceiling=ceiling, sigma2_cap_ratio=cap
        )
        params.rho_values = onset_hazard_bins(train_tracks, params)
        fold_params[fold] = params
        for building in held_out:
            fold_of_building[str(building)] = fold
        print(
            f"fold {fold}: mu1 {params.mu1*1e3:+.3f} mV/d  sig1 {params.sigma1_passage*1e3:.3f}mV  "
            f"mu1nf {params.mu1_passage*1e3:+.3f}  mu2 {params.mu2*1e3:+.3f}  "
            f"sig2 {params.sigma2_passage*1e3:.3f}mV  "
            f"lam {params.lam:.3f}  sigJ(rel) {params.sigma_j:.2f}  sref {params.scale_ref*1e3:.2f}mV  "
            f"rho {params.rho:.2e}  pi0 {params.pi0:.4f}  nf {params.n_nf}  "
            f"LL {params.loglik:.0f} ({params.n_iter} it, {time.time()-started:.0f}s)",
            flush=True,
        )

    production = fit_em(
        tracks,
        max_iter=args.max_iter,
        mu2_ceiling=args.mu2_ceiling if args.mu2_ceiling < 0 else None,
        sigma2_cap_ratio=args.sigma2_cap if args.sigma2_cap > 0 else None,
    )
    production.rho_values = onset_hazard_bins(tracks, production)

    # No-onset plateau rebound at the floor: forward 42-day mean of stalled
    # floor rows whose window stayed above the barrier (the plateau branch is
    # the no-onset path, so the survivor-conditional mean is its drift).
    entries_by_building: dict[str, list[dict]] = {}
    for device_id, entry in bundle["series"].items():
        entries_by_building.setdefault(entry["building"], []).append(entry)

    def measure_rebound(fit_entries: list[dict]) -> float:
        deltas: list[float] = []
        for entry in fit_entries:
            days, m = entry["days"], entry["margin"]
            if entry["crossing"] >= 0:
                keep = days <= entry["crossing"]
                days, m = days[keep], m[keep]
            if days.size < 2 or entry["first_below"] < 0:
                continue
            index_of = {int(t): i for i, t in enumerate(days)}
            for i in range(days.size):
                if not (0.0 < m[i] < 0.05) or (days[i] - entry["first_below"]) < 42.0:
                    continue
                j = None
                for off in (42, 43, 41, 44, 40):
                    j = index_of.get(int(days[i]) + off)
                    if j is not None:
                        break
                if j is None:
                    continue
                window = (days > days[i]) & (days <= days[i] + 42)
                if window.any() and float(m[window].min()) <= 0.0:
                    continue  # onset path: priced by the hazard, not here
                deltas.append((m[j] - m[i]) / float(days[j] - days[i]))
        if len(deltas) < 40:
            return 0.0
        return max(float(np.mean(deltas)), 0.0)

    for fold, params in fold_params.items():
        held = {b for b, f in fold_of_building.items() if f == fold}
        params.rebound = measure_rebound(
            [e for b, es in entries_by_building.items() if b not in held for e in es]
        )
    production.rebound = measure_rebound(
        [e for es in entries_by_building.values() for e in es]
    )
    print(
        "floor rebound (no-onset, mV/d): "
        + "  ".join(f"f{f}:{p.rebound*1e3:+.3f}" for f, p in fold_params.items())
        + f"  prod:{production.rebound*1e3:+.3f}"
    )
    print(
        "onset hazard (floor-fresh / floor-chronic / 50-150mV / >150mV, per day):\n  "
        + "  ".join(
            f"f{fold}:" + "/".join(f"{r:.1e}" for r in p.rho_values)
            for fold, p in fold_params.items()
        )
        + "\n  prod:" + "/".join(f"{r:.1e}" for r in production.rho_values)
    )

    mu_sp, sigma_sp, ll_sp = single_phase_loglik(tracks)
    lr = 2.0 * (production.loglik - ll_sp)
    print(
        f"production: mu1 {production.mu1*1e3:+.3f}  sig1 {production.sigma1_passage*1e3:.3f}mV  "
        f"mu2 {production.mu2*1e3:+.3f}  sig2 {production.sigma2_passage*1e3:.3f}mV  "
        f"lam {production.lam:.3f}  sigJ(rel) {production.sigma_j:.2f}  "
        f"sref {production.scale_ref*1e3:.2f}mV  rho {production.rho:.2e}  pi0 {production.pi0:.4f}"
    )
    print(
        f"LR 2-phase vs 1-phase: {lr:.0f} (1-phase mu {mu_sp*1e3:+.3f} mV/d, "
        f"sigma {sigma_sp*1e3:.3f}); degradation check max|diff| "
        f"{degradation_check(production):.4f}"
    )

    # ---- per-device out-of-fold onset posteriors ------------------------------
    fold_of_device: dict[str, int] = {}
    pi_tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for device_id, entry in bundle["series"].items():
        fold = fold_of_building.get(entry["building"], -1)
        fold_of_device[device_id] = fold
    params_of = lambda fold: fold_params.get(fold, production)
    for track in tracks:
        params = params_of(fold_of_device[track.device])
        pi_tables[track.device] = (
            track.days,
            forward_pi(track, params),
            trailing_drift(track.days, track.margin, params.mu1),
        )
    for device_id, entry in bundle["series"].items():
        if device_id in pi_tables:
            continue
        days = entry["days"]
        if entry["crossing"] >= 0:
            days = days[days <= entry["crossing"]]
        if days.size:
            params = params_of(fold_of_device[device_id])
            pi_tables[device_id] = (
                days,
                np.full(days.size, params.pi0),
                np.full(days.size, params.mu1),
            )
    print(f"pi tables for {len(pi_tables)} devices, {time.time()-started:.0f}s")

    model = TwoPhaseModel(
        params_by_fold=fold_params,
        fold_of_device=fold_of_device,
        production_params=production,
        pi_tables=pi_tables,
        end_ordinal={d: e["end"] for d, e in bundle["series"].items()},
        climatology=bundle["climatology"],
        first_below={d: e["first_below"] for d, e in bundle["series"].items()},
        calibration=None,
        reflection_weight=1.0,
        barrier_shift=args.barrier_shift,
    )

    # ---- score the scenario frame ---------------------------------------------
    frame = pd.read_parquet(args.frame)
    devices_arr = frame.battery.to_numpy(dtype=object)
    margin = frame.margin.to_numpy(dtype=float)
    remaining = frame.remaining.to_numpy(dtype=float)
    cutoff = frame.cutoff_ord.to_numpy(dtype=float)

    recovered = np.array(
        [model.end_ordinal.get(str(d), np.nan) for d in devices_arr], dtype=float
    ) - np.round(remaining)
    mismatches = int(np.nansum(np.abs(recovered - cutoff) > 0.5))
    print(f"origin recovery mismatches: {mismatches} of {len(frame)}")

    # onset posterior, trailing drift and dwell per row
    pi_rows = np.empty(len(frame))
    drift_rows = np.empty(len(frame))
    dwell = np.full(len(frame), -1.0)
    for row, (dev, cut) in enumerate(zip(devices_arr, cutoff)):
        entry = bundle["series"].get(str(dev))
        params = params_of(fold_of_device.get(str(dev), -1))
        pi_rows[row], drift_rows[row] = model.state_at(str(dev), float(cut), params)
        if entry is not None and entry["first_below"] >= 0:
            clamped = min(float(cut), float(entry["days"][-1]))
            if entry["first_below"] <= clamped:
                dwell[row] = clamped - entry["first_below"]
    fold_rows = frame.battery.map(lambda d: fold_of_device.get(str(d), -1)).to_numpy()
    frame = frame.assign(pi=pi_rows, dwell=dwell, fold=fold_rows)

    # ---- floor law from the decision-scale outcomes ----------------------------
    # One dispersion scalar (and the reflection weight) closes the passage law.
    # Increment-scale fits cannot identify it: the daily-row populations mix
    # already-post-onset devices with floor-sitters at weights the planner
    # never sees (measured: every increment-side fit escaped to its grid
    # ceiling). The decision-relevant weighting is the scenario frame itself,
    # so the scalar is fitted there, per fold on the OTHER folds' rows -- the
    # fit_calibration discipline -- through the full model structure (pi +
    # per-device drift + margin-binned onset hazard).
    finite = np.isfinite(margin)
    near = finite & (margin < 0.05) & (remaining >= 42.0)
    due_arr = frame.due.to_numpy(dtype=float)
    sigma_grid = np.arange(0.4e-3, 2.501e-3, 0.1e-3)
    weight_grid = (
        (0.0, 0.25, 0.5, 0.75, 1.0)
        if args.reflection_weight < 0
        else (args.reflection_weight,)
    )
    candidate_p: dict[tuple[float, float], np.ndarray] = {}
    from dataclasses import replace as dc_replace

    near_rows = np.flatnonzero(near)
    for w in weight_grid:
        for sigma_day in sigma_grid:
            p_near = np.empty(near_rows.shape[0])
            for fold in np.unique(fold_rows[near_rows]):
                params = dc_replace(
                    params_of(int(fold)),
                    sigma1_nf=float(sigma_day) / params_of(int(fold)).scale_ref,
                )
                sel = fold_rows[near_rows] == fold
                rows = near_rows[sel]
                p_near[sel] = mixture_crossing(
                    margin[rows],
                    pi_rows[rows],
                    42.0,
                    params,
                    mu1_row=drift_rows[rows],
                    dwell_row=dwell[rows],
                    reflection_weight=float(w),
                    barrier_shift=args.barrier_shift,
                )
            candidate_p[(float(sigma_day), float(w))] = p_near

    def best_candidate(row_mask: np.ndarray) -> tuple[float, float]:
        best, best_ll = (1.0e-3, 1.0), -np.inf
        y = due_arr[near_rows][row_mask]
        for key, p_near in candidate_p.items():
            p = np.clip(p_near[row_mask], 1e-6, 1.0 - 1e-6)
            ll = float(y @ np.log(p) + (1.0 - y) @ np.log(1.0 - p))
            if ll > best_ll:
                best, best_ll = key, ll
        return best

    fold_of_near = fold_rows[near_rows]
    chosen = {}
    for fold in sorted(fold_params):
        sigma_day, w = best_candidate(fold_of_near != fold)
        fold_params[fold].sigma1_nf = sigma_day / fold_params[fold].scale_ref
        fold_params[fold].n_nf = int((fold_of_near != fold).sum())
        chosen[fold] = (sigma_day, w)
    sigma_prod, reflection = best_candidate(np.ones(near_rows.shape[0], dtype=bool))
    production.sigma1_nf = sigma_prod / production.scale_ref
    production.n_nf = near_rows.shape[0]
    model.reflection_weight = (
        reflection if args.reflection_weight < 0 else args.reflection_weight
    )
    print(
        "floor law (frame-fitted): "
        + "  ".join(
            f"f{fold}:sig{s*1e3:.2f}mV/w{w:.2f}" for fold, (s, w) in chosen.items()
        )
        + f"  prod:sig{sigma_prod*1e3:.2f}mV/w{reflection:.2f} (n={near_rows.shape[0]})"
    )

    safe_margin = np.where(finite, margin, 9.0)
    p42 = model.predict_rows(safe_margin, remaining, devices_arr, horizons=np.array([42.0]))[:, 0]
    p42 = np.where(finite, p42, 0.0)
    frame = frame.assign(p42_raw=p42)

    # ---- g4 level first (it decides whether calibration is bolted on) ---------
    macro = [(0, 15), (16, 31), (32, 47)]
    blocks6 = [(i * 8, i * 8 + 7) for i in range(6)]
    ratios_raw = block_ratios(frame, "p42_raw", macro)
    level_ok_raw = all(0.8 <= b["ratio"] <= 1.25 for b in ratios_raw)

    calibrations: dict[int, RemainingCalibration] = {}
    if not level_ok_raw:
        corrected = np.empty(len(frame))
        for fold in sorted(set(fold_rows)):
            others = frame[frame.fold != fold]
            calibration = RemainingCalibration.fit(
                others.remaining.to_numpy(),
                others.p42_raw.to_numpy(),
                others.due.to_numpy(dtype=float),
            )
            calibrations[int(fold)] = calibration
            mask = fold_rows == fold
            corrected[mask] = np.clip(
                frame.p42_raw.to_numpy()[mask]
                * calibration.factor_for(remaining[mask]),
                0.0,
                1.0,
            )
        calibrations[-1] = RemainingCalibration.fit(
            frame.remaining.to_numpy(),
            frame.p42_raw.to_numpy(),
            frame.due.to_numpy(dtype=float),
        )
        frame = frame.assign(p42=corrected)
    else:
        frame = frame.assign(p42=frame.p42_raw)

    ratios_cal = block_ratios(frame, "p42", macro)
    ratios6_cal = block_ratios(frame, "p42", blocks6)
    g4_pass = all(0.8 <= b["ratio"] <= 1.25 for b in ratios_cal)

    # ---- g1 dwell table --------------------------------------------------------
    near = frame[(frame.margin < 0.02) & (frame.dwell >= 0)]
    bands = np.clip(
        np.searchsorted(np.asarray(DWELL_EDGES), near.dwell.to_numpy(), side="right") - 1,
        0,
        len(DWELL_LABELS) - 1,
    )
    g1_table = []
    for index, label in enumerate(DWELL_LABELS):
        rows = near[bands == index]
        g1_table.append(
            {
                "dwell": label,
                "n": int(len(rows)),
                "mean_p42": round(float(rows.p42.mean()), 3) if len(rows) else None,
                "mean_p42_raw": round(float(rows.p42_raw.mean()), 3) if len(rows) else None,
                "mean_pi": round(float(rows.pi.mean()), 3) if len(rows) else None,
                "incumbent_p_cal": round(float(rows.p_cal.mean()), 3) if len(rows) else None,
                "realized": round(float(rows.due.mean()), 3) if len(rows) else None,
            }
        )
    means = [b["mean_p42"] for b in g1_table[:3] if b["mean_p42"] is not None]
    g1_pass = (
        len(means) == 3
        and means[0] > means[1] > means[2]
        and means[0] >= 0.5
        and means[2] <= 0.35
    )

    # ---- g2 zombies vs genuine -------------------------------------------------
    zombie_rows = frame[frame.battery.isin(ZOMBIES)]
    zombie_median = float(zombie_rows.p42.median())
    per_zombie = {
        battery: {
            "n": int(len(rows)),
            "median_p42": round(float(rows.p42.median()), 3),
            "median_pi": round(float(rows.pi.median()), 3),
            "median_margin": round(float(rows.margin.median()), 4),
            "median_incumbent_p_cal": round(float(rows.p_cal.median()), 3),
        }
        for battery, rows in zombie_rows.groupby("battery")
    }
    open_frame = frame[(frame.scenario <= 15)]
    ranks = open_frame.groupby("scenario").p42.rank(ascending=False, method="first")
    genuine_top = open_frame[(ranks <= 12) & open_frame.due]
    genuine_median = float(genuine_top.p42.median()) if len(genuine_top) else float("nan")
    g2_pass = zombie_median < 0.3 and genuine_median > 0.5
    # Slot-occupancy view -- the audit's actual currency: how many of the
    # fifteen slots do the zombie five hold under each ranking?
    occupancy = {}
    for column in ("p42", "p_cal"):
        rank_all = frame.groupby("scenario")[column].rank(ascending=False, method="first")
        held = frame[(rank_all <= 15) & frame.battery.isin(ZOMBIES)]
        occupancy[column] = round(float(len(held)) / frame.scenario.nunique(), 2)

    # ---- g3 ranking -------------------------------------------------------------
    due = frame.due.to_numpy(dtype=int)
    ap_mine = float(average_precision_score(due, frame.p42.to_numpy()))
    ap_mine_raw = float(average_precision_score(due, frame.p42_raw.to_numpy()))
    ap_incumbent_raw = float(average_precision_score(due, frame.p.fillna(0.0).to_numpy()))
    ap_incumbent_cal = float(average_precision_score(due, frame.p_cal.fillna(0.0).to_numpy()))
    mid_rate = top_k_rate(frame, "p42", 16, 31)
    open_rate = top_k_rate(frame, "p42", 0, 15)
    mid_rate_inc = top_k_rate(frame, "p_cal", 16, 31)
    open_rate_inc = top_k_rate(frame, "p_cal", 0, 15)
    g3_pass = max(ap_mine, ap_mine_raw) >= 0.45

    # ---- sum-p / budget guard ----------------------------------------------------
    per_scen = frame.groupby("scenario").agg(
        sum_mine=("p42", "sum"), sum_inc=("p_cal", "sum"), realized=("due", "sum")
    )
    budget_mine = np.minimum(15, np.ceil(1.6 * per_scen.sum_mine + 1.0))
    budget_inc = np.minimum(15, np.ceil(1.6 * per_scen.sum_inc + 1.0))
    budget_delta = float((budget_inc - budget_mine).mean())
    shrunk = int((budget_mine < budget_inc).sum())
    guard = {
        "mean_sum_p_mine": round(float(per_scen.sum_mine.mean()), 2),
        "mean_sum_p_incumbent": round(float(per_scen.sum_inc.mean()), 2),
        "mean_budget_mine": round(float(budget_mine.mean()), 2),
        "mean_budget_incumbent": round(float(budget_inc.mean()), 2),
        "mean_budget_shrink_slots": round(budget_delta, 2),
        "scenarios_with_smaller_budget": shrunk,
        "sum_p_by_block_mine": [
            round(float(per_scen.sum_mine.iloc[low : high + 1].mean()), 2)
            for low, high in macro
        ],
        "sum_p_by_block_incumbent": [
            round(float(per_scen.sum_inc.iloc[low : high + 1].mean()), 2)
            for low, high in macro
        ],
    }

    verdicts = {"g1": g1_pass, "g2": g2_pass, "g3": g3_pass, "g4": g4_pass}
    report = {
        "params_by_fold": {str(f): p.as_dict() for f, p in fold_params.items()},
        "production_params": production.as_dict(),
        "single_phase": {"mu": mu_sp, "sigma": sigma_sp, "loglik": ll_sp},
        "lr_statistic": lr,
        "degradation_check_max_abs_diff": degradation_check(production),
        "reflection_weight": args.reflection_weight,
        "barrier_shift": args.barrier_shift,
        "calibration_used": bool(calibrations),
        "calibration_factors": {
            str(f): list(c.factors) for f, c in calibrations.items()
        },
        "g1_dwell_table": g1_table,
        "g1_reference": EMPIRICAL_DWELL,
        "g1_pass": g1_pass,
        "g2_zombie_median_p42": round(zombie_median, 3),
        "g2_per_zombie": per_zombie,
        "g2_open_genuine_top12_median_p42": round(genuine_median, 3),
        "g2_zombie_top15_slots_per_scenario": occupancy,
        "g2_pass": g2_pass,
        "g3_pr_auc": round(ap_mine, 4),
        "g3_pr_auc_raw": round(ap_mine_raw, 4),
        "g3_pr_auc_incumbent_raw": round(ap_incumbent_raw, 4),
        "g3_pr_auc_incumbent_cal": round(ap_incumbent_cal, 4),
        "g3_mid_top12_rate": round(mid_rate, 3),
        "g3_mid_top12_rate_incumbent": round(mid_rate_inc, 3),
        "g3_open_top12_rate": round(open_rate, 3),
        "g3_open_top12_rate_incumbent": round(open_rate_inc, 3),
        "g3_pass": g3_pass,
        "g4_blocks_raw": ratios_raw,
        "g4_blocks": ratios_cal,
        "g4_blocks6": ratios6_cal,
        "g4_pass": g4_pass,
        "budget_guard": guard,
        "verdicts": verdicts,
        "all_pass": all(verdicts.values()),
        "origin_recovery_mismatches": mismatches,
        "seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print("\n=== gates ===")
    print("g1 dwell (margin<0.02):")
    for band in g1_table:
        print(f"  {band}")
    for gate in ("g1", "g2", "g3", "g4"):
        print(f"{gate}: {'PASS' if verdicts[gate] else 'FAIL'}")
    print(
        f"g2 zombies {zombie_median:.3f} | genuine {genuine_median:.3f} | "
        f"zombie top-15 slots/scen mine {occupancy['p42']} vs incumbent {occupancy['p_cal']} "
        f"(audit: 4.0 planned)"
    )
    print(
        f"g3 AP {ap_mine:.4f} (raw {ap_mine_raw:.4f}) vs incumbent raw {ap_incumbent_raw:.4f} "
        f"cal {ap_incumbent_cal:.4f} | mid12 {mid_rate:.3f} vs {mid_rate_inc:.3f} (ref 0.214) "
        f"| open12 {open_rate:.3f} vs {open_rate_inc:.3f} (ref 0.589)"
    )
    print(f"g4 blocks: {[b['ratio'] for b in ratios_cal]} (raw {[b['ratio'] for b in ratios_raw]})")
    print(f"budget guard: {guard}")
    print(f"ALL {'PASS' if report['all_pass'] else 'FAIL'}  ({time.time()-started:.0f}s)")

    if report["all_pass"] or args.force:
        model.calibration = calibrations if calibrations else None
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)
        print(f"wrote {args.model_out}")
    else:
        print("gates failed: artifact not written")


if __name__ == "__main__":
    main()
