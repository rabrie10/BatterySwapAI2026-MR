"""ROADBLOCK V16 re-audit: quantify the NEW self-imposed constraints.

(a) X-gate 50: which scenarios the exchange skips, and what the paired arm
    (A+B refilled) measured there — the forfeited/protected points for gate
    variants {none, s40-47, X<100, X<50}.
(b) Refill p>0.05 floor: per-scenario refill supply beyond the slot limit.
(c) Production dark-gate staleness clamp: reproduce the runtime (clamped)
    staleness against the frame (true) staleness; count the dark rows and
    dues the production rule can/cannot see, and price the difference with
    the paired per-battery earnings.
(d) pi-hybrid worth: mid-block top-12 0.214 -> 0.292 converted to realized
    points via catch-conversion arithmetic + the L1 analytic->realized scale.

Inputs: outputs/research_rowfeat.parquet, outputs/paired_selection.json,
dataset/train. Output: outputs/roadblock_v16.json. No planner runs here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsai.smoothing import SmoothingCache

_EPOCH = pd.Timestamp("1970-01-01")
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def main() -> None:
    out: dict = {}
    frame = pd.read_parquet(ROOT / "outputs" / "research_rowfeat.parquet")
    paired = json.load(open(ROOT / "outputs" / "paired_selection.json"))
    scen_rows = paired["scenarios"]

    # ---------------------------------------------------------- (a) X-gate 50
    med_x = frame.groupby("scenario")["remaining"].median() - 12.0
    deltas = np.array(
        [
            (s.get("arm_ab_refill") or {}).get("total", 0.0) or 0.0
            for s in scen_rows
        ],
        dtype=float,
    )
    gates = {
        "no_gate": np.zeros(48, dtype=bool),
        "s40_47(measured)": np.arange(48) >= 40,
        "X<100": (med_x < 100.0).to_numpy(),
        "X<50(shipped)": (med_x < 50.0).to_numpy(),
    }
    table = {}
    for name, skip in gates.items():
        applied = np.where(skip, 0.0, deltas)
        table[name] = {
            "skipped_scenarios": int(skip.sum()),
            "skipped_set": [int(i) for i in np.flatnonzero(skip)],
            "mean_delta_48": round(float(applied.mean()), 1),
            "value_forfeited(negative deltas skipped)": round(
                float(deltas[skip & (deltas < 0)].sum() / 48.0), 1
            ),
            "harm_avoided(positive deltas skipped)": round(
                float(deltas[skip & (deltas > 0)].sum() / 48.0), 1
            ),
        }
    out["a_x_gate"] = {
        "median_X_per_scenario": [round(float(v), 0) for v in med_x],
        "arm_ab_refill_deltas": [round(float(d), 0) for d in deltas],
        "gate_variants": table,
    }
    log(f"X<50 skips {int(gates['X<50(shipped)'].sum())} scenarios")

    # ------------------------------------------------- (b) refill p>0.05 floor
    supply = []
    for scenario, sub in frame.groupby("scenario"):
        p = sub["p_cal"].to_numpy(dtype=float)
        limit = int(min(15, np.ceil(1.6 * p.sum() + 1)))
        above = int((np.sort(p)[::-1][limit:] > 0.05).sum())
        supply.append(
            {"scenario": int(scenario), "limit": limit, "refill_supply": above}
        )
    supply_df = pd.DataFrame(supply)
    out["b_refill_floor"] = {
        "floor": 0.05,
        "mean_refill_supply_beyond_limit": round(
            float(supply_df["refill_supply"].mean()), 2
        ),
        "scenarios_with_supply_0": int((supply_df["refill_supply"] == 0).sum()),
        "scenarios_with_supply_le2": int((supply_df["refill_supply"] <= 2).sum()),
        "refill_increment_measured(paired, ab_refill - ab)": round(
            float(
                np.mean(
                    [
                        (s.get("arm_ab_refill") or {}).get("total", 0.0) or 0.0
                        for s in scen_rows
                    ]
                )
                - np.mean(
                    [
                        (s.get("arm_ab") or {}).get("total", 0.0) or 0.0
                        for s in scen_rows
                    ]
                )
            ),
            1,
        ),
        "note": "supply = batteries beyond the slot limit with p_cal>0.05 "
        "(zombie/gate exclusions not netted; upper bound)",
    }

    # --------------------------- (c) production staleness clamp vs frame truth
    log("building SmoothingCache for clamped staleness ...")
    raw = pd.read_parquet(
        ROOT / "dataset" / "train" / "battery_metrics.parquet", engine="fastparquet"
    )
    cache = SmoothingCache()
    cache.update(raw)
    del raw

    batteries = frame["battery"].to_numpy()
    cutoffs = frame["cutoff_ord"].to_numpy()
    stale_prod = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        series = cache.devices.get(batteries[i])
        if series is None:
            continue
        local = int(cutoffs[i]) - series.origin
        if local < 0:
            continue
        # runtime grid ends at the last STABLE day <= cutoff (causal cache):
        volt = series.voltage[: min(local, len(series) - 1) + 1]
        stable = np.flatnonzero(np.isfinite(volt))
        if stable.size == 0:
            continue
        clamp_index = int(stable[-1])  # index = min(index, len-1) at runtime
        smooth = series.smooth_voltage[: clamp_index + 1]
        valid = np.flatnonzero(np.isfinite(smooth))
        if valid.size == 0:
            continue
        stale_prod[i] = clamp_index - int(valid[-1])

    m = frame["margin"].to_numpy(dtype=float)
    st_true = frame["staleness"].to_numpy(dtype=float)
    remaining = frame["remaining"].to_numpy(dtype=float)
    due = frame["due"].to_numpy(dtype=bool)
    dark_frame = (st_true > 30.0) & ((m - 0.001 * st_true) < 0.02)
    dark_prod = (
        (stale_prod > 30.0)
        & ((m - 0.001 * stale_prod) < 0.02)
        & (remaining >= 30.0)
    )
    dark_frame_r30 = dark_frame & (remaining >= 30.0)
    # price the missing rows with the paired per-battery earnings
    earnings = {
        r["battery"]: (r["mean_delta"], r["n"])
        for r in paired["composition"]["arm_a_by_battery"]
        if r["gate"] == "dark"
    }
    missing = dark_frame_r30 & ~dark_prod
    missing_rows = frame.loc[missing, ["scenario", "battery", "due", "p_cal"]]
    priced = 0.0
    for battery, n in missing_rows["battery"].value_counts().items():
        if battery in earnings:
            priced += earnings[battery][0] * min(n, earnings[battery][1])
    out["c_staleness_clamp"] = {
        "dark_rows_frame(true staleness)": int(dark_frame.sum()),
        "dark_rows_frame_remaining_ge30": int(dark_frame_r30.sum()),
        "dark_rows_production(clamped)": int(dark_prod.sum()),
        "overlap": int((dark_frame_r30 & dark_prod).sum()),
        "dues_frame_r30": int((dark_frame_r30 & due).sum()),
        "dues_production": int((dark_prod & due).sum()),
        "dues_forfeited_by_clamp": int((missing & due).sum()),
        "forfeited_rows": int(missing.sum()),
        "forfeited_value_per_scen(paired per-battery earnings)": round(
            priced / 48.0, 1
        ),
        "note": "production staleness is measured inside the causal grid "
        "(clamped at the last stable day), so grid-overhang devices read ~0; "
        "the frame's staleness is the true gap. remaining>=30 applied to both.",
    }
    log(
        f"dark frame(r30) {int(dark_frame_r30.sum())} vs production "
        f"{int(dark_prod.sum())}, dues {int((dark_frame_r30 & due).sum())} vs "
        f"{int((dark_prod & due).sum())}"
    )

    # ------------------------------------------------- (d) pi-hybrid conversion
    mid = frame[(frame["scenario"] >= 16) & (frame["scenario"] < 32)]
    d2e = mid["days_to_eol"].to_numpy(dtype=float)
    dues_per_mid_scen = float(mid.groupby("scenario")["due"].sum().mean())
    # value of one converted mid-block catch: avoided emergency (late+op) minus
    # catch cost, from the same analytic constants as L1, with the realized
    # per-catch band measured by the paired harness / lm A-B as cross-checks.
    analytic_per_catch = 10.0 * 27.3 + 6.0 - (0.5 * 5.0 + 5.6)  # ~271
    realized_band = (245.0, 300.0)
    delta_rate = 0.292 - 0.214
    extra_catches_per_mid_scen = delta_rate * 12
    victim_due_share = 0.354  # displaced lowest-p planned realizes due
    net_new = extra_catches_per_mid_scen * (1.0 - victim_due_share)
    gross_48 = extra_catches_per_mid_scen * 16 / 48
    net_48 = net_new * 16 / 48
    out["d_pi_hybrid_worth"] = {
        "mid_top12_rate": [0.214, 0.292],
        "extra_catches_per_mid_scenario_gross": round(extra_catches_per_mid_scen, 3),
        "per_catch_value_analytic": round(analytic_per_catch, 0),
        "per_catch_value_realized_band": realized_band,
        "worth_48mean_if_pure_reorder(no victim cost)": [
            round(gross_48 * realized_band[0], 1),
            round(gross_48 * realized_band[1], 1),
        ],
        "worth_48mean_if_conversions_displace_weakest(0.354 due)": [
            round(net_48 * realized_band[0], 1),
            round(net_48 * realized_band[1], 1),
        ],
        "l1_crosscheck_god_gap_fraction": {
            "mid_blocks_ordering_gap_analytic": [2038.1, 1672.7],
            "captured_fraction": round(delta_rate / (1.0 - 0.214), 3),
            "implied_48mean_analytic": round(
                (2038.1 + 1672.7) / 6.0 * (delta_rate / (1.0 - 0.214)), 1
            ),
            "realized_scale(x1.13-1.25 from L1 conversion)": [1.13, 1.25],
        },
        "dues_per_mid_scenario": round(dues_per_mid_scen, 2),
    }

    out_path = ROOT / "outputs" / "roadblock_v16.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "a_x_gate"}, indent=2))
    print(json.dumps(out["a_x_gate"]["gate_variants"], indent=2))


if __name__ == "__main__":
    main()
