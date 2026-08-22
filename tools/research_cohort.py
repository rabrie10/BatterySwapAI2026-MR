"""RESEARCH probe 1: peer/cohort dynamics (building-level recent crossings).

Question: do OBSERVABLE recent EOL crossings (visible in the data before the
cutoff -- smooth_v < 2.4 matches eol_times.csv exactly per PHYSICS_VERIFICATION)
carry hazard information about the building-mates still alive, and does the
fleet-level trailing crossing count predict this window's due COUNT (a volume
knob, the kind of lever that survives the planner)?

Outputs: outputs/research_cohort.json (tables printed too). Runtime: seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_EPOCH = pd.Timestamp("1970-01-01")


def rate(mask_due: np.ndarray) -> float:
    return float(np.mean(mask_due)) if mask_due.size else float("nan")


def main() -> None:
    frame = pd.read_parquet(ROOT / "outputs" / "frame_oof_raw_beta.parquet")
    cal = pd.read_parquet(ROOT / "outputs" / "frame_oof_cal.parquet")
    frame = frame.merge(
        cal[["scenario", "battery", "p"]].rename(columns={"p": "p_cal"}),
        on=["scenario", "battery"],
        how="left",
    )
    scenarios = json.load(open(ROOT / "dataset" / "train" / "scenarios.json"))
    starts = pd.to_datetime([s["start_time"] for s in scenarios]).normalize()
    eol = pd.read_csv(ROOT / "dataset" / "train" / "eol_times.csv")
    eol["end_time"] = pd.to_datetime(eol["end_time"])
    eol = eol.dropna(subset=["end_time"])
    devices = pd.read_csv(ROOT / "dataset" / "train" / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    eol["building"] = eol["device_id"].map(building_of)

    frame["start"] = frame["scenario"].map(dict(enumerate(starts)))
    report: dict = {}

    # ------------------------------------------------------------------ anchors
    blocks = {"open_0_15": range(0, 16), "mid_16_31": range(16, 32), "late_32_47": range(32, 48)}
    anchors = {}
    for name, rng in blocks.items():
        sub = frame[frame["scenario"].isin(rng)]
        rates = []
        for s, grp in sub.groupby("scenario"):
            top = grp.nlargest(12, "p_cal")
            rates.append(top["due"].mean())
        anchors[name] = {
            "top12_rate_pcal": round(float(np.mean(rates)), 3),
            "dues_per_scen": round(float(sub.groupby("scenario")["due"].sum().mean()), 2),
        }
    report["anchors"] = anchors

    # -------------------------------------------- building recent-crossing counts
    # crossings strictly BEFORE the cutoff are observable at plan time.
    for lag in (42, 60, 90):
        counts = np.zeros(len(frame), dtype=int)
        for i, start in enumerate(starts):
            m = frame["scenario"].to_numpy() == i
            window_eol = eol[(eol["end_time"] >= start - pd.Timedelta(days=lag)) & (eol["end_time"] < start)]
            per_bld = window_eol.groupby("building").size()
            counts[m] = frame.loc[m, "building"].map(per_bld).fillna(0).to_numpy(dtype=int)
        frame[f"bld_cross_{lag}"] = counts
    # fleet-level trailing crossings
    fleet42 = {}
    for i, start in enumerate(starts):
        fleet42[i] = int(((eol["end_time"] >= start - pd.Timedelta(days=42)) & (eol["end_time"] < start)).sum())
    frame["fleet_cross_42"] = frame["scenario"].map(fleet42)

    # ------------------------------------------------- P(due | k) tables
    def k_table(sub: pd.DataFrame, col: str) -> dict:
        out = {}
        k = sub[col].clip(upper=3)
        for kk in (0, 1, 2, 3):
            m = k == kk
            out[f"k={kk}{'+' if kk == 3 else ''}"] = {
                "n": int(m.sum()),
                "due_rate": round(rate(sub.loc[m, "due"].to_numpy()), 4),
            }
        out["base"] = {"n": int(len(sub)), "due_rate": round(rate(sub["due"].to_numpy()), 4)}
        return out

    report["p_due_given_bldcross60"] = {
        "all": k_table(frame, "bld_cross_60"),
        "mid_16_31": k_table(frame[frame["scenario"].between(16, 31)], "bld_cross_60"),
        "band_margin_005_020": k_table(frame[frame["margin"].between(0.05, 0.20)], "bld_cross_60"),
        "mid_band": k_table(
            frame[frame["scenario"].between(16, 31) & frame["margin"].between(0.05, 0.20)],
            "bld_cross_60",
        ),
        "invisible_p_lt_002": k_table(frame[frame["p_cal"] < 0.02], "bld_cross_60"),
    }

    # ---------------------------------------- within-building temporal contrast
    # Controls for building identity: same building, scenarios with vs without
    # recent crossings. Mantel-Haenszel style pooled rates over buildings that
    # have BOTH states.
    strata = []
    for bld, grp in frame.groupby("building"):
        has = grp[grp["bld_cross_60"] >= 1]
        not_has = grp[grp["bld_cross_60"] == 0]
        if len(has) >= 30 and len(not_has) >= 30:
            strata.append(
                {
                    "building": bld,
                    "n_k1": int(len(has)),
                    "rate_k1": rate(has["due"].to_numpy()),
                    "n_k0": int(len(not_has)),
                    "rate_k0": rate(not_has["due"].to_numpy()),
                }
            )
    if strata:
        w = np.array([min(s["n_k1"], s["n_k0"]) for s in strata], dtype=float)
        r1 = np.array([s["rate_k1"] for s in strata])
        r0 = np.array([s["rate_k0"] for s in strata])
        report["within_building_contrast"] = {
            "n_buildings_with_both_states": len(strata),
            "weighted_rate_k>=1": round(float(np.sum(w * r1) / np.sum(w)), 4),
            "weighted_rate_k=0": round(float(np.sum(w * r0) / np.sum(w)), 4),
            "buildings_where_k1_higher": int(np.sum(r1 > r0)),
            "per_building": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                for s in sorted(strata, key=lambda s: -(s["n_k1"] + s["n_k0"]))[:12]
            ],
        }

    # --------------------------------- scenario-level count autocorrelation
    per_scen = frame.groupby("scenario").agg(
        dues=("due", "sum"), fleet_cross_42=("fleet_cross_42", "first")
    )
    dues = per_scen["dues"].to_numpy(dtype=float)
    pred = per_scen["fleet_cross_42"].to_numpy(dtype=float)
    acf = {}
    for lag in (1, 2, 3, 4, 6, 8):
        a, b = dues[:-lag], dues[lag:]
        acf[f"lag{lag}"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    # honest one-parameter predictor: scale trailing crossings to match mean,
    # fit on first half, evaluate on second half (and vice versa)
    def half_mae(train_idx, test_idx):
        scale = dues[train_idx].mean() / max(pred[train_idx].mean(), 1e-9)
        const = dues[train_idx].mean()
        return (
            float(np.mean(np.abs(dues[test_idx] - scale * pred[test_idx]))),
            float(np.mean(np.abs(dues[test_idx] - const))),
        )

    h1 = np.arange(24)
    h2 = np.arange(24, 48)
    mae_a = half_mae(h1, h2)
    mae_b = half_mae(h2, h1)
    report["scenario_count_prediction"] = {
        "due_count_acf": acf,
        "corr_dues_vs_trailing42_crossings": round(float(np.corrcoef(dues, pred)[0, 1]), 3),
        "mae_split_test_second_half": {"trailing_pred": round(mae_a[0], 2), "constant": round(mae_a[1], 2)},
        "mae_split_test_first_half": {"trailing_pred": round(mae_b[0], 2), "constant": round(mae_b[1], 2)},
        "dues_by_scenario": [int(x) for x in dues],
        "trailing42_by_scenario": [int(x) for x in pred],
    }

    # ----------------------------- identity lift: are dues in recently-hit buildings?
    lifts = {}
    for name, rng in blocks.items():
        sub = frame[frame["scenario"].isin(rng)]
        due_in_hot = rate((sub.loc[sub["due"], "bld_cross_60"] >= 1).to_numpy())
        alive_in_hot = rate((sub["bld_cross_60"] >= 1).to_numpy())
        lifts[name] = {
            "frac_dues_in_hot_bld": round(due_in_hot, 3),
            "frac_alive_in_hot_bld": round(alive_in_hot, 3),
            "lift": round(due_in_hot / max(alive_in_hot, 1e-9), 2),
        }
    report["due_location_lift"] = lifts

    # --------------------------- does cohort add ON TOP of p ordering? (mid block)
    # rank-blend probe: order by p_cal, break p-ties / boost by bld_cross_60
    mid = frame[frame["scenario"].between(16, 31)].copy()
    base_rates, boosted_rates = [], []
    for s, grp in mid.groupby("scenario"):
        grp = grp.copy()
        grp["rank_p"] = grp["p_cal"].rank(pct=True)
        grp["rank_k"] = grp["bld_cross_60"].clip(upper=3).rank(pct=True)
        base = grp.nlargest(12, "rank_p")["due"].mean()
        for w in (0.25,):
            blended = grp.nlargest(12, "rank_p", keep="all").head(12)  # anchor
        grp["blend"] = 0.75 * grp["rank_p"] + 0.25 * grp["rank_k"]
        boost = grp.nlargest(12, "blend")["due"].mean()
        base_rates.append(base)
        boosted_rates.append(boost)
    report["mid_top12_rank_blend"] = {
        "p_only": round(float(np.mean(base_rates)), 3),
        "p75_k25_blend": round(float(np.mean(boosted_rates)), 3),
    }

    out = ROOT / "outputs" / "research_cohort.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
