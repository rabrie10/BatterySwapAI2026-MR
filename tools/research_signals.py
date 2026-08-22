"""RESEARCH probe 2+3: raw-daily channel value + within-day shape beyond beta.

Consumes outputs/research_rowfeat.parquet (from tools/research_extract.py).

A. Raw-daily marginal value (frontier B): among dues failing in window days
   1-14, how many are raw-visible (raw_min3 < threshold) while the smoothed
   margin still reads healthy and the model p is below the planner's pick
   range? Gate precision over all rows, per block, with the swap-EV value
   arithmetic (catch ~ +270, waste ~ -133).

B. Within-day hourly-shape statistics beyond beta30: univariate AUC of pulse
   depth, upper room, night-day gap, deep fraction, skew (trailing-14d and
   rise-ratio forms) on the mid-block knee band (margin 0.05-0.20,
   remaining >= 30), against the mined axes (beta30, margin, p).

Output: outputs/research_signals.json. Runtime: seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    m = np.isfinite(score)
    score, label = score[m], label[m]
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    r_pos = ranks[label]
    u = r_pos.sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def main() -> None:
    f = pd.read_parquet(ROOT / "outputs" / "research_rowfeat.parquet")
    report: dict = {}

    # ------------------------------------------------ A. raw-daily channel
    fail14 = f[f["due"] & (f["days_to_eol"] > 0) & (f["days_to_eol"] <= 14)]
    report["fail14_per_scenario"] = round(len(fail14) / 48, 2)

    def gate_stats(sub: pd.DataFrame, col: str, thr: float, margin_floor: float) -> dict:
        flag = (sub[col] < thr) & (sub["margin"] > margin_floor)
        return {"n": int(flag.sum()), "frac": round(float(flag.mean()), 3)}

    grids = {}
    for col in ("raw_min3", "raw_min7"):
        for thr in (2.40, 2.42, 2.45):
            key = f"{col}<{thr}"
            grids[key] = {
                "dues_fail14_flagged_marginGT0.03": gate_stats(fail14, col, thr, 0.03),
                "dues_fail14_flagged_marginGT0.03_pcalLT0.1": gate_stats(
                    fail14[fail14["p_cal"] < 0.10], col, thr, 0.03
                ),
                "dues_fail14_flagged_marginGT0.03_pcalLT0.02": gate_stats(
                    fail14[fail14["p_cal"] < 0.02], col, thr, 0.03
                ),
            }
    report["fail14_raw_visibility"] = grids

    # where do the p-invisible dues sit? (frontier B population)
    invis = f[f["due"] & (f["p_cal"] < 0.02)]
    report["invisible_dues"] = {
        "n": int(len(invis)),
        "per_scenario": round(len(invis) / 48, 2),
        "days_to_eol_quartiles": [
            round(float(x), 1) for x in np.nanpercentile(invis["days_to_eol"], [25, 50, 75])
        ],
        "margin_quartiles": [
            round(float(x), 3) for x in np.nanpercentile(invis["margin"], [25, 50, 75])
        ],
        "staleness_quartiles": [
            round(float(x), 1) for x in np.nanpercentile(invis["staleness"], [25, 50, 75])
        ],
        "frac_raw_min3_lt_2.42": round(float((invis["raw_min3"] < 2.42).mean()), 3),
        "frac_raw_min7_lt_2.45": round(float((invis["raw_min7"] < 2.45).mean()), 3),
        "frac_beta30_ge_0.008": round(float((invis["beta30"] >= 0.008).mean()), 3),
    }

    # gate precision on ALL rows (candidate-add mechanism: order change)
    prec = {}
    for col, thr, mfloor, pcap in (
        ("raw_min3", 2.42, 0.03, 0.10),
        ("raw_min3", 2.42, 0.03, 1.01),
        ("raw_min3", 2.40, 0.03, 0.10),
        ("raw_min7", 2.45, 0.03, 0.10),
        ("raw_min3", 2.42, 0.05, 0.10),
    ):
        flag = (f[col] < thr) & (f["margin"] > mfloor) & (f["p_cal"] < pcap)
        sub = f[flag]
        n = len(sub)
        r42 = float(sub["due"].mean()) if n else float("nan")
        r14 = float(((sub["days_to_eol"] > 0) & (sub["days_to_eol"] <= 14)).mean()) if n else float("nan")
        by_block = {}
        for name, rng in (("open", range(0, 16)), ("mid", range(16, 32)), ("late", range(32, 48))):
            s2 = sub[sub["scenario"].isin(rng)]
            by_block[name] = {
                "n_per_scen": round(len(s2) / 16, 2),
                "due_rate": round(float(s2["due"].mean()), 3) if len(s2) else None,
            }
        ev = n / 48 * (r42 * 270 - (1 - r42) * 133) if n else 0.0
        prec[f"{col}<{thr} & margin>{mfloor} & p<{pcap}"] = {
            "n_total": n,
            "n_per_scen": round(n / 48, 2),
            "due_rate_42d": round(r42, 3),
            "fail14_rate": round(r14, 3),
            "by_block": by_block,
            "ev_points_per_scen(+270catch/-133waste)": round(ev, 1),
        }
    report["raw_gate_precision"] = prec

    # staleness interaction: is the raw gate mostly a staleness story?
    flag = (f["raw_min3"] < 2.42) & (f["margin"] > 0.03) & (f["p_cal"] < 0.10)
    report["raw_gate_staleness"] = {
        "flagged_staleness_quartiles": [
            round(float(x), 1)
            for x in np.nanpercentile(f.loc[flag, "staleness"], [25, 50, 75])
        ],
        "all_staleness_p90": round(float(np.nanpercentile(f["staleness"], 90)), 1),
    }

    # ------------------------------------------- B. within-day shape AUCs
    def auc_table(sub: pd.DataFrame) -> dict:
        label = sub["due"].to_numpy()
        out = {"n": int(len(sub)), "n_due": int(label.sum())}
        for col in (
            "p_cal",
            "margin",
            "beta30",
            "h14_depth",
            "h14_room",
            "h14_nightday",
            "h14_skew",
            "h14_deep_frac",
            "rise_depth",
            "rise_room",
            "rise_nightday",
            "rise_deep_frac",
            "raw_min3",
            "raw_slope7",
        ):
            if col not in sub.columns:
                continue
            a = auc(sub[col].to_numpy(dtype=float), label)
            # orient: report max(a, 1-a) with sign
            out[col] = {"auc": round(a, 3), "oriented": round(max(a, 1 - a), 3)}
        return out

    band_all = f[(f["margin"].between(0.05, 0.20)) & (f["remaining"] >= 30)]
    band_mid = band_all[band_all["scenario"].between(16, 31)]
    knee_pool = f[
        (f["margin"].between(0.05, 0.15)) & (f["beta30"] >= 0.008) & (f["remaining"] >= 30)
    ]
    report["auc_band_margin005_020_all"] = auc_table(band_all)
    report["auc_band_margin005_020_mid"] = auc_table(band_mid)
    report["auc_knee_pool(margin005_015_beta008)"] = auc_table(knee_pool)

    # do the new stats add WITHIN beta-elevated? split knee pool by beta median
    hi_beta = knee_pool[knee_pool["beta30"] >= knee_pool["beta30"].median()]
    report["auc_knee_pool_hibeta_half"] = auc_table(hi_beta)

    out = ROOT / "outputs" / "research_signals.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
