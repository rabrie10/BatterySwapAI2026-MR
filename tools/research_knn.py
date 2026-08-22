"""RESEARCH probe 4: kNN hazard on the last-90-day margin TRAJECTORY shape.

Nonparametric second opinion, building-free by construction: each query row's
hazard is the due rate among its k nearest historical device-windows from
OTHER buildings (and other batteries), where distance is Euclidean on the
shape-normalized trajectory (15 blocks of 6-day means).

Normalizations:
  anchor  -- subtract the final block (shape relative to the current level)
  zscore  -- per-trajectory standardization (pure shape)
  level   -- no normalization (control: how much is level, which margin has)

Controls: leave-building-out is the deploy-honest condition (at deploy the
reference set is the fully-resolved train split; the query building is new).
A stricter time-resolved variant (ref cutoff + 42d <= query cutoff) is also
reported.

Metrics: AUC on the mid-block knee band; per-scenario top-12 realized rate on
scenarios 16-31 (the frontier-A currency, baseline 0.214); AUC among
p_cal < 0.02 rows (frontier B).

Output: outputs/research_knn.json. Runtime: ~1-2 minutes.
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
    u = ranks[label].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def block_means(traj: np.ndarray, blocks: int = 15) -> np.ndarray:
    n, width = traj.shape
    step = width // blocks
    out = np.full((n, blocks), np.nan, dtype=np.float32)
    for b in range(blocks):
        seg = traj[:, b * step : (b + 1) * step]
        with np.errstate(all="ignore"):
            out[:, b] = np.nanmean(seg, axis=1)
    return out


def knn_scores(
    queries: np.ndarray,
    q_meta: pd.DataFrame,
    refs: np.ndarray,
    r_meta: pd.DataFrame,
    k: int,
    time_resolved: bool,
) -> np.ndarray:
    """Mean due-rate of k nearest refs from other buildings/batteries."""
    r_due = r_meta["due"].to_numpy()
    r_bld = r_meta["building"].to_numpy()
    r_bat = r_meta["battery"].to_numpy()
    r_cut = r_meta["cutoff_ord"].to_numpy()
    scores = np.full(len(queries), np.nan)
    chunk = 512
    ref_sq = np.einsum("ij,ij->i", refs, refs)
    for lo in range(0, len(queries), chunk):
        hi = min(lo + chunk, len(queries))
        q = queries[lo:hi]
        d2 = ref_sq[None, :] - 2.0 * q @ refs.T + np.einsum("ij,ij->i", q, q)[:, None]
        for row in range(hi - lo):
            i = lo + row
            mask = (r_bld != q_meta["building"].iat[i]) & (r_bat != q_meta["battery"].iat[i])
            if time_resolved:
                mask &= r_cut + 42 <= q_meta["cutoff_ord"].iat[i]
            d = d2[row][mask]
            if d.size < k:
                continue
            idx = np.argpartition(d, k)[:k]
            scores[i] = float(r_due[mask][idx].mean())
    return scores


def main() -> None:
    f = pd.read_parquet(ROOT / "outputs" / "research_rowfeat.parquet")
    npz = np.load(ROOT / "outputs" / "research_traj.npz")
    traj, valid = npz["traj"], npz["valid"]

    # forward-fill interior gaps
    filled = pd.DataFrame(traj).ffill(axis=1).to_numpy(dtype=np.float32)
    blocks = block_means(filled)
    ok = valid >= 54
    ok &= np.isfinite(blocks).all(axis=1)
    f = f.reset_index(drop=True)
    print(f"usable trajectories: {ok.sum()} / {len(f)}")

    level = blocks.copy()
    anchor = blocks - blocks[:, -1:]
    mu = blocks.mean(axis=1, keepdims=True)
    sd = blocks.std(axis=1, keepdims=True)
    zscore = (blocks - mu) / np.maximum(sd, 1e-4)

    ref_mask = ok.copy()
    mid_mask = ok & f["scenario"].between(16, 31).to_numpy()
    r_meta = f.loc[ref_mask, ["due", "building", "battery", "cutoff_ord"]].reset_index(drop=True)
    q_meta = f.loc[mid_mask, ["due", "building", "battery", "cutoff_ord", "scenario",
                              "margin", "remaining", "p_cal", "beta30"]].reset_index(drop=True)

    report: dict = {"n_ref": int(ref_mask.sum()), "n_query_mid": int(mid_mask.sum())}
    scores_by_variant: dict[str, np.ndarray] = {}
    for name, mat in (("anchor", anchor), ("zscore", zscore), ("level", level)):
        for k in (25, 50):
            s = knn_scores(mat[mid_mask], q_meta, mat[ref_mask], r_meta, k, False)
            scores_by_variant[f"{name}_k{k}"] = s
            print(f"{name} k={k} done", flush=True)
    # time-resolved variant for the best-behaved shape norm
    s_tr = knn_scores(anchor[mid_mask], q_meta, anchor[ref_mask], r_meta, 50, True)
    scores_by_variant["anchor_k50_timeresolved"] = s_tr

    # ------------------------------------------------------------- metrics
    band = (
        q_meta["margin"].between(0.05, 0.20)
        & (q_meta["remaining"] >= 30)
    ).to_numpy()
    label_band = q_meta["due"].to_numpy()[band]
    aucs = {"band_n": int(band.sum()), "band_due": int(label_band.sum())}
    aucs["p_cal"] = round(auc(q_meta["p_cal"].to_numpy()[band], label_band), 3)
    aucs["margin(neg)"] = round(auc(-q_meta["margin"].to_numpy()[band], label_band), 3)
    aucs["beta30"] = round(auc(q_meta["beta30"].to_numpy()[band], label_band), 3)
    for name, s in scores_by_variant.items():
        aucs[f"knn_{name}"] = round(auc(s[band], label_band), 3)
    report["auc_mid_band"] = aucs

    # frontier B: invisible rows
    inv = (q_meta["p_cal"] < 0.02).to_numpy()
    label_inv = q_meta["due"].to_numpy()[inv]
    report["auc_mid_invisible_plt002"] = {
        "n": int(inv.sum()),
        "n_due": int(label_inv.sum()),
        "knn_anchor_k50": round(auc(scores_by_variant["anchor_k50"][inv], label_inv), 3),
        "knn_zscore_k50": round(auc(scores_by_variant["zscore_k50"][inv], label_inv), 3),
        "margin(neg)": round(auc(-q_meta["margin"].to_numpy()[inv], label_inv), 3),
        "beta30": round(auc(q_meta["beta30"].to_numpy()[inv], label_inv), 3),
    }

    # frontier A currency: per-scenario top-12 realized rate, s16-31
    q = q_meta.copy()
    for name in ("anchor_k50", "zscore_k50", "level_k50"):
        q[f"knn_{name}"] = scores_by_variant[name]
    top12 = {}
    full_mid = f[f["scenario"].between(16, 31)]  # includes rows without traj
    base_rates = [g.nlargest(12, "p_cal")["due"].mean() for _, g in full_mid.groupby("scenario")]
    top12["p_cal_baseline_all_rows"] = round(float(np.mean(base_rates)), 3)
    for name in ("knn_anchor_k50", "knn_zscore_k50", "knn_level_k50"):
        rates_alone, rates_blend = [], []
        for s, g in q.groupby("scenario"):
            g = g.copy()
            g["rank_p"] = g["p_cal"].rank(pct=True)
            g["rank_k"] = g[name].rank(pct=True)
            rates_alone.append(g.nlargest(12, name)["due"].mean())
            g["blend"] = 0.5 * g["rank_p"] + 0.5 * g["rank_k"]
            rates_blend.append(g.nlargest(12, "blend")["due"].mean())
        top12[f"{name}_alone"] = round(float(np.mean(rates_alone)), 3)
        top12[f"{name}_blend50"] = round(float(np.mean(rates_blend)), 3)
    report["top12_mid"] = top12

    out = ROOT / "outputs" / "research_knn.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
