"""Ensemble combination x gate matrix on the OOF prediction matrix.

Reads outputs/ensemble_matrix.parquet (tools/ensemble_matrix.py) and scores
every combination on the gate ladder:

  gate (a) pooled frame PR-AUC + per-block top-12 realized rate
           (mid block s16-31 is the frontier; best single = twophase 0.292)
  gate (b) hard-holdout PR-AUC on the five transfer-stress groups, mean vs
           the 0.428 bar. Protocol: the cens component is replaced by the
           harness's stride-8/it-150 refit that never saw the held buildings
           (p_censhard_<fold>); twophase/qhead stay their own building-OOF
           predictions; scores are built over the WHOLE fleet (deployment
           view) and PR-AUC is taken on held rows only (harness convention,
           level AP on the score).

Combinations: per-scenario rank averages (equal + 0.25-step weight sweeps),
Borda(=equal rank avg)/min-rank, noisy-OR of calibrated probabilities,
logit-mean, and a LOBO logistic stack {logit p_cens, logit p_tp, ranks}
(control: the stack must beat BOTH inputs AND the rank average).

All rank transforms are within-scenario percentile ranks — rank-based by
construction to dodge the cross-building level-inflation trap.

Usage:
    OMP_NUM_THREADS=2 python tools/ensemble_eval.py \
        [--matrix outputs/ensemble_matrix.parquet] \
        [--report outputs/ensemble_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HARD_FOLDS = (
    "hard_large5",
    "hard_small10",
    "hard_mosteol5",
    "hard_hirate6",
    "hard_betashift5",
)
HARD_BAR = 0.428
MID_BAR = 0.292  # best single (twophase) mid-block top-12
AP_BAR = 0.3843  # best single (twophase) pooled AP
EPS = 1e-9


def pct_rank(scenario: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Within-scenario percentile rank in (0, 1]; NaN scores sink to the bottom."""
    filled = np.where(np.isfinite(score), score, -np.inf)
    out = np.zeros_like(filled, dtype=float)
    for s in np.unique(scenario):
        mask = scenario == s
        out[mask] = rankdata(filled[mask], method="average") / mask.sum()
    return out


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def top_k_rate(scenario, due, score, lo, hi, k=12) -> float:
    rates = []
    for s in range(lo, hi + 1):
        mask = scenario == s
        if not mask.any():
            continue
        order = np.argsort(-score[mask], kind="stable")[:k]
        rates.append(float(due[mask][order].sum()) / k)
    return float(np.mean(rates)) if rates else float("nan")


def gate_a(scenario, due, score) -> dict:
    blocks = {}
    for lo, hi, label in ((0, 15, "open"), (16, 31, "mid"), (32, 47, "late")):
        mask = (scenario >= lo) & (scenario <= hi)
        blocks[label] = round(float(average_precision_score(due[mask], score[mask])), 4)
    return {
        "ap": round(float(average_precision_score(due, score)), 4),
        "ap_blocks": blocks,
        "top12_open": round(top_k_rate(scenario, due, score, 0, 15), 4),
        "top12_mid": round(top_k_rate(scenario, due, score, 16, 31), 4),
        "top12_late": round(top_k_rate(scenario, due, score, 32, 47), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("outputs/ensemble_matrix.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("outputs/ensemble_matrix_meta.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/ensemble_results.json"))
    args = parser.parse_args()

    started = time.time()
    m = pd.read_parquet(REPO_ROOT / args.matrix)
    meta = json.loads((REPO_ROOT / args.meta).read_text())
    scenario = m.scenario.to_numpy(dtype=int)
    due = m.due.to_numpy(dtype=int)
    building = m.building.to_numpy(dtype=object)

    # ---- component levels ----------------------------------------------------
    p_c = m.p_cens.to_numpy(dtype=float)            # calibrated cens (production OOF)
    p_t = m.p_tp.to_numpy(dtype=float)              # twophase p42 (calibrated OOF)
    p_q = m.p_qh.to_numpy(dtype=float)              # qhead (raw OOF level)
    raw_min3 = m.raw_min3.to_numpy(dtype=float)
    raw_slope7 = m.raw_slope7.to_numpy(dtype=float)

    # ---- component ranks -------------------------------------------------------
    r_c = pct_rank(scenario, p_c)
    r_t = pct_rank(scenario, p_t)
    r_q = pct_rank(scenario, p_q)
    r_raw = pct_rank(
        scenario,
        0.5 * pct_rank(scenario, -raw_min3) + 0.5 * pct_rank(scenario, -raw_slope7),
    )

    # decorrelation evidence (pooled Spearman on within-scenario ranks)
    comps = {"cens": r_c, "tp": r_t, "qh": r_q, "raw": r_raw}
    corr = {
        f"{a}~{b}": round(float(spearmanr(comps[a], comps[b]).statistic), 3)
        for i, a in enumerate(comps)
        for b in list(comps)[i + 1:]
    }

    # ---- hard-holdout machinery ------------------------------------------------
    hard_held = {
        fold: np.isin(building, np.asarray(meta["hard_folds"][fold], dtype=object))
        for fold in HARD_FOLDS
    }
    hard_c_raw = {fold: m[f"p_censhard_{fold}"].to_numpy(dtype=float) for fold in HARD_FOLDS}
    hard_r_c = {fold: pct_rank(scenario, hard_c_raw[fold]) for fold in HARD_FOLDS}

    def gate_b(score_of_fold) -> dict:
        """score_of_fold(fold) -> full-frame score; AP on held rows per fold."""
        per = {}
        for fold in HARD_FOLDS:
            held = hard_held[fold]
            score = score_of_fold(fold)
            per[fold] = round(float(average_precision_score(due[held], score[held])), 4)
        per["mean"] = round(float(np.mean([per[f] for f in HARD_FOLDS])), 4)
        return per

    results: dict[str, dict] = {}

    def record(name: str, score: np.ndarray, score_of_fold=None) -> None:
        entry = {"gate_a": gate_a(scenario, due, score)}
        if score_of_fold is not None:
            entry["gate_b"] = gate_b(score_of_fold)
        results[name] = entry
        g = entry["gate_a"]
        hb = entry.get("gate_b", {}).get("mean", float("nan"))
        print(
            f"{name:34s} AP {g['ap']:.4f}  mid12 {g['top12_mid']:.3f} "
            f"open {g['top12_open']:.3f} late {g['top12_late']:.3f}  hard {hb:.4f}",
            flush=True,
        )

    # ---- singles ---------------------------------------------------------------
    record("single_cens_cal", p_c, lambda f: hard_c_raw[f])
    record("single_tp", p_t, lambda f: p_t)
    record("single_qh", p_q, lambda f: p_q)
    record("single_raw", r_raw, lambda f: r_raw)

    # ---- rank averages (Borda = equal-weight rank average) ----------------------
    def rank_mix(weights: dict[str, float]) -> np.ndarray:
        total = sum(weights.values())
        return sum(w * comps[k] for k, w in weights.items()) / total

    def rank_mix_fold(weights: dict[str, float]):
        def build(fold: str) -> np.ndarray:
            total = sum(weights.values())
            parts = []
            for k, w in weights.items():
                parts.append(w * (hard_r_c[fold] if k == "cens" else comps[k]))
            return sum(parts) / total
        return build

    pairs = [
        ("rankavg_ct", {"cens": 1, "tp": 1}),
        ("rankavg_cq", {"cens": 1, "qh": 1}),
        ("rankavg_tq", {"tp": 1, "qh": 1}),
        ("rankavg_cr", {"cens": 1, "raw": 1}),
        ("rankavg_tr", {"tp": 1, "raw": 1}),
        ("rankavg_ctq", {"cens": 1, "tp": 1, "qh": 1}),
        ("rankavg_ctr", {"cens": 1, "tp": 1, "raw": 1}),
        ("rankavg_ctqr", {"cens": 1, "tp": 1, "qh": 1, "raw": 1}),
    ]
    for name, weights in pairs:
        record(name, rank_mix(weights), rank_mix_fold(weights))

    # weight sweep ct (0.25 steps; 0/1 are the singles above)
    for wt in (0.25, 0.5, 0.75):
        weights = {"cens": 1 - wt, "tp": wt}
        record(f"rankmix_c{1-wt:.2f}_t{wt:.2f}", rank_mix(weights), rank_mix_fold(weights))

    # weight sweep ctq on the 0.25 grid (interior points not already covered)
    for wc, wt, wq in (
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
        (0.5, 0.5, 0.0),  # == c0.50_t0.50, kept out (duplicate)
    ):
        if wq == 0.0:
            continue
        weights = {"cens": wc, "tp": wt, "qh": wq}
        record(f"rankmix_c{wc}_t{wt}_q{wq}", rank_mix(weights), rank_mix_fold(weights))
    # ctr sweep with raw as a light third voice
    for wc, wt, wr in ((0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.375, 0.375, 0.25)):
        weights = {"cens": wc, "tp": wt, "raw": wr}
        record(f"rankmix_c{wc}_t{wt}_r{wr}", rank_mix(weights), rank_mix_fold(weights))

    # ---- min-rank (union: best rank across rankers) -----------------------------
    record("minrank_ct", np.maximum(r_c, r_t),
           lambda f: np.maximum(hard_r_c[f], r_t))
    record("minrank_ctq", np.maximum.reduce([r_c, r_t, r_q]),
           lambda f: np.maximum.reduce([hard_r_c[f], r_t, r_q]))

    # ---- noisy-OR of calibrated probabilities -----------------------------------
    record("noisyor_ct", 1.0 - (1.0 - p_c) * (1.0 - p_t),
           lambda f: 1.0 - (1.0 - hard_c_raw[f]) * (1.0 - p_t))
    record("noisyor_ctq", 1.0 - (1.0 - p_c) * (1.0 - p_t) * (1.0 - p_q),
           lambda f: 1.0 - (1.0 - hard_c_raw[f]) * (1.0 - p_t) * (1.0 - p_q))

    # ---- logit mean --------------------------------------------------------------
    for wt in (0.25, 0.5, 0.75, 0.875):
        record(
            f"logitmean_c{1-wt:.3g}_t{wt:.3g}",
            (1 - wt) * logit(p_c) + wt * logit(p_t),
            (lambda wt_: lambda f: (1 - wt_) * logit(hard_c_raw[f]) + wt_ * logit(p_t))(wt),
        )

    # logit mean including qhead (0.25 grid, tp kept dominant or equal)
    for wc, wt, wq in (
        (0.25, 0.5, 0.25),
        (0.0, 0.75, 0.25),
        (0.25, 0.75, 0.25),  # renormalised below
        (0.125, 0.75, 0.125),
        (0.0, 0.5, 0.5),
    ):
        total = wc + wt + wq
        wc_, wt_, wq_ = wc / total, wt / total, wq / total
        record(
            f"logitmean_c{wc}_t{wt}_q{wq}",
            wc_ * logit(p_c) + wt_ * logit(p_t) + wq_ * logit(p_q),
            (lambda a, b, c: lambda f: a * logit(hard_c_raw[f]) + b * logit(p_t) + c * logit(p_q))(
                wc_, wt_, wq_
            ),
        )

    # ---- max of calibrated p (union without OR inflation) -------------------------
    record("maxp_ct", np.maximum(p_c, p_t), lambda f: np.maximum(hard_c_raw[f], p_t))

    # ---- within-scenario z-scored logit mix (level spacing, scale-free) -----------
    def zlogit(scenario_arr: np.ndarray, p: np.ndarray) -> np.ndarray:
        z = logit(p)
        out = np.zeros_like(z)
        for s in np.unique(scenario_arr):
            mask = scenario_arr == s
            mu, sd = z[mask].mean(), z[mask].std()
            out[mask] = (z[mask] - mu) / max(sd, EPS)
        return out

    z_c, z_t, z_q = zlogit(scenario, p_c), zlogit(scenario, p_t), zlogit(scenario, p_q)
    z_c_hard = {f: zlogit(scenario, hard_c_raw[f]) for f in HARD_FOLDS}
    for wt in (0.5, 0.75):
        record(
            f"zlogit_c{1-wt:.2f}_t{wt:.2f}",
            (1 - wt) * z_c + wt * z_t,
            (lambda wt_: lambda f: (1 - wt_) * z_c_hard[f] + wt_ * z_t)(wt),
        )
    record(
        "zlogit_c0.25_t0.5_q0.25",
        0.25 * z_c + 0.5 * z_t + 0.25 * z_q,
        lambda f: 0.25 * z_c_hard[f] + 0.5 * z_t + 0.25 * z_q,
    )

    # ---- LOBO logistic stack ------------------------------------------------------
    remaining = m.remaining.to_numpy(dtype=float)

    def stack_features(pc: np.ndarray, rc: np.ndarray, kind: str) -> np.ndarray:
        cols = [logit(pc), logit(p_t), rc, r_t]
        if kind in ("ctq", "ctq_rem"):
            cols += [logit(p_q), r_q]
        if kind == "ctq_rem":
            rem = remaining / 100.0
            cols += [rem, rem * logit(p_t), rem * logit(pc)]
        return np.column_stack(cols)

    def lobo_stack(kind: str) -> np.ndarray:
        X = stack_features(p_c, r_c, kind)
        out = np.zeros(len(due), dtype=float)
        for b in np.unique(building):
            held = building == b
            model = LogisticRegression(C=1.0, max_iter=2000)
            model.fit(X[~held], due[~held])
            out[held] = model.predict_proba(X[held])[:, 1]
        return out

    def hard_stack(kind: str):
        def build(fold: str) -> np.ndarray:
            X = stack_features(hard_c_raw[fold], hard_r_c[fold], kind)
            held = hard_held[fold]
            model = LogisticRegression(C=1.0, max_iter=2000)
            model.fit(X[~held], due[~held])
            out = np.zeros(len(due), dtype=float)
            out[held] = model.predict_proba(X[held])[:, 1]
            return out
        return build

    record("stack_lobo_ct", lobo_stack("ct"), hard_stack("ct"))
    record("stack_lobo_ctq", lobo_stack("ctq"), hard_stack("ctq"))
    record("stack_lobo_ctq_rem", lobo_stack("ctq_rem"), hard_stack("ctq_rem"))

    # ---- round 2: protect tp's mid-block signal ------------------------------------
    # finer tp-dominant logit grid
    for wc, wt, wq in (
        (0.1, 0.8, 0.1),
        (0.05, 0.85, 0.1),
        (0.05, 0.9, 0.05),
        (0.15, 0.7, 0.15),
        (0.2, 0.6, 0.2),
    ):
        record(
            f"logitmean_c{wc}_t{wt}_q{wq}",
            wc * logit(p_c) + wt * logit(p_t) + wq * logit(p_q),
            (lambda a, b, c: lambda f: a * logit(hard_c_raw[f]) + b * logit(p_t) + c * logit(p_q))(
                wc, wt, wq
            ),
        )

    # median consensus in z space
    record(
        "median_z_ctq",
        np.median(np.column_stack([z_c, z_t, z_q]), axis=1),
        lambda f: np.median(np.column_stack([z_c_hard[f], z_t, z_q]), axis=1),
    )

    # promote-only / demote-only around tp (blend = mean z of cens+qhead)
    blend = 0.5 * (z_c + z_q)
    blend_hard = {f: 0.5 * (z_c_hard[f] + z_q) for f in HARD_FOLDS}
    for a in (0.25, 0.5):
        record(
            f"tp_promote_a{a}",
            z_t + a * np.maximum(blend - z_t, 0.0),
            (lambda a_: lambda f: z_t + a_ * np.maximum(blend_hard[f] - z_t, 0.0))(a),
        )
        record(
            f"tp_demote_a{a}",
            z_t + a * np.minimum(blend - z_t, 0.0),
            (lambda a_: lambda f: z_t + a_ * np.minimum(blend_hard[f] - z_t, 0.0))(a),
        )

    # remaining-keyed logit weights (deployment axis: same as RemainingCalibration;
    # regimes = open >225 d, mid 115-225 d, late <=115 d of remaining observation)
    def remkeyed_factory(w_open, w_mid, w_late):
        def build(pc: np.ndarray) -> np.ndarray:
            lc, lt, lq = logit(pc), logit(p_t), logit(p_q)
            def mix(w):
                return w[0] * lc + w[1] * lt + w[2] * lq
            return np.where(
                remaining > 225.0, mix(w_open),
                np.where(remaining > 115.0, mix(w_mid), mix(w_late)),
            )
        return build

    remkeyed = remkeyed_factory((0.25, 0.5, 0.25), (0.05, 0.9, 0.05), (0.125, 0.75, 0.125))
    record("logitmean_remkeyed", remkeyed(p_c), lambda f: remkeyed(hard_c_raw[f]))

    remkeyed_pm = remkeyed_factory((0.25, 0.5, 0.25), (0.0, 1.0, 0.0), (0.125, 0.75, 0.125))
    record("remkeyed_puremid", remkeyed_pm(p_c), lambda f: remkeyed_pm(hard_c_raw[f]))

    remkeyed_qm = remkeyed_factory((0.25, 0.5, 0.25), (0.0, 0.9, 0.1), (0.125, 0.75, 0.125))
    record("remkeyed_qmid", remkeyed_qm(p_c), lambda f: remkeyed_qm(hard_c_raw[f]))

    # attribution: which second voice carries the mid drag / hard gain
    record(
        "logitmean_c0_t0.9_q0.1",
        0.9 * logit(p_t) + 0.1 * logit(p_q),
        lambda f: 0.9 * logit(p_t) + 0.1 * logit(p_q),
    )
    record(
        "logitmean_c0.1_t0.9_q0",
        0.1 * logit(p_c) + 0.9 * logit(p_t),
        lambda f: 0.1 * logit(hard_c_raw[f]) + 0.9 * logit(p_t),
    )

    # ---- deployment-shaped permutation: mixed ORDER, single-model LEVELS ------------
    def permute_levels(levels: np.ndarray, order_score: np.ndarray) -> np.ndarray:
        out = np.empty_like(levels, dtype=float)
        for s in np.unique(scenario):
            mask = np.flatnonzero(scenario == s)
            sorted_levels = np.sort(levels[mask])[::-1]
            rank_pos = np.argsort(-order_score[mask], kind="stable")
            out[mask[rank_pos]] = sorted_levels
        return out

    mix_best = 0.125 * logit(p_c) + 0.75 * logit(p_t) + 0.125 * logit(p_q)
    mix_best_hard = {
        f: 0.125 * logit(hard_c_raw[f]) + 0.75 * logit(p_t) + 0.125 * logit(p_q)
        for f in HARD_FOLDS
    }
    record(
        "perm_tpLevels_mixOrder",
        permute_levels(p_t, mix_best),
        lambda f: permute_levels(p_t, mix_best_hard[f]),
    )
    record(
        "perm_censLevels_mixOrder",
        permute_levels(p_c, mix_best),
        lambda f: permute_levels(hard_c_raw[f], mix_best_hard[f]),
    )
    rmix = rank_mix({"cens": 0.25, "tp": 0.75})
    rmix_hard = rank_mix_fold({"cens": 0.25, "tp": 0.75})
    record(
        "perm_tpLevels_rankmix2575Order",
        permute_levels(p_t, rmix),
        lambda f: permute_levels(p_t, rmix_hard(f)),
    )
    record(
        "perm_tpLevels_remkeyedPuremidOrder",
        permute_levels(p_t, remkeyed_pm(p_c)),
        lambda f: permute_levels(p_t, remkeyed_pm(hard_c_raw[f])),
    )
    record(
        "perm_censLevels_remkeyedPuremidOrder",
        permute_levels(p_c, remkeyed_pm(p_c)),
        lambda f: permute_levels(hard_c_raw[f], remkeyed_pm(hard_c_raw[f])),
    )

    # ---- robustness of the leading family vs best single ----------------------------
    def jackknife(score: np.ndarray, reference: np.ndarray) -> dict:
        """Leave-one-scenario-out pooled-AP delta + per-scenario mid top-12 delta."""
        deltas = []
        for s in np.unique(scenario):
            keep = scenario != s
            a = average_precision_score(due[keep], score[keep])
            b = average_precision_score(due[keep], reference[keep])
            deltas.append(a - b)
        deltas = np.asarray(deltas)
        mid_delta = []
        for s in range(16, 32):
            mask = scenario == s
            top_a = np.argsort(-score[mask], kind="stable")[:12]
            top_b = np.argsort(-reference[mask], kind="stable")[:12]
            mid_delta.append(int(due[mask][top_a].sum()) - int(due[mask][top_b].sum()))
        return {
            "loo_ap_delta_min": round(float(deltas.min()), 4),
            "loo_ap_delta_max": round(float(deltas.max()), 4),
            "loo_ap_delta_positive_frac": round(float((deltas > 0).mean()), 3),
            "mid_top12_catch_delta_per_scen": mid_delta,
            "mid_top12_catch_delta_sum": int(np.sum(mid_delta)),
        }

    results["robustness_remkeyed_puremid_vs_tp"] = jackknife(remkeyed_pm(p_c), p_t)
    results["robustness_logitfam_hard_family"] = {
        "note": "hard means across the whole logit-mean family (selection-free)",
        "members": {
            name: entry["gate_b"]["mean"]
            for name, entry in results.items()
            if name.startswith("logitmean_") and "gate_b" in entry
        },
    }

    # ---- comparator: production cens OOF on the hard groups' rows ------------------
    results["comparator_cens_prod_on_hard"] = {
        "gate_b": gate_b(lambda f: p_c),
        "note": "production 5-fold OOF cal cens restricted to hard rows "
        "(protocol-optimistic vs the 0.428 refit bar; context only)",
    }
    print(
        "comparator cens-prod-on-hard      "
        + str(results["comparator_cens_prod_on_hard"]["gate_b"]),
        flush=True,
    )

    # ---- verdict machinery ---------------------------------------------------------
    for name, entry in results.items():
        if "gate_a" not in entry:
            continue
        g, hb = entry["gate_a"], entry.get("gate_b", {})
        entry["gates"] = {
            "a_ap_beats_best_single": g["ap"] > AP_BAR,
            "a_mid12_beats_best_single": g["top12_mid"] > MID_BAR,
            "b_hard_mean_ge_bar": bool(hb and hb["mean"] >= HARD_BAR),
        }

    report = {
        "spearman_rank_correlations": corr,
        "bars": {"ap_best_single": AP_BAR, "mid12_best_single": MID_BAR, "hard": HARD_BAR},
        "results": results,
        "seconds": round(time.time() - started, 1),
    }
    (REPO_ROOT / args.report).write_text(json.dumps(report, indent=2))
    print(f"\ncorrelations: {corr}")
    print(f"wrote {args.report} ({time.time()-started:.0f}s)")


if __name__ == "__main__":
    main()
