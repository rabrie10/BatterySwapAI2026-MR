"""ROADBLOCK L1: information-frontier bound on the 48-scenario mean.

Computes, in the exact analytic convention of tools/xaware_rule_lab.py
(score_selection: best-day timing, evaluator emergency queue, 4.1+1.5 op per
planned swap, isolated emergency visit op per miss):

  ORACLE            swap exactly the due set (unrestricted foresight)
  GOD-VISIBLE       perfect ranking (true recorded days-to-EOL) restricted to
                    batteries with ANY signal at cutoff: p_cal >= 0.001 OR the
                    dark-decay gate OR the raw-dip gate; optimal k per scenario
  GOD-VISIBLE cap15 same, k restricted to <= 15
  variants          eligibility p_cal>=0.001 without gates; p_cal>=0.02
  MODEL-RANK        the actual calibrated ranking (p_cal desc), optimal k,
                    k<=15, fixed k=15, and the shipped budget
                    min(15, ceil(1.6*sum(p)+1))

Inputs: outputs/research_rowfeat.parquet + dataset/train/scenarios.json.
Output: outputs/roadblock_l1.json + printed tables. No planner runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HORIZON = 42
OP_PER_SWAP = 4.1
CAPACITY_PER_SWAP = 1.5
CATCH_EARLY_DAYS = 5.0
CAP = 15


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "outputs" / "research_rowfeat.parquet")
    scenarios = json.load(open(ROOT / "dataset" / "train" / "scenarios.json"))

    # Per-scenario emergency offset and per-building emergency visit op cost.
    emerg_offset = {}
    c_em_of = {}
    for index, sc in enumerate(scenarios):
        start = pd.Timestamp(sc["start_time"]).normalize()
        settings = sc["settings"]
        horizon = int(settings["planning_window_days"])
        window_close = (start + pd.Timedelta(days=horizon)).normalize()
        emerg_offset[index] = horizon + (6 - window_close.weekday())
        travel = pd.DataFrame(sc["travel_costs"])
        from_base = travel[travel["from"] == settings["base_location"]].set_index(
            "to"
        )["hours"]
        base = settings["base_location"]
        ot_start = float(settings["overtime_start"])
        lookup = {}
        for building, t_out in from_base.items():
            hours = 0.75 if building == base else 2.0 * float(t_out) + 1.75
            lookup[building] = hours + 2.0 * max(hours - ot_start, 0.0)
        c_em_of[index] = lookup

    frame["emerg_offset"] = frame["scenario"].map(emerg_offset).astype(float)
    frame["c_em"] = [
        c_em_of[s].get(b, np.nan) for s, b in zip(frame["scenario"], frame["building"])
    ]
    assert frame["c_em"].notna().all(), "building missing from travel matrix"

    # Effective EOL: recorded, else the evaluator substitute end_time + 30 d.
    eff = frame["days_to_eol"].to_numpy(dtype=float)
    sub = frame["remaining"].to_numpy(dtype=float) + 30.0
    frame["d2e_eff"] = np.where(np.isfinite(eff), eff, sub)
    frame["d2e_rec"] = eff  # NaN when never recorded

    # Signal gates (bsai.calibrate.ResurrectionGate thresholds).
    m = frame["margin"].to_numpy(dtype=float)
    st = frame["staleness"].to_numpy(dtype=float)
    raw3 = frame["raw_min3"].to_numpy(dtype=float)
    pc = frame["p_cal"].to_numpy(dtype=float)
    frame["gate_dark"] = (st > 30.0) & ((m - 0.001 * st) < 0.02)
    frame["gate_dip"] = (
        np.isfinite(raw3) & (raw3 < 2.40) & (m > 0.03) & (pc < 0.10)
    )
    frame["block"] = frame["scenario"] // 8
    return frame


def score_selection(sub: pd.DataFrame, chosen: np.ndarray) -> dict:
    due = sub["due"].to_numpy()
    d2e = sub["d2e_eff"].to_numpy(dtype=float)
    hits = chosen & due
    waste = chosen & ~due
    miss = due & ~chosen

    early = 0.5 * np.minimum(CATCH_EARLY_DAYS, np.clip(d2e[hits], 0.0, None)).sum()
    early += 0.5 * np.clip(d2e[waste] - HORIZON, 0.0, None).sum()
    late_at_swap = 10.0 * np.clip(-d2e[waste], 0.0, None).sum()

    offset = float(sub["emerg_offset"].iloc[0])
    miss_ids = sub.loc[miss, "battery"].to_numpy()
    order = np.argsort(miss_ids)
    queue = np.arange(len(miss_ids), dtype=float)
    late = 10.0 * np.clip(offset + queue - d2e[miss][order], 0.0, None).sum()

    op_planned = (OP_PER_SWAP + CAPACITY_PER_SWAP) * float(chosen.sum())
    op_emerg = float(sub.loc[miss, "c_em"].sum())
    return {
        "swaps": int(chosen.sum()),
        "due": int(due.sum()),
        "hits": int(hits.sum()),
        "missed": int(miss.sum()),
        "early": float(early),
        "late": float(late + late_at_swap),
        "op_planned": op_planned,
        "op_emerg": op_emerg,
        "total": float(early + late + late_at_swap + op_planned + op_emerg),
    }


def scan_ranking(sub: pd.DataFrame, order: np.ndarray, kmax: int | None = None):
    """Cost at every prefix size of a fixed ranking; returns list of records."""
    n = len(order)
    if kmax is None:
        kmax = n
    kmax = min(kmax, n)
    records = []
    chosen = np.zeros(len(sub), dtype=bool)
    records.append((0, score_selection(sub, chosen)))
    for k in range(1, kmax + 1):
        chosen[order[k - 1]] = True
        records.append((k, score_selection(sub, chosen.copy())))
    return records


def best_of(records, klimit=None):
    pool = [r for r in records if klimit is None or r[0] <= klimit]
    k, rec = min(pool, key=lambda item: item[1]["total"])
    rec = dict(rec)
    rec["k"] = k
    return rec


def god_order(sub: pd.DataFrame, eligible: np.ndarray) -> np.ndarray:
    """Perfect ranking of the eligible: recorded days-to-EOL ascending, never-
    recorded last (by substitute date), so dues come first, imminent first."""
    rec = sub["d2e_rec"].to_numpy(dtype=float)
    eff = sub["d2e_eff"].to_numpy(dtype=float)
    key = np.where(np.isfinite(rec), rec, 1e6 + eff)
    idx = np.flatnonzero(eligible)
    return idx[np.argsort(key[idx], kind="stable")]


def model_order(sub: pd.DataFrame, eligible: np.ndarray) -> np.ndarray:
    p = sub["p_cal"].to_numpy(dtype=float)
    idx = np.flatnonzero(eligible)
    return idx[np.argsort(-p[idx], kind="stable")]


def main() -> None:
    frame = load_frame()
    groups = [
        (int(s), sub.reset_index(drop=True))
        for s, sub in frame.groupby("scenario", sort=True)
    ]

    per_rule: dict[str, list[dict]] = {}

    def record(rule: str, scenario: int, block: int, rec: dict) -> None:
        rec = dict(rec)
        rec["scenario"] = scenario
        rec["block"] = block
        per_rule.setdefault(rule, []).append(rec)

    for scenario, sub in groups:
        block = int(sub["block"].iloc[0])
        due = sub["due"].to_numpy()
        pc = sub["p_cal"].to_numpy(dtype=float)
        gate = sub["gate_dark"].to_numpy() | sub["gate_dip"].to_numpy()

        # ORACLE: swap exactly the due set.
        record("ORACLE", scenario, block, score_selection(sub, due.copy()))

        # GOD-VISIBLE with gates.
        elig = (pc >= 0.001) | gate
        order = god_order(sub, elig)
        recs = scan_ranking(sub, order)
        record("GOD_VIS_GATES_opt", scenario, block, best_of(recs))
        record("GOD_VIS_GATES_cap15", scenario, block, best_of(recs, CAP))
        # eligibility stats
        vis_due = int((due & elig).sum())
        record(
            "ELIGIBILITY",
            scenario,
            block,
            {
                "eligible": int(elig.sum()),
                "due": int(due.sum()),
                "visible_due": vis_due,
                "invisible_due": int(due.sum()) - vis_due,
                "gate_only_due": int((due & gate & (pc < 0.001)).sum()),
                "total": 0.0,
            },
        )

        # GOD-VISIBLE without gates.
        elig_ng = pc >= 0.001
        recs_ng = scan_ranking(sub, god_order(sub, elig_ng))
        record("GOD_VIS_NOGATES_opt", scenario, block, best_of(recs_ng))
        record("GOD_VIS_NOGATES_cap15", scenario, block, best_of(recs_ng, CAP))

        # GOD-VISIBLE at the auditor's invisibility line 0.02.
        elig_02 = (pc >= 0.02) | gate
        recs_02 = scan_ranking(sub, god_order(sub, elig_02))
        record("GOD_VIS_p02_opt", scenario, block, best_of(recs_02))

        # MODEL ranking (calibrated p), everything eligible.
        all_elig = np.ones(len(sub), dtype=bool)
        m_order = model_order(sub, all_elig)
        m_recs = scan_ranking(sub, m_order, kmax=40)
        record("MODEL_opt", scenario, block, best_of(m_recs))
        record("MODEL_cap15", scenario, block, best_of(m_recs, CAP))
        fixed15 = [r for r in m_recs if r[0] == min(15, len(sub))][0][1]
        fixed15 = dict(fixed15)
        fixed15["k"] = min(15, len(sub))
        record("MODEL_k15", scenario, block, fixed15)
        budget = int(min(CAP, np.ceil(1.6 * pc.sum() + 1)))
        kb = min(budget, len(m_recs) - 1)
        shipped = dict(m_recs[kb][1])
        shipped["k"] = kb
        record("MODEL_budget", scenario, block, shipped)

        # MODEL ranking with gate floors folded in (max(p, gate rate)).
        p_gate = pc.copy()
        p_gate = np.maximum(
            p_gate, np.where(sub["gate_dark"].to_numpy(), 0.45, 0.0)
        )
        p_gate = np.maximum(p_gate, np.where(sub["gate_dip"].to_numpy(), 0.40, 0.0))
        idx = np.argsort(-p_gate, kind="stable")
        g_recs = scan_ranking(sub, idx, kmax=40)
        record("MODEL_GATE_cap15", scenario, block, best_of(g_recs, CAP))

    # ------------------------------------------------------------- summaries
    def summarize(rule: str) -> dict:
        rows = pd.DataFrame(per_rule[rule])
        out = {
            "mean_total": round(float(rows["total"].mean()), 1),
            "block_means": [
                round(float(v), 1)
                for v in rows.groupby("block")["total"].mean().to_numpy()
            ],
        }
        for col in ("early", "late", "op_planned", "op_emerg"):
            if col in rows:
                out[f"mean_{col}"] = round(float(rows[col].mean()), 1)
        for col in ("swaps", "hits", "missed", "k"):
            if col in rows:
                out[f"mean_{col}"] = round(float(rows[col].mean()), 2)
        if "k" in rows:
            ks = rows["k"].to_numpy()
            out["k_distribution"] = {
                "min": int(ks.min()),
                "p25": float(np.percentile(ks, 25)),
                "median": float(np.median(ks)),
                "p75": float(np.percentile(ks, 75)),
                "max": int(ks.max()),
                "n_above_15": int((ks > 15).sum()),
            }
        return out

    payload: dict = {"convention": "xaware_rule_lab score_selection", "rules": {}}
    for rule in per_rule:
        if rule == "ELIGIBILITY":
            continue
        payload["rules"][rule] = summarize(rule)

    elig_rows = pd.DataFrame(per_rule["ELIGIBILITY"])
    payload["eligibility"] = {
        "mean_eligible_per_scenario": round(float(elig_rows["eligible"].mean()), 1),
        "mean_due": round(float(elig_rows["due"].mean()), 2),
        "mean_visible_due": round(float(elig_rows["visible_due"].mean()), 2),
        "mean_invisible_due": round(float(elig_rows["invisible_due"].mean()), 2),
        "mean_gate_only_due": round(float(elig_rows["gate_only_due"].mean()), 2),
        "block_invisible_due": [
            round(float(v), 2)
            for v in elig_rows.groupby("block")["invisible_due"].mean().to_numpy()
        ],
    }

    # cap-15 clipping cost for the god plan and the model plan.
    god_opt = pd.DataFrame(per_rule["GOD_VIS_GATES_opt"])
    god_cap = pd.DataFrame(per_rule["GOD_VIS_GATES_cap15"])
    payload["cap15_clip"] = {
        "god_scenarios_clipped": int((god_opt["k"] > 15).sum()),
        "god_mean_cost_of_cap": round(
            float((god_cap["total"] - god_opt["total"]).mean()), 2
        ),
        "god_max_cost_of_cap": round(
            float((god_cap["total"] - god_opt["total"]).max()), 1
        ),
    }
    model_opt = pd.DataFrame(per_rule["MODEL_opt"])
    model_cap = pd.DataFrame(per_rule["MODEL_cap15"])
    payload["cap15_clip"]["model_scenarios_clipped"] = int(
        (model_opt["k"] > 15).sum()
    )
    payload["cap15_clip"]["model_mean_cost_of_cap"] = round(
        float((model_cap["total"] - model_opt["total"]).mean()), 2
    )
    payload["cap15_clip"]["model_opt_k"] = summarize("MODEL_opt")["k_distribution"]

    out_path = ROOT / "outputs" / "roadblock_l1.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
