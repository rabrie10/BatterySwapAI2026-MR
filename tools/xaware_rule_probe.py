"""Fast probes on the cached x-aware frame: validation vs ranking_v7 conventions
and fine sweeps around the winning family without re-running the forecaster.

    python tools/xaware_rule_probe.py --cache <scratch>/xaware_frame.parquet --mode validate
    python tools/xaware_rule_probe.py --cache <scratch>/xaware_frame.parquet --mode fine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.xaware_rule_lab import (  # noqa: E402
    G_FLAT,
    HORIZON,
    OP_PER_SWAP,
    CAPACITY_PER_SWAP,
    score_selection,
    top_k_mask,
)


def ranking_v7_convention(sub: pd.DataFrame, chosen: np.ndarray) -> dict:
    """Reproduce ranking_v7's exact cost formula for cross-validation."""
    due = sub["due"].to_numpy()
    d2e = sub["days_to_eol"].to_numpy()
    hits = chosen & due
    waste_days = np.clip(d2e[chosen & ~due] - HORIZON, 0.0, None)
    early = 0.5 * waste_days.sum() + 0.5 * 5.0 * hits.sum()
    miss = due & ~chosen
    late = 10.0 * np.clip(48.0 - d2e[miss], 0.0, None).sum()
    return {"early": float(early), "late": float(late), "timing": float(early + late),
            "k": int(chosen.sum()), "hits": int(hits.sum()), "due": int(due.sum())}


def validate(frame: pd.DataFrame) -> None:
    rows = []
    for scenario, sub in frame.groupby("scenario"):
        sub = sub.reset_index(drop=True)
        p = sub["p"].to_numpy()
        for k in (8, 10, 12, 15, 18, 21, 25, 30):
            chosen = top_k_mask(sub, p, k)
            rec = ranking_v7_convention(sub, chosen)
            rec.update(rule=f"k={k}")
            rows.append(rec)
        for t in (0.10, 0.15, 0.20, 0.26, 0.32, 0.40, 0.50, 0.62):
            chosen = p > t
            rec = ranking_v7_convention(sub, chosen)
            rec.update(rule=f"p>{t}")
            rows.append(rec)
    result = pd.DataFrame(rows)
    out = result.groupby("rule", sort=False).agg(
        swaps=("k", "mean"), early=("early", "mean"), late=("late", "mean"),
        timing=("timing", "mean"),
    ).round(1)
    out["early_per_swap"] = (
        result.groupby("rule", sort=False)["early"].sum()
        / result.groupby("rule", sort=False)["k"].sum().clip(lower=1)
    ).round(1)
    print(out.to_string())


def sweep_fine(frame: pd.DataFrame, rules: list[tuple[str, callable]]) -> pd.DataFrame:
    rows = []
    groups = [(s, sub.reset_index(drop=True)) for s, sub in frame.groupby("scenario")]
    for name, rule in rules:
        for scenario, sub in groups:
            chosen = np.asarray(rule(sub), dtype=bool)
            rec = score_selection(sub, chosen)
            rec.update(rule=name, scenario=scenario, block=int(sub["block"].iloc[0]))
            rows.append(rec)
    result = pd.DataFrame(rows)
    grouped = result.groupby("rule", sort=False)
    summary = grouped.agg(
        total=("total", "mean"), early=("early", "mean"), late=("late", "mean"),
        swaps=("swaps", "mean"), catches=("hits", "mean"), misses=("missed", "mean"),
    )
    summary["early_per_swap"] = grouped["early"].sum() / grouped["swaps"].sum().clip(lower=1)
    return summary.sort_values("total").round(2), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", default="validate", choices=["validate", "fine"])
    args = parser.parse_args()
    frame = pd.read_parquet(args.cache)
    if args.mode == "validate":
        validate(frame)
        return

    def tail_ev(sub, gamma=1.0):
        p = sub["p"].to_numpy()
        gain = gamma * p * (
            10.0 * (sub["emerg_offset"].to_numpy() - sub["e_due"].to_numpy())
            + sub["c_em"].to_numpy() - 2.5
        )
        early = 0.5 * (
            sub["q_obs"].to_numpy() * sub["mean_excess"].to_numpy()
            + sub["q_unobs"].to_numpy() * np.clip(sub["x_plan"].to_numpy(), 0.0, None)
        )
        return gain - early - (OP_PER_SWAP + CAPACITY_PER_SWAP)

    rules = [
        ("k19_anchor", lambda sub: top_k_mask(sub, sub["p"].to_numpy(), 19)),
        ("oracle_due", lambda sub: sub["due"].to_numpy().copy()),
    ]
    rules.append(
        (
            "tailev_g1_m-5_pmin0.02",
            lambda sub: (tail_ev(sub, 1.0) > -5.0) & (sub["p"].to_numpy() > 0.02),
        )
    )
    for c_op in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        for pmin in (0.02,):
            rules.append(
                (
                    f"flatg_tail_c{c_op:g}_pmin{pmin:g}",
                    lambda sub, c_op=c_op, pmin=pmin: (
                        sub["p"].to_numpy() * G_FLAT
                        > 0.5
                        * (
                            sub["q_obs"].to_numpy() * sub["mean_excess"].to_numpy()
                            + sub["q_unobs"].to_numpy()
                            * np.clip(sub["x_plan"].to_numpy(), 0.0, None)
                        )
                        + c_op
                    )
                    & (sub["p"].to_numpy() > pmin),
                )
            )
    for pmin in (0.01, 0.03, 0.05):
        rules.append(
            (
                f"flatg_tail_c2.5_pmin{pmin:g}",
                lambda sub, pmin=pmin: (
                    sub["p"].to_numpy() * G_FLAT
                    > 0.5
                    * (
                        sub["q_obs"].to_numpy() * sub["mean_excess"].to_numpy()
                        + sub["q_unobs"].to_numpy()
                        * np.clip(sub["x_plan"].to_numpy(), 0.0, None)
                    )
                    + 2.5
                )
                & (sub["p"].to_numpy() > pmin),
            )
        )
    for c_op in (2.5,):
        def r5b(sub, c_op=c_op):
            p = sub["p"].to_numpy()
            early = 0.5 * (
                sub["q_obs"].to_numpy() * sub["mean_excess"].to_numpy()
                + sub["q_unobs"].to_numpy()
                * np.clip(sub["x_plan"].to_numpy(), 0.0, None)
            )
            return p * G_FLAT > early + c_op
        rules.append((f"flatg_tail_c{c_op:g}", r5b))
    summary, per_scenario = sweep_fine(frame, rules)
    print(summary.to_string())

    # Paired comparison against the k=19 anchor: mean delta and its s.e.
    wide = per_scenario.pivot(index="scenario", columns="rule", values="total")
    anchor = wide["k19_anchor"]
    print("\npaired vs k=19 (negative = better):")
    for rule in summary.index:
        if rule == "k19_anchor":
            continue
        delta = wide[rule] - anchor
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        print(f"  {rule:28s} mean_delta={delta.mean():8.1f}  se={se:6.1f}")


if __name__ == "__main__":
    main()
