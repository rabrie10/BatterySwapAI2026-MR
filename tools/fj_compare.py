"""Pair two validate_v6 reports scenario by scenario.

The 48 scenarios overlap heavily and differ enormously in difficulty -- the
opening ones carry a substitute end of life that makes a wasted swap worth ~182
and the closing ones ~17 -- so the mean of a difference is worth reading and the
difference of means is not. Everything here is paired on the scenario id.

    python tools/fj_compare.py --base outputs/fj_v8_baseline.json \
                               --candidate outputs/fj_tcn_plan_shipped.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPONENTS = ("early_swap", "late_swap", "battery_swap", "travel", "overtime",
              "daily_limit", "weekly_limit", "building_change", "room_change")


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return ({str(row["scenario"]): row for row in payload["scenarios"]},
            payload["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", type=str, default="candidate")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    base_rows, base_summary = load(args.base)
    cand_rows, cand_summary = load(args.candidate)
    shared = sorted(set(base_rows) & set(cand_rows), key=lambda s: int(str(s).split("_")[-1]))
    if len(shared) != len(base_rows) or len(shared) != len(cand_rows):
        print(f"warning: pairing on {len(shared)} shared scenarios of "
              f"{len(base_rows)} / {len(cand_rows)}")

    old = np.array([base_rows[s]["total_cost"] for s in shared])
    new = np.array([cand_rows[s]["total_cost"] for s in shared])
    delta = new - old
    t = float(delta.mean() / (delta.std(ddof=1) / np.sqrt(delta.size))) if delta.size > 1 else float("nan")
    wins = int((delta < 0).sum())
    losses = int((delta > 0).sum())

    print(f"{args.base.name}  ->  {args.candidate.name}   ({len(shared)} scenarios)")
    print(f"  mean total   {old.mean():9.2f}  ->  {new.mean():9.2f}   "
          f"delta {delta.mean():+8.2f}")
    print(f"  paired t {t:+.2f},  {wins} wins / {losses} losses"
          f"{' / ' + str(len(shared) - wins - losses) + ' ties' if len(shared) - wins - losses else ''}")

    print("\n  component            base    candidate      delta")
    rows = {}
    for name in COMPONENTS:
        a = np.array([base_rows[s].get(name, 0.0) for s in shared])
        b = np.array([cand_rows[s].get(name, 0.0) for s in shared])
        rows[name] = {"base": round(float(a.mean()), 2),
                      "candidate": round(float(b.mean()), 2),
                      "delta": round(float((b - a).mean()), 2)}
        print(f"  {name:<18} {a.mean():9.2f} {b.mean():11.2f} {(b - a).mean():+11.2f}")

    print("\n  decisions           base    candidate      delta")
    decisions = {}
    for name in ("served", "due", "hit", "missed"):
        a = np.array([base_rows[s].get(name, 0.0) for s in shared], dtype=float)
        b = np.array([cand_rows[s].get(name, 0.0) for s in shared], dtype=float)
        decisions[name] = {"base": round(float(a.mean()), 3),
                           "candidate": round(float(b.mean()), 3),
                           "delta": round(float((b - a).mean()), 3)}
        print(f"  {name:<18} {a.mean():9.3f} {b.mean():11.3f} {(b - a).mean():+11.3f}")
    served_a = np.array([base_rows[s]["served"] for s in shared], dtype=float)
    served_b = np.array([cand_rows[s]["served"] for s in shared], dtype=float)
    hit_a = np.array([base_rows[s]["hit"] for s in shared], dtype=float)
    hit_b = np.array([cand_rows[s]["hit"] for s in shared], dtype=float)
    due_a = np.array([base_rows[s]["due"] for s in shared], dtype=float)
    due_b = np.array([cand_rows[s]["due"] for s in shared], dtype=float)
    precision = (float(hit_a.sum() / served_a.sum()), float(hit_b.sum() / served_b.sum()))
    recall = (float(hit_a.sum() / due_a.sum()), float(hit_b.sum() / due_b.sum()))
    waste = (float((served_a - hit_a).sum() / len(shared)),
             float((served_b - hit_b).sum() / len(shared)))
    print(f"  {'precision':<18} {precision[0]:9.3f} {precision[1]:11.3f} "
          f"{precision[1] - precision[0]:+11.3f}")
    print(f"  {'recall':<18} {recall[0]:9.3f} {recall[1]:11.3f} "
          f"{recall[1] - recall[0]:+11.3f}")
    print(f"  {'wasted swaps':<18} {waste[0]:9.3f} {waste[1]:11.3f} "
          f"{waste[1] - waste[0]:+11.3f}")

    # Six non-overlapping blocks of eight consecutive scenarios: the project's
    # standing check that a mean is not one scenario's accident.
    print("\n  blocks of 8 (mean delta):", end=" ")
    blocks = []
    for start in range(0, len(shared) - 7, 8):
        value = float(delta[start:start + 8].mean())
        blocks.append(round(value, 1))
        print(f"{value:+.1f}", end="  ")
    print(f"\n  blocks improved: {sum(1 for b in blocks if b < 0)}/{len(blocks)}")

    print(f"\n  runtime  base {base_summary['runtime']['mean_seconds_per_scenario']:.2f} s/scenario"
          f"  candidate {cand_summary['runtime']['mean_seconds_per_scenario']:.2f} s/scenario"
          f"  (projected for 96: {cand_summary['runtime']['projected_minutes_for_96']:.1f} min)")

    if args.report:
        args.report.write_text(json.dumps({
            "base": str(args.base), "candidate": str(args.candidate),
            "n": len(shared), "base_mean": round(float(old.mean()), 2),
            "candidate_mean": round(float(new.mean()), 2),
            "delta": round(float(delta.mean()), 2), "t": round(t, 3),
            "wins": wins, "losses": losses, "blocks": blocks,
            "components": rows, "decisions": decisions,
            "precision": [round(p, 4) for p in precision],
            "recall": [round(r, 4) for r in recall],
            "wasted_swaps": [round(w, 3) for w in waste],
            "runtime_seconds_per_scenario": [
                base_summary["runtime"]["mean_seconds_per_scenario"],
                cand_summary["runtime"]["mean_seconds_per_scenario"]],
            "projected_minutes_for_96":
                cand_summary["runtime"]["projected_minutes_for_96"],
        }, indent=1))
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
