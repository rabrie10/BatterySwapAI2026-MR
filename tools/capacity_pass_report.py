"""Before/after component table for two validate_v6 report JSONs.

    python tools/capacity_pass_report.py outputs/val_ship_final.json outputs/val_capacity_pass.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    before_path, after_path = Path(sys.argv[1]), Path(sys.argv[2])
    before = json.load(before_path.open())
    after = json.load(after_path.open())
    b_sum, a_sum = before["summary"], after["summary"]

    print(f"before: {before_path}  after: {after_path}")
    print(f"{'component':16s} {'before':>9s} {'after':>9s} {'delta':>8s}")
    rows = [("mean_total_cost", b_sum["mean_total_cost"], a_sum["mean_total_cost"])]
    for component in b_sum["components"]:
        rows.append(
            (component, b_sum["components"][component], a_sum["components"][component])
        )
    for name, b_val, a_val in rows:
        print(f"{name:16s} {b_val:9.2f} {a_val:9.2f} {a_val - b_val:+8.2f}")
    print(
        f"{'runtime mean s':16s} {b_sum['runtime']['mean_seconds_per_scenario']:9.2f} "
        f"{a_sum['runtime']['mean_seconds_per_scenario']:9.2f}"
    )
    print(
        f"{'runtime max s':16s} {b_sum['runtime']['max_seconds_per_scenario']:9.2f} "
        f"{a_sum['runtime']['max_seconds_per_scenario']:9.2f}"
    )

    b_rows = {r["scenario"]: r for r in before["scenarios"]}
    print("\nper-scenario total deltas beyond +-1:")
    moved = 0
    for row in after["scenarios"]:
        b_row = b_rows.get(row["scenario"])
        if b_row is None:
            continue
        delta = row["total_cost"] - b_row["total_cost"]
        if abs(delta) > 1.0:
            moved += 1
            parts = {
                c: round(row[c] - b_row[c], 1)
                for c in (
                    "daily_limit",
                    "weekly_limit",
                    "overtime",
                    "travel",
                    "late_swap",
                    "early_swap",
                )
                if abs(row[c] - b_row[c]) > 0.05
            }
            print(f"  {row['scenario']:>5s} {delta:+9.1f}  {parts}")
    if not moved:
        print("  none")


if __name__ == "__main__":
    main()
