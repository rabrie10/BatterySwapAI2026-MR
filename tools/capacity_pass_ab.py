"""A/B control for the capacity post-pass.

Runs the exact ship validation with ``CompetitionPlanner._capacity_repair``
stubbed to the identity, writing outputs/capacity_baseline_rerun.json. Diffing
that against outputs/val_ship_final.json measures pure run-to-run drift
(CP-SAT's 1 s wall-clock termination under box load); diffing it against
outputs/val_capacity_pass.json measures the pass with drift mostly cancelled.

    python tools/capacity_pass_ab.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_solution.planner import CompetitionPlanner


def main() -> None:
    CompetitionPlanner._capacity_repair = (
        lambda self, plan, *args, **kwargs: plan
    )
    sys.argv = [
        "validate_v6.py",
        "--folds", "outputs/v8_folds_cens.joblib",
        "--model", "models/v8_cens.joblib",
        "--robust-samples", "0",
        "--local-search", "240",
        "--uncertain-search", "240",
        "--due-multiplier", "1.6",
        "--due-buffer", "1.0",
        "--max-planned", "15",
        "--report", "outputs/capacity_baseline_rerun.json",
    ]
    spec = importlib.util.spec_from_file_location(
        "validate_v6", REPO_ROOT / "tools" / "validate_v6.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
