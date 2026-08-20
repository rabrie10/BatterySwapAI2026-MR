"""Official BatterySwapAI submission entry point."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from batteryswap_public.interfaces import Planner
from batteryswap_public.utils import make_submissions

from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig


DEFAULT_HAZARD_FORECASTER_PATH = Path("models/risk_forecaster_discrete_hazard.pkl")


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_competition_planner() -> Planner:
    """Load the configured planner or the required discrete-hazard forecaster."""

    planner_path_value = os.environ.get("BATTERYSWAP_PLANNER_PATH")
    if planner_path_value:
        planner_path = Path(planner_path_value)
        if not planner_path.exists():
            raise FileNotFoundError(f"Planner artifact does not exist: {planner_path}")
        planner = _load_pickle(planner_path)
        if not isinstance(planner, Planner):
            raise TypeError("BATTERYSWAP_PLANNER_PATH did not contain a Planner")
        return planner

    configured_forecaster = os.environ.get("BATTERYSWAP_FORECASTER_PATH")
    forecaster_path = (
        Path(configured_forecaster)
        if configured_forecaster
        else DEFAULT_HAZARD_FORECASTER_PATH
    )
    if not forecaster_path.exists():
        raise FileNotFoundError(f"Hazard forecaster artifact does not exist: {forecaster_path}")
    forecaster = _load_pickle(forecaster_path)

    risk_multiplier = float(os.environ.get("BATTERYSWAP_LATE_RISK_MULTIPLIER", "1.0"))
    solver_seconds = float(os.environ.get("BATTERYSWAP_SOLVER_SECONDS", "2.0"))
    local_search = int(os.environ.get("BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS", "160"))
    uncertain_search = int(
        os.environ.get("BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS", "70")
    )
    robust_samples = int(os.environ.get("BATTERYSWAP_ROBUST_SAMPLES", "4"))
    config = PlannerConfig(
        late_risk_multiplier=risk_multiplier,
        local_search_evaluations=local_search,
        uncertain_local_search_evaluations=uncertain_search,
        robust_emergency_samples=robust_samples,
        optimizer=OptimizationConfig(solver_seconds=solver_seconds),
    )
    return CompetitionPlanner(forecaster=forecaster, config=config)


def main() -> None:
    planner = load_competition_planner()
    dataset_path = Path(os.environ.get("BATTERYSWAP_DATASET_PATH", "/tmp/data"))
    splits = [
        split.strip()
        for split in os.environ.get("BATTERYSWAP_SPLITS", "public,private").split(",")
        if split.strip()
    ]
    make_submissions(lambda: planner, dataset_path=dataset_path, splits=splits)
    if not Path("submission.csv").exists():
        raise RuntimeError("Submission generation did not create submission.csv")


if __name__ == "__main__":
    main()
