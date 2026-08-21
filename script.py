"""Official BatterySwapAI submission entry point.

Task 1 combines the V7 Wiener first-passage model with V9's independent
scenario-incidence estimate. Task 2 ranks a broad V7 candidate pool, then caps
the number serviced with V9's fleet-level estimate. The two meet at the v1
forecast contract.

The planner instance is deliberately shared across every scenario:
``make_submissions`` calls the loader once per scenario, and the smoothing cache
inside the forecaster is what keeps the run inside its 30-minute budget.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import joblib

from batteryswap_public.interfaces import Planner
from batteryswap_public.utils import make_submissions

from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from bsai.forecaster import HazardForecaster
from bsai.portfolio import PortfolioForecaster
from bsai.runtime import (
    HARD_DEADLINE_SECONDS,
    SOFT_DEADLINE_SECONDS,
    BudgetedPlanner,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/v7_wiener.joblib")
DEFAULT_INCIDENCE_MODEL_PATH = Path("models/v9_incidence.joblib")


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def load_forecaster():
    """Load V7 plus V9's independent fleet-incidence portfolio model."""
    path = Path(os.environ.get("BATTERYSWAP_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        LOGGER.error("Model artifact missing at %s; falling back to voltage trend", path)
        return None
    try:
        base = HazardForecaster(joblib.load(path))
        incidence_path = Path(
            os.environ.get(
                "BATTERYSWAP_INCIDENCE_MODEL_PATH", DEFAULT_INCIDENCE_MODEL_PATH
            )
        )
        if not incidence_path.exists():
            LOGGER.warning(
                "Incidence model missing at %s; using ordinary V7 decisions",
                incidence_path,
            )
            return base
        return PortfolioForecaster(base, joblib.load(incidence_path))
    except Exception:
        LOGGER.exception("Could not load %s; falling back to voltage trend", path)
        return None


def build_planner_config(solver_seconds: float, local: int, uncertain: int) -> PlannerConfig:
    return PlannerConfig(
        late_risk_multiplier=_float_env("BATTERYSWAP_LATE_RISK_MULTIPLIER", 1.0),
        minimum_expected_improvement=_float_env(
            "BATTERYSWAP_MINIMUM_EXPECTED_IMPROVEMENT", 0.0
        ),
        local_search_evaluations=local,
        uncertain_local_search_evaluations=uncertain,
        robust_emergency_samples=_int_env("BATTERYSWAP_ROBUST_SAMPLES", 4),
        optimizer=OptimizationConfig(solver_seconds=solver_seconds),
    )


def load_competition_planner() -> Planner:
    override = os.environ.get("BATTERYSWAP_PLANNER_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"Planner artifact does not exist: {path}")
        with path.open("rb") as handle:
            planner = pickle.load(handle)
        if not isinstance(planner, Planner):
            raise TypeError("BATTERYSWAP_PLANNER_PATH did not contain a Planner")
        return planner

    config = build_planner_config(
        _float_env("BATTERYSWAP_SOLVER_SECONDS", 1.0),
        _int_env("BATTERYSWAP_LOCAL_SEARCH_EVALUATIONS", 80),
        _int_env("BATTERYSWAP_UNCERTAIN_LOCAL_SEARCH_EVALUATIONS", 35),
    )
    inner = CompetitionPlanner(forecaster=load_forecaster(), config=config)
    fast = build_planner_config(0.25, 12, 6)
    return BudgetedPlanner(
        inner,
        fast_config=fast,
        soft_deadline=_float_env("BATTERYSWAP_SOFT_DEADLINE", SOFT_DEADLINE_SECONDS),
        hard_deadline=_float_env("BATTERYSWAP_HARD_DEADLINE", HARD_DEADLINE_SECONDS),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
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
    if isinstance(planner, BudgetedPlanner):
        LOGGER.info(
            "planned=%d degraded=%d deferred=%d elapsed=%.0fs",
            planner.scenarios_planned,
            planner.scenarios_degraded,
            planner.scenarios_deferred,
            planner.elapsed,
        )


if __name__ == "__main__":
    main()
