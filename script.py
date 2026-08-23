"""Official BatterySwapAI submission entry point.

Task 1 is the V8 phase-1 Wiener first-passage model in ``bsai``; Task 2 is the
existing ``batteryswap_solution`` planner, which reaches 77.83 on train
scenarios 0-11 when the risk it is given is correct. The two meet at the v1
forecast contract, so the model can be replaced without touching any scheduling
code.

**The default artifact is the public-leaderboard incumbent and nothing else.**
V8 phase 1 (``models/v7_wiener.joblib``, commit ``db85121``) scored **2078.28**
on public. Two later generations scored worse from a better local number, in
opposite directions: V9 (``models/v9_blend.joblib``, ``c36d4a3``, local 1753.46)
went to **2137.22** by planning one more swap per scenario for zero extra
catches, and V19 (``157513e``, local 1715.9) went to **2113.43** by cutting
volume and missing real failures. Local out-of-fold rank does not decide what
ships here; a confirmed public score does. ``_describe_model`` below logs the
identity of whatever is actually loaded, so a silent swap cannot happen twice.

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
from bsai.runtime import (
    HARD_DEADLINE_SECONDS,
    SOFT_DEADLINE_SECONDS,
    BudgetedPlanner,
)

LOGGER = logging.getLogger(__name__)

# The public incumbent. See the module docstring: this is the only artifact with
# a confirmed leaderboard score better than every alternative measured so far.
DEFAULT_MODEL_PATH = Path("models/v7_wiener.joblib")
INCUMBENT_MODEL_VERSION = "bsai-wiener/v1"

# The sequence reranker. V8's own CDF multiset is handed out in the sequence
# model's order, so per-scenario risk mass is unchanged and only *which* battery
# carries which curve changes. Set BATTERYSWAP_SEQUENCE_PATH="" to ship plain V8.
DEFAULT_SEQUENCE_PATH = Path("models/sequence_tcn.json")


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _is_lfs_pointer(path: Path) -> bool:
    """Is this the 132-byte stand-in rather than the artifact itself?

    ``.gitattributes`` tracks ``*.joblib`` through Git LFS, so the blob in the
    repository *is* a pointer and the real bytes only appear if the checkout ran
    the LFS smudge filter with the objects available. A clone without
    ``git lfs`` produces a text file where the model should be.
    """
    try:
        head = path.read_bytes()[:64]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs")


def _describe_model(model: object, path: Path) -> str:
    """One line naming exactly which Task 1 model is about to plan.

    A joblib path is not an identity: ``models/v7_wiener.joblib`` is rewritten
    in place by ``tools/fit_calibration.py``, so the calibration factors are
    part of what has to be visible. This string is logged at INFO on every run
    and asserted against in ``tests/test_submission_identity.py``.
    """
    calibration = getattr(model, "calibration", None)
    factors = getattr(calibration, "factors", ())
    return (
        f"task1 model={type(model).__module__}.{type(model).__name__} "
        f"version={getattr(model, 'model_version', '?')} "
        f"path={path} "
        f"volatility_scale={getattr(model, 'volatility_scale', '?')} "
        f"level_scale={getattr(model, 'level_scale', 1.0)} "
        f"calibration={[round(float(f), 4) for f in factors] or None}"
    )


def load_forecaster() -> HazardForecaster | None:
    """Load the shipped model, or None so the planner uses its own fallback.

    ``models/v7_wiener.joblib`` unpickles as ``bsai.wiener.WienerModel`` holding
    a ``bsai.calibrate.RemainingCalibration``; ``models/v9_blend.joblib`` (kept
    for comparison, not shipped) additionally needs ``bsai.blend.BlendedModel``
    and scikit-learn heads. ``COPY bsai/ ./bsai`` covers all of them -- see
    HANDOVER.md trap 9, where a missing module silently downgraded the
    submission to the voltage-trend forecaster.

    The identity of what was loaded is logged rather than assumed. Set
    ``BATTERYSWAP_ALLOW_NON_INCUMBENT=1`` to run a non-incumbent artifact
    deliberately; without it a mismatch is logged as a warning so it shows up in
    the submission transcript.
    """
    path = Path(os.environ.get("BATTERYSWAP_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        LOGGER.error("Model artifact missing at %s; falling back to voltage trend", path)
        return None
    if _is_lfs_pointer(path):
        # A pointer unpickles as garbage, joblib.load raises, and the fallback
        # below silently downgrades the whole submission to the voltage-trend
        # forecaster -- which is a valid plan and a catastrophic score. Say so
        # in one line that names the fix, so it is visible in the transcript
        # rather than deduced from the result.
        LOGGER.error(
            "%s is a %d-byte Git-LFS pointer, not a model. Run `git lfs install "
            "&& git lfs pull` in this checkout. Falling back to voltage trend, "
            "which scores several thousand points worse.",
            path,
            path.stat().st_size,
        )
        return None
    try:
        model = joblib.load(path)
    except Exception:
        LOGGER.exception("Could not load %s; falling back to voltage trend", path)
        return None
    LOGGER.info("%s", _describe_model(model, path))
    version = getattr(model, "model_version", None)
    if version != INCUMBENT_MODEL_VERSION and not os.environ.get(
        "BATTERYSWAP_ALLOW_NON_INCUMBENT"
    ):
        LOGGER.warning(
            "loaded %s, not the public incumbent %s -- V9 (bsai-blend/v2) scored "
            "2137.22 on public against V8's 2078.28",
            version,
            INCUMBENT_MODEL_VERSION,
        )
    return _maybe_sequence(model)


def _maybe_sequence(model) -> HazardForecaster:
    """Wrap V8 in the order-only sequence remap, or say why it did not.

    Every failure here is logged at ERROR and falls back to plain V8 rather than
    to something worse -- but silence is the actual danger, because a fallback
    that scores 2126 instead of 2055 looks like a normal run. The artifact is
    plain JSON precisely so there is no Git-LFS pointer to resolve; the check
    below exists because `models/v7_wiener.joblib` taught this project that an
    unresolved pointer degrades a submission without a word.
    """
    raw = os.environ.get("BATTERYSWAP_SEQUENCE_PATH", str(DEFAULT_SEQUENCE_PATH))
    if not raw:
        LOGGER.info("sequence remap disabled by BATTERYSWAP_SEQUENCE_PATH=''")
        return HazardForecaster(model)
    path = Path(raw)
    if not path.exists():
        LOGGER.error(
            "sequence artifact missing at %s; shipping plain V8 ordering, which "
            "measured 2126.53 against the remap's 1967.10 out of fold", path)
        return HazardForecaster(model)
    try:
        from bsai.sequence import build_forecaster

        weight = _float_env("BATTERYSWAP_SEQUENCE_WEIGHT", 1.0)
        forecaster = build_forecaster(model, path, weight=weight)
    except Exception:
        LOGGER.exception("Could not build the sequence remap from %s; plain V8", path)
        return HazardForecaster(model)
    LOGGER.info(
        "task1 remap=bsai-sequence/v1 path=%s folds=%d parameters_per_fold=%d "
        "weight=%.2f order_only=True",
        path, len(forecaster.sequence_scorer.model.folds),
        sum(t.size for t in forecaster.sequence_scorer.model.folds[0].tensors.values()),
        weight,
    )
    return forecaster


def build_planner_config(solver_seconds: float, local: int, uncertain: int) -> PlannerConfig:
    return PlannerConfig(
        late_risk_multiplier=_float_env("BATTERYSWAP_LATE_RISK_MULTIPLIER", 1.0),
        minimum_expected_improvement=_float_env(
            "BATTERYSWAP_MINIMUM_EXPECTED_IMPROVEMENT", 0.0
        ),
        local_search_evaluations=local,
        uncertain_local_search_evaluations=uncertain,
        # Four stratified emergency samples, not the deterministic expectation.
        # On V8's own ordering the deterministic path (0) is worth -35.30 and is
        # what this file shipped before. It is *not* used with the sequence
        # remap: measured through this entry point the two combine to 16.48
        # s/scenario, projecting 28.6 minutes for the official 96 against a
        # 30-minute cap and past bsai/runtime.py's 27.5-minute hard deadline.
        # The remap plus this planner measured 2126.53 -> 1967.10 out of fold by
        # building (t = -3.21, 5 of 6 blocks), and 759 s for the 48 train
        # scenarios end to end through this file -- 25.3 minutes projected for 96
        # against a 30-minute cap, with 0 scenarios degraded and 1.76 GB peak.
        # Runtime headroom decides this one; see docs/SUBMISSION_TCN.md 1 and 8.
        robust_emergency_samples=_int_env("BATTERYSWAP_ROBUST_SAMPLES", 4),
        candidate_margin_hours=_float_env("BATTERYSWAP_CANDIDATE_MARGIN", 12.0),
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

    # The search budget stays exactly where V8 shipped it. Raising it to
    # 120/120 is worth a further -17.5 and measured 829 s for 48 scenarios
    # through this same entry point, which projects to ~27.7 minutes for the
    # official 96 against a 30-minute cap -- not enough headroom to be worth
    # 17 points. Only the objective changes, above.
    config = build_planner_config(
        _float_env("BATTERYSWAP_SOLVER_SECONDS", 0.5),
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
