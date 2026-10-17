"""Explicit shipped settings shared by submission, training and validation."""

from pathlib import Path

MODEL_PATH = Path("models/v7_wiener.joblib")
VOLATILITY_SCALE = 1.0
PROBABILITY_SCALE = 1.5
SOLVER_SECONDS = 1.0
LOCAL_SEARCH_EVALUATIONS = 240
UNCERTAIN_SEARCH_EVALUATIONS = 240
ROBUST_EMERGENCY_SAMPLES = 0
CANDIDATE_MARGIN_HOURS = 12.0
