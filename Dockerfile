# Submission will be executed with this image.
# Adapted from the official BatterySwapAI2026-Example Dockerfile for this
# repo's layout: Task 2 lives in batteryswap_solution/, Task 1 in src/risk/,
# and the fitted forecaster artifact in models/.
FROM huggingface/competitions:latest

# Default to running on train split only for local testing; the real
# evaluation run overrides this (BATTERYSWAP_SPLITS=public,private).
ENV BATTERYSWAP_SPLITS=train

WORKDIR /app

# NOTE: allowed requirements are specified by the competition.
# WARNING: customizations to requirements.txt are ignored by the official
# runtime; this local install only approximates it for Docker testing.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything script.py needs at runtime.
# src/ must be present so unpickling models/risk_forecaster.pkl succeeds:
# pickle resolves the artifact's class as src.risk.model.Task1Forecaster,
# which requires src/ to be importable, not just batteryswap_solution/.
COPY batteryswap_solution/ ./batteryswap_solution
COPY src/ ./src
COPY models/ ./models
COPY script.py ./

# Default to making submissions.
CMD ["python3", "script.py"]
