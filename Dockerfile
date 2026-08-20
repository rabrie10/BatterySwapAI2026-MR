# Submission will be executed with this image.
# Adapted from the official BatterySwapAI2026-Example Dockerfile for this
# repo's layout: Task 2 lives in batteryswap_solution/, Task 1 in src/risk/,
# and the fitted forecaster artifact in models/.
FROM huggingface/competitions:latest

# Default to running on train split only for local testing; the real
# evaluation run overrides this (BATTERYSWAP_SPLITS=public,private).
ENV BATTERYSWAP_SPLITS=train

WORKDIR /app

# Use the base image's bundled virtual environment and add only the CPU
# packages imported by this submission. The broad development requirements
# include Torch/CUDA even though neither Task 1 nor Task 2 imports them.
ENV PATH="/app/env/bin:${PATH}"
COPY requirements.submission.txt ./
RUN /app/env/bin/pip install --no-cache-dir -r requirements.submission.txt

# Copy everything script.py needs at runtime.
# src/ must be present so the discrete-hazard artifact can resolve its
# src.risk.* classes while unpickling.
COPY batteryswap_solution/ ./batteryswap_solution
COPY src/ ./src
COPY models/risk_forecaster_discrete_hazard.pkl ./models/risk_forecaster_discrete_hazard.pkl
COPY script.py ./

# Default to making submissions.
CMD ["python3", "script.py"]
