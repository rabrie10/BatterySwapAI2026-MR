# Submission will be executed with this image.
# Adapted from the official BatterySwapAI2026-Example Dockerfile for this
# repo's layout: Task 1 lives in bsai/, Task 2 in batteryswap_solution/, and
# the fitted forecaster artifact in models/.
FROM huggingface/competitions:latest

WORKDIR /app

# NOTE: allowed requirements are specified by the competition.
# WARNING: customizations to requirements.txt are ignored by the official
# runtime; this local install only approximates it for Docker testing.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything script.py needs at runtime.
# bsai/ must be present before models/v9_blend.joblib can be loaded: joblib
# resolves the artifact's class as bsai.blend.BlendedModel, which holds a
# bsai.wiener.WienerModel and a bsai.calibrate.RemainingCalibration, and those in
# turn import bsai.features, bsai.hazard, bsai.margin and bsai.smoothing -- so the
# whole package has to come along. src/ is kept for the frozen v4 control artifact, which
# unpickles as src.risk.model.Task1Forecaster.
COPY batteryswap_solution/ ./batteryswap_solution
COPY bsai/ ./bsai
COPY src/ ./src
COPY models/ ./models
COPY script.py ./

# No BATTERYSWAP_SPLITS default is baked in here. script.py's own fallback is
# public,private, matching the official run -- and matching the official
# checklist's own docker run command, which passes no override at all. Baking
# train in as an image-level default meant that exact command would have
# silently produced a train-only submission. Pass -e BATTERYSWAP_SPLITS=train
# explicitly when testing locally against the train-only dataset checkout.

# Default to making submissions.
CMD ["python3", "script.py"]
