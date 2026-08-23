"""The sequence model at inference: NumPy only, no Torch, no LFS, no network.

`docs/FINAL_TCN_REPRESENTATION.md` measured a causal TCN trained in PyTorch. This
module is what actually runs in the submission container, and it is deliberately
not the training code:

* **NumPy forward pass.** The network is 12,899 parameters and six dilated causal
  convolutions; reimplementing it removes a Torch import (and any version drift
  in it) from the critical path. `tests/test_sequence.py` asserts the NumPy and
  Torch outputs agree to 1e-5 on real windows.
* **Plain-JSON weights.** `.gitattributes` tracks `*.pt`, `*.npz` and `*.npy`
  through Git LFS, so a checkout without the smudge filter would hand the model
  a 132-byte pointer -- the exact failure `script.py::_is_lfs_pointer` exists to
  catch for the Wiener artifact. A `.json` file is tracked normally, so there is
  no pointer to resolve and no fallback to trigger.
* **No lookup table.** The gate measurements used a precomputed per-row score
  keyed on `(device, remaining)`. That works for the 48 cached training
  scenarios and would silently miss **every** row of an unseen test building,
  leaving the submission as plain V8. Here the window is rebuilt from the
  forecaster's own smoothing cache and scored live.
* **All five folds, averaged.** Fold routing is meaningless on a building no
  fold ever saw. Every fold model is out of fold for a test building, so the
  ensemble averages their quantiles. `tools/fj_tcn.py --gate2 --ensemble`
  measures exactly this configuration leave-one-fold-out.

The deployment is order-only: this module produces a *rank*, and
`bsai.rerank.RankRemapModel` hands V8's own per-scenario CDF multiset out in that
order, so the risk mass a scenario carries is unchanged by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rerank import centred_rank, decision_level
from .hazard import HORIZON_GRID

HISTORY = 120
HORIZONS = (7, 14, 21, 28, 42)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CHANNELS = 5
V_SCALE = 0.1
EOL_THRESHOLD = 2.4
DILATIONS = (1, 2, 4, 8, 16, 32)
GROUPS = 4
EPS = 1e-5
STALE_CAP = 30.0


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching `torch.nn.GELU()`'s default (not the tanh variant)."""
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               groups: int = GROUPS) -> np.ndarray:
    """(B, C, L) normalised over (channels-in-group, length), as PyTorch does."""
    batch, channels, length = x.shape
    grouped = x.reshape(batch, groups, channels // groups, length)
    mean = grouped.mean(axis=(2, 3), keepdims=True)
    variance = grouped.var(axis=(2, 3), keepdims=True)
    out = ((grouped - mean) / np.sqrt(variance + EPS)).reshape(batch, channels, length)
    return out * weight[None, :, None] + bias[None, :, None]


def causal_conv(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                dilation: int) -> np.ndarray:
    """Left-padded dilated 1-D convolution: output t sees inputs <= t only."""
    out_channels, in_channels, kernel = weight.shape
    pad = (kernel - 1) * dilation
    padded = np.pad(x, ((0, 0), (0, 0), (pad, 0)))
    length = x.shape[2]
    # (B, Cin, K, L) gathered so the contraction is one einsum.
    taps = np.stack(
        [padded[:, :, k * dilation:k * dilation + length] for k in range(kernel)],
        axis=2,
    )
    return np.einsum("oik,bikl->bol", weight, taps) + bias[None, :, None]


@dataclass
class FoldWeights:
    tensors: dict
    temperature_centre: float
    temperature_spread: float


@dataclass
class SequenceModel:
    """The trained ensemble, forward pass only."""

    folds: list
    history: int = HISTORY
    horizons: tuple = HORIZONS
    quantiles: tuple = QUANTILES
    version: str = "bsai-sequence/v1"

    @classmethod
    def load(cls, path: Path) -> "SequenceModel":
        text = Path(path).read_text()
        if text.lstrip().startswith("version https://git-lfs"):
            raise OSError(
                f"{path} is a Git-LFS pointer, not the model. This file is meant "
                "to be tracked normally; run `git lfs pull` or restore it."
            )
        payload = json.loads(text)
        if payload.get("format") != cls.version:
            raise ValueError(f"{path} is {payload.get('format')!r}, not {cls.version!r}")
        folds = []
        for entry in payload["folds"]:
            tensors = {
                name: np.asarray(value["data"], dtype=np.float64).reshape(value["shape"])
                for name, value in entry["tensors"].items()
            }
            folds.append(FoldWeights(
                tensors=tensors,
                temperature_centre=float(entry["temperature_centre"]),
                temperature_spread=float(entry["temperature_spread"]),
            ))
        model = cls(folds=folds)
        counted = sum(t.size for t in folds[0].tensors.values())
        if counted != payload["parameters_per_fold"]:
            raise ValueError(
                f"{path}: {counted} parameters loaded, "
                f"{payload['parameters_per_fold']} declared"
            )
        return model

    def _forward_one(self, x: np.ndarray, fold: FoldWeights) -> np.ndarray:
        t = fold.tensors
        h = causal_conv(x, t["stem.weight"], t["stem.bias"], 1)
        for index, dilation in enumerate(DILATIONS):
            y = causal_conv(h, t[f"blocks.{index}.conv.weight"],
                            t[f"blocks.{index}.conv.bias"], dilation)
            y = group_norm(y, t[f"blocks.{index}.norm.weight"],
                           t[f"blocks.{index}.norm.bias"])
            h = h + gelu(y)
        last = h[:, :, -1]
        hidden = gelu(last @ t["head.0.weight"].T + t["head.0.bias"])
        out = hidden @ t["head.2.weight"].T + t["head.2.bias"]
        return out.reshape(-1, len(self.horizons), len(self.quantiles))

    def predict(self, windows: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Mean over folds of the predicted change quantiles, in volts.

        ``windows`` is (B, C, L) with the temperature channel *not yet*
        standardised; each fold applies its own training statistics, which is
        what makes averaging them legitimate rather than an average of models
        that disagree about their inputs.
        """
        total = np.zeros((windows.shape[0], len(self.horizons), len(self.quantiles)))
        for fold in self.folds:
            x = windows.copy()
            x[:, 2] = (temperature - fold.temperature_centre) / max(
                fold.temperature_spread, 1e-6)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            total += self._forward_one(x, fold)
        return (total / len(self.folds)) * V_SCALE


# ------------------------------------------------------------------ windowing

class _Filled:
    """Causal forward fill of one device's grid, cached and grown in place."""

    __slots__ = ("length", "filled", "mask", "stale", "temperature")

    def __init__(self, voltage: np.ndarray, temperature: np.ndarray) -> None:
        good = np.isfinite(voltage)
        last = np.maximum.accumulate(np.where(good, np.arange(good.size), -1))
        usable = last >= 0
        self.filled = np.full(good.size, np.nan)
        self.filled[usable] = np.asarray(voltage)[last[usable]]
        self.stale = np.where(usable, np.arange(good.size) - last, 0.0)
        self.mask = good.astype(float)
        warm = np.isfinite(temperature)
        last_t = np.maximum.accumulate(np.where(warm, np.arange(warm.size), -1))
        ok = last_t >= 0
        self.temperature = np.full(warm.size, np.nan)
        self.temperature[ok] = np.asarray(temperature)[last_t[ok]]
        self.length = good.size


def window_at(state: _Filled, index: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The (C, L) window ending at ``index``, and its raw temperature channel.

    Returns None when the device has fewer than ``HISTORY`` days of grid before
    the cutoff, or no observed voltage at it -- those rows keep V8's own order.
    """
    if index < HISTORY - 1 or index >= state.length:
        return None
    anchor = state.filled[index]
    if not np.isfinite(anchor):
        return None
    start = index - HISTORY + 1
    voltage = state.filled[start:index + 1]
    out = np.empty((CHANNELS, HISTORY))
    out[0] = (voltage - EOL_THRESHOLD) / 0.5
    out[1] = (voltage - anchor) / V_SCALE
    out[2] = 0.0
    out[3] = state.mask[start:index + 1]
    out[4] = np.minimum(state.stale[start:index + 1], STALE_CAP) / STALE_CAP
    return out, state.temperature[start:index + 1]


def crossing_probability(quantiles: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """P(change <= threshold) from predicted quantiles, with exponential tails.

    A plain clamp outside the outermost knots sends most of the population to
    exactly zero and destroys the ordering, so the outermost gap sets a decay
    rate instead. This is the same reading `tools/fj_tcn_gates.py` gated on.
    """
    levels = np.asarray(QUANTILES)
    knots = np.sort(quantiles, axis=1)
    out = np.empty(quantiles.shape[0])
    for row in range(quantiles.shape[0]):
        x, point = knots[row], threshold[row]
        if point <= x[0]:
            out[row] = levels[0] * np.exp((point - x[0]) / max(x[1] - x[0], 1e-4))
        elif point >= x[-1]:
            out[row] = 1.0 - (1.0 - levels[-1]) * np.exp(
                (x[-1] - point) / max(x[-1] - x[-2], 1e-4))
        else:
            out[row] = float(np.interp(point, x, levels))
    return np.clip(out, 1e-9, 1 - 1e-9)


# -------------------------------------------------------------------- scoring

@dataclass
class LiveSequenceScorer:
    """Rank by `centred_rank(p_V8) + weight * centred_rank(p_sequence)`.

    The weight is 1.0 -- the equal-weight rank average that gate 2 measured at
    cross-margin concordance 0.7802 against V8's 0.7359. It is not a fitted
    parameter; it exists so that 0.0 reproduces the incumbent exactly.

    A row the sequence model cannot score (under 120 days of grid, no observed
    voltage at the cutoff, or a device absent from the cache) keeps its
    incumbent rank, so a cold start degrades to V8 rather than to noise.
    """

    model: SequenceModel
    weight: float = 1.0
    horizons: tuple = HORIZON_GRID
    cache: object | None = None
    origin_ordinal: int | None = None
    scored: int = 0
    skipped: int = 0
    _states: dict = field(default_factory=dict)

    def bind(self, cache, origin_ordinal: int) -> None:
        self.cache = cache
        self.origin_ordinal = int(origin_ordinal)

    def _state(self, device: str):
        series = self.cache.devices.get(device) if self.cache is not None else None
        if series is None:
            return None, None
        state = self._states.get(device)
        if state is None or state.length != len(series):
            state = _Filled(series.smooth_voltage, series.smooth_temperature)
            self._states[device] = state
        return state, series

    def score(self, features, remaining, devices, grid) -> np.ndarray:
        own = centred_rank(decision_level(grid, remaining, self.horizons))
        if devices is None or self.cache is None or self.origin_ordinal is None:
            self.skipped += int(own.size)
            return own * (1.0 + self.weight)
        margin = features[:, 0].astype(float) - EOL_THRESHOLD
        windows, temperatures, rows = [], [], []
        for position, device in enumerate(devices):
            state, series = self._state(str(device))
            if state is None:
                continue
            built = window_at(state, series.index_of(self.origin_ordinal))
            if built is None:
                continue
            windows.append(built[0])
            temperatures.append(built[1])
            rows.append(position)
        if not rows:
            self.skipped += int(own.size)
            return own * (1.0 + self.weight)
        rows = np.asarray(rows)
        predicted = self.model.predict(np.stack(windows), np.stack(temperatures))
        column = list(self.horizons).index(42) if 42 in self.horizons else -1
        column = HORIZONS.index(42)
        probability = crossing_probability(predicted[:, column, :], -margin[rows])
        self.scored += int(rows.size)
        self.skipped += int(own.size - rows.size)
        other = own.copy()
        other[rows] = centred_rank(probability)
        return own + self.weight * other


def build_forecaster(model, artifact: Path, *, weight: float = 1.0):
    """V8 wrapped in the order-only remap, with the sequence scorer bound live.

    The scorer needs two things the model interface does not carry: the device's
    smoothed grid and the scenario's prediction origin. Both are known to the
    forecaster, so the binding happens there -- in a subclass, so V8's own path
    and `bsai/forecaster.py` are untouched.
    """
    from .forecaster import HazardForecaster, _normal_date, _ordinal
    from .rerank import RankRemapModel

    scorer = LiveSequenceScorer(model=SequenceModel.load(artifact), weight=weight)

    class SequenceForecaster(HazardForecaster):
        """`HazardForecaster` that tells the scorer where in time it is."""

        def predict(self, battery_data, locations, *, prediction_origin,
                    horizon_days, evaluation_observation_end):
            self.cache.update(battery_data)
            scorer.bind(self.cache, _ordinal(_normal_date(prediction_origin)))
            return super().predict(
                battery_data, locations,
                prediction_origin=prediction_origin,
                horizon_days=horizon_days,
                evaluation_observation_end=evaluation_observation_end,
            )

    forecaster = SequenceForecaster(RankRemapModel(base=model, scorer=scorer))
    forecaster.sequence_scorer = scorer
    return forecaster
