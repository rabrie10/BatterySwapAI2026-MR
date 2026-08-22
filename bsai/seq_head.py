"""Sequence head: a small learned encoder over the last 120 days.

The incumbent's features are hand-crafted summaries of the smoothed grid plus
within-day shape statistics. The hypothesis this module tests is that the
*trajectory itself* -- the last 120 days of smoothed margin, temperature
contrast, within-day dV/dT, raw daily medians and their presence masks --
carries temporal patterns those summaries miss, specifically in the knee-entry
population (margin ~0.12 failing ~25 days later) that dominates the mid-year
ranking gap (top-12 realized 0.214).

The statistical framing is identical to ``bsai/wiener.py``: the model learns
the distribution of the censor-aware margin DROP over a window (sample-rich,
hundreds of thousands of windows) rather than classifying the 82 crossing
events. It predicts nine quantiles of the drop, conditioned on the horizon,
and the crossing probability is read off the quantile curve:

    P(cross within h | margin m) = P(drop_h >= m)

evaluated by monotone interpolation across the predicted quantiles with
exponential tails. Horizon enters the head as a conditioning scalar (one model
serves the whole grid, exactly as the incumbent's horizon column does), and
``predict_grid`` applies the same effective-horizon clipping, calibration hook
and monotone accumulation as ``WienerModel``.

Architecture: a dilated 1-D CNN (receptive field 129 days > 120), GroupNorm
(never BatchNorm -- building shift would leak through batch statistics),
~28k parameters. The trunk runs once per cutoff; the light quantile head runs
once per (cutoff, horizon), which is what makes full-fidelity training on
~550k windows affordable on CPU.

Everything here is new code; no existing module is modified. The
``SeqModel`` wrapper presents the ``WienerModel`` interface (``predict_grid``,
``cdf_at``, ``calibration``, ``model_version='bsai-seq/v1'``) so the planner,
``tools/fit_calibration.py`` and ``tools/validate_v6.py`` run unchanged. For
validation it identifies each row through a feature fingerprint (voltage,
staleness, observations, age_days, remaining) against windows captured by the
actual incremental deployment caches -- see ``tools/seq_pack.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .features import FEATURE_NAMES, FeatureContext
from .hazard import HORIZON_GRID
from .margin import EOL_THRESHOLD

SEQ_WINDOW = 120
N_CHANNELS = 7
# Channel layout of the *unscaled* per-day matrix (see build_channels):
#   0 margin (smoothed voltage - 2.4, forward-filled)
#   1 smoothed temperature (forward-filled; window mean subtracted at input)
#   2 within-day dV/dT (bsai/shape beta, nan -> 0)
#   3 beta-present mask
#   4 raw daily margin (bsai/rawdaily median - 2.4, nan -> smoothed margin)
#   5 raw-present mask
#   6 smoothed-day-present mask (staleness is its trailing run of zeros)

# The nine quantiles {0.05, 0.15, ..., 0.95} of the drop distribution.
QUANTILES = tuple(round(0.05 + 0.1 * i, 2) for i in range(10))

# Fit horizons: identical to bsai.wiener.FIT_HORIZONS. Inference clamps the
# conditioning scalar to this range, which mirrors the incumbent GBDT whose
# piecewise-constant horizon response is also flat outside the fitted range.
SEQ_FIT_HORIZONS = (7, 14, 21, 28, 42, 63, 91, 126)

TARGET_SCALE = 10.0  # drops trained in decivolts so the head starts at O(1)
MARGIN_SCALE = 4.0
TEMP_SCALE = 0.25
BETA_CLIP = (-0.1, 0.2)
BETA_SCALE = 20.0
MIN_MARGIN = 1e-4  # volts; below this the crossing is certain (as in wiener)

# Feature columns used for the validation-time row fingerprint. All are exact
# under the deployment clamp (they are functions of the valid smoothed days at
# or before the evaluated index) and are captured from the same incremental
# caches the forecaster runs. Six columns: a collision then requires two
# devices with byte-identical histories, in which case their windows are
# identical too and sharing one entry is harmless (verified at pack time).
KEY_FEATURES = (
    "voltage",
    "staleness",
    "observations",
    "age_days",
    "voltage_max",
    "voltage_min",
)
KEY_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in KEY_FEATURES)


# ---------------------------------------------------------------------------
# channel assembly
# ---------------------------------------------------------------------------

def _forward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill nan, then back-fill the leading gap with the first value."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    mask = np.isfinite(out)
    if not mask.any():
        return np.zeros_like(out)
    idx = np.where(mask, np.arange(out.size), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(mask))
    idx[idx < 0] = first
    return out[idx]


def build_channels(
    smooth_voltage: np.ndarray,
    smooth_temperature: np.ndarray,
    beta_daily: np.ndarray | None,
    raw_daily_voltage: np.ndarray | None,
) -> np.ndarray:
    """Per-day unscaled channel matrix, shape (len(grid), N_CHANNELS) float32.

    All four inputs live on the same daily grid (the smoothing cache's grid for
    one device); ``beta_daily`` and ``raw_daily_voltage`` may be None or carry
    nan where the day is missing. Values are causal by construction: day ``d``
    of any input only uses measurements up to day ``d``.
    """
    length = int(np.asarray(smooth_voltage).shape[0])
    margin = np.asarray(smooth_voltage, dtype=np.float64) - EOL_THRESHOLD
    present = np.isfinite(margin)
    margin_f = _forward_fill(margin)

    temp = np.asarray(smooth_temperature, dtype=np.float64)
    if np.isfinite(temp).any():
        temp_f = _forward_fill(temp)
    else:
        temp_f = np.full(length, 20.0)

    if beta_daily is None:
        beta = np.zeros(length)
        beta_mask = np.zeros(length)
    else:
        beta = np.asarray(beta_daily, dtype=np.float64)
        beta_mask = np.isfinite(beta).astype(np.float64)
        beta = np.where(np.isfinite(beta), beta, 0.0)

    if raw_daily_voltage is None:
        raw = margin_f.copy()
        raw_mask = np.zeros(length)
    else:
        raw_v = np.asarray(raw_daily_voltage, dtype=np.float64)
        raw_mask = np.isfinite(raw_v).astype(np.float64)
        raw = np.where(np.isfinite(raw_v), raw_v - EOL_THRESHOLD, margin_f)

    channels = np.stack(
        [margin_f, temp_f, beta, beta_mask, raw, raw_mask, present.astype(np.float64)],
        axis=1,
    )
    return channels.astype(np.float32)


def pad_channels(channels: np.ndarray) -> np.ndarray:
    """Left-pad with SEQ_WINDOW-1 synthetic days so any window slice is valid.

    Pad rows replicate the first day's margin/temperature (masks zero), which
    reads as "device existed but reported nothing yet" -- the same statement
    the masks make about genuine gaps.
    """
    pad = np.repeat(channels[:1], SEQ_WINDOW - 1, axis=0)
    pad[:, 2] = 0.0  # beta value
    pad[:, 3] = 0.0  # beta mask
    pad[:, 4] = pad[:, 0]  # raw margin falls back to smoothed margin
    pad[:, 5] = 0.0  # raw mask
    pad[:, 6] = 0.0  # present mask
    return np.concatenate([pad, channels], axis=0)


def window_at(padded: np.ndarray, index: int) -> np.ndarray:
    """The 120-day window ending at grid index ``index`` (inclusive)."""
    return padded[index : index + SEQ_WINDOW]


def window_from_channels(channels: np.ndarray, index: int) -> np.ndarray:
    """The window ending at ``index`` from an *unpadded* channel matrix."""
    lo = index - SEQ_WINDOW + 1
    if lo >= 0:
        return channels[lo : index + 1].copy()
    head = np.repeat(channels[:1], -lo, axis=0)
    head[:, 2] = 0.0
    head[:, 3] = 0.0
    head[:, 4] = head[:, 0]
    head[:, 5] = 0.0
    head[:, 6] = 0.0
    return np.concatenate([head, channels[: index + 1]], axis=0)


def gather_windows(
    data: np.ndarray, offsets: np.ndarray, cutoffs: np.ndarray
) -> np.ndarray:
    """Batched window extraction from a concatenated padded channel bank.

    ``data`` is the concatenation of every device's padded channel matrix;
    ``offsets[i]`` is where device ``i``'s padded matrix begins. The window for
    (device ``d``, grid cutoff ``c``) is ``data[offsets[d]+c : +SEQ_WINDOW]``.
    """
    starts = offsets + cutoffs
    index = starts[:, None] + np.arange(SEQ_WINDOW)[None, :]
    return data[index]


def input_from_windows(windows: np.ndarray) -> np.ndarray:
    """Scale raw channel windows (B, 120, 7) into network input (B, 7, 120).

    Fixed physical scalings only -- nothing fitted, nothing per-building, so a
    fresh building cannot shift the normalisation (the BatchNorm failure mode).
    The temperature channel becomes the deviation from its own 120-day mean.
    """
    w = np.asarray(windows, dtype=np.float32)
    out = np.empty((w.shape[0], N_CHANNELS, SEQ_WINDOW), dtype=np.float32)
    out[:, 0] = w[:, :, 0] * MARGIN_SCALE
    temp = w[:, :, 1]
    out[:, 1] = (temp - temp.mean(axis=1, keepdims=True)) * TEMP_SCALE
    out[:, 2] = np.clip(w[:, :, 2], *BETA_CLIP) * BETA_SCALE
    out[:, 3] = w[:, :, 3]
    out[:, 4] = w[:, :, 4] * MARGIN_SCALE
    out[:, 5] = w[:, :, 5]
    out[:, 6] = w[:, :, 6]
    return out


def horizon_scalars(margin: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    """Conditioning scalars for the quantile head: margin and encoded horizon."""
    h = np.clip(
        np.asarray(horizon, dtype=np.float32),
        SEQ_FIT_HORIZONS[0],
        SEQ_FIT_HORIZONS[-1],
    )
    m = np.asarray(margin, dtype=np.float32) * MARGIN_SCALE
    return np.stack([m, h / 42.0, np.log(h / 42.0)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

class SeqQuantileNet(nn.Module):
    """Dilated 1-D CNN trunk + horizon-conditioned monotone quantile head.

    ``encode`` runs once per cutoff; ``head`` runs once per (cutoff, horizon).
    Quantiles are non-crossing by construction: the first output is the base
    and the rest are cumulative softplus increments.
    """

    N_SCALARS = 3

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        width: int = 32,
        embed: int = 64,
        n_quantiles: int = len(QUANTILES),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        specs = ((5, 1), (3, 2), (3, 4), (3, 8), (3, 16), (3, 32))
        layers: list[nn.Module] = []
        previous = in_channels
        for kernel, dilation in specs:
            layers.append(
                nn.Conv1d(
                    previous,
                    width,
                    kernel,
                    dilation=dilation,
                    padding=(kernel - 1) // 2 * dilation,
                )
            )
            layers.append(nn.GroupNorm(4, width))
            layers.append(nn.SiLU())
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3 * width + self.N_SCALARS, embed)
        self.fc2 = nn.Linear(embed, embed)
        self.out = nn.Linear(embed, n_quantiles)
        with torch.no_grad():
            self.out.bias.fill_(-1.0)
            self.out.bias[0] = -2.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1), h[..., -1]], dim=1)

    def head(self, z: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.dropout(z), scalars], dim=1)
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        raw = self.out(h)
        base = raw[:, :1]
        steps = F.softplus(raw[:, 1:])
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x), scalars)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pinball_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over rows and the nine quantiles."""
    taus = torch.as_tensor(QUANTILES, dtype=predicted.dtype, device=predicted.device)
    diff = target[:, None] - predicted
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()


# ---------------------------------------------------------------------------
# quantiles -> crossing probability
# ---------------------------------------------------------------------------

_TAUS = np.asarray(QUANTILES, dtype=np.float64)
_LOG_LO = float(np.log(_TAUS[1] / _TAUS[0]))          # lower-tail decay rate
_LOG_HI = float(np.log((1 - _TAUS[-2]) / (1 - _TAUS[-1])))  # upper-tail rate


def crossing_probability(quantiles: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """P(drop >= margin) from the predicted quantile curve.

    Piecewise-linear CDF between the nine quantiles; exponential tails outside
    them, with decay rates matched to the local quantile spacing (so the tail
    continues the curve rather than truncating it -- plateau rows far above
    every quantile keep a small, margin-ordered probability instead of a hard
    zero, and near-certain rows approach one instead of stopping at 0.95).

    ``quantiles`` and ``margin`` are in the same (scaled) units.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    m = np.asarray(margin, dtype=np.float64)
    n = q.shape[0]

    # Position of m in each row's quantile vector: j = #(quantiles <= m).
    j = (m[:, None] >= q).sum(axis=1)

    p = np.empty(n, dtype=np.float64)

    lo = j == 0
    if lo.any():
        lam = np.maximum(q[lo, 1] - q[lo, 0], 1e-6) / _LOG_LO
        cdf = _TAUS[0] * np.exp((m[lo] - q[lo, 0]) / lam)
        p[lo] = 1.0 - cdf

    hi = j == q.shape[1]
    if hi.any():
        lam = np.maximum(q[hi, -1] - q[hi, -2], 1e-6) / _LOG_HI
        p[hi] = (1.0 - _TAUS[-1]) * np.exp(-(m[hi] - q[hi, -1]) / lam)

    mid = ~(lo | hi)
    if mid.any():
        jm = j[mid]
        rows = np.flatnonzero(mid)
        q_lo = q[rows, jm - 1]
        q_hi = q[rows, np.minimum(jm, q.shape[1] - 1)]
        t_lo = _TAUS[jm - 1]
        t_hi = np.where(jm < q.shape[1], _TAUS[np.minimum(jm, q.shape[1] - 1)], 1.0)
        span = q_hi - q_lo
        frac = np.where(span > 1e-9, (m[mid] - q_lo) / np.maximum(span, 1e-9), 1.0)
        cdf = t_lo + (t_hi - t_lo) * np.clip(frac, 0.0, 1.0)
        p[mid] = 1.0 - cdf

    return np.clip(p, 0.0, 1.0)


@torch.no_grad()
def probability_grid(
    net: SeqQuantileNet,
    windows: np.ndarray,
    margins: np.ndarray,
    remaining: np.ndarray,
    horizons: tuple[int, ...] = HORIZON_GRID,
    batch: int = 4096,
) -> np.ndarray:
    """P(cross within h) for every horizon in the grid, shape (rows, len(grid)).

    Mirrors ``WienerModel.predict_grid``'s effective-horizon rule: the window
    that matters is ``min(h, remaining)``; zero when no observation time is
    left. The trunk runs once per row; the head once per (row, horizon).
    """
    net.eval()
    rows = windows.shape[0]
    grid = np.asarray(horizons, dtype=np.float64)
    out = np.zeros((rows, len(horizons)), dtype=np.float64)
    if rows == 0:
        return out

    margins = np.asarray(margins, dtype=np.float64)
    remaining = np.asarray(remaining, dtype=np.float64)

    for start in range(0, rows, batch):
        stop = min(start + batch, rows)
        x = torch.from_numpy(input_from_windows(windows[start:stop]))
        z = net.encode(x)
        b = stop - start
        eff = np.clip(np.minimum(grid[None, :], remaining[start:stop, None]), 0.0, None)
        scalars = horizon_scalars(
            np.repeat(margins[start:stop], len(horizons)),
            eff.reshape(-1),
        )
        z_tiled = z.repeat_interleave(len(horizons), dim=0)
        quantiles = net.head(z_tiled, torch.from_numpy(scalars)).numpy()
        # quantiles are in target (decivolt) units; margins scaled to match
        m_scaled = np.repeat(margins[start:stop], len(horizons)) * TARGET_SCALE
        p = crossing_probability(quantiles, m_scaled)
        p = np.where(eff.reshape(-1) <= 0.0, 0.0, p)
        p = np.where(
            np.repeat(margins[start:stop], len(horizons)) <= MIN_MARGIN, 1.0, p
        )
        out[start:stop] = p.reshape(b, len(horizons))
    return out


@torch.no_grad()
def probability_at(
    net: SeqQuantileNet,
    windows: np.ndarray,
    margins: np.ndarray,
    horizon: np.ndarray,
    batch: int = 8192,
) -> np.ndarray:
    """P(cross within ``horizon``) for one per-row horizon (the 42d decision)."""
    net.eval()
    rows = windows.shape[0]
    out = np.zeros(rows, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    horizon = np.asarray(horizon, dtype=np.float64)
    for start in range(0, rows, batch):
        stop = min(start + batch, rows)
        x = torch.from_numpy(input_from_windows(windows[start:stop]))
        z = net.encode(x)
        scalars = horizon_scalars(margins[start:stop], horizon[start:stop])
        quantiles = net.head(z, torch.from_numpy(scalars)).numpy()
        p = crossing_probability(quantiles, margins[start:stop] * TARGET_SCALE)
        p = np.where(horizon[start:stop] <= 0.0, 0.0, p)
        p = np.where(margins[start:stop] <= MIN_MARGIN, 1.0, p)
        out[start:stop] = p
    return out


# ---------------------------------------------------------------------------
# WienerModel-compatible wrapper for the validation/planner path
# ---------------------------------------------------------------------------

def make_keys(features: np.ndarray, remaining: np.ndarray) -> list[bytes]:
    """Row fingerprints: float32 bytes of the key features + rounded remaining.

    The forecaster hands ``predict_grid`` float32 feature rows; the pack tool
    captures the identical rows through the same incremental caches, so byte
    equality identifies the (device, scenario) pair without device ids --
    which ``OofHazardModel`` does not forward to fold models.
    """
    cols = np.ascontiguousarray(
        np.asarray(features, dtype=np.float32)[:, list(KEY_COLUMNS)]
    )
    rem = np.rint(np.asarray(remaining, dtype=np.float64)).astype(np.int32)
    return [cols[i].tobytes() + rem[i].tobytes() for i in range(cols.shape[0])]


@dataclass
class SeqModel:
    """Presents the WienerModel interface over the sequence net.

    Holds the deployment-captured windows and the fingerprint index so that
    ``predict_grid(features, remaining)`` -- the exact call signature
    ``OofHazardModel`` makes -- can recover each row's trajectory. Unmatched
    rows raise rather than silently scoring zero, the same refusal policy the
    out-of-fold dispatcher applies to unknown buildings.
    """

    net: SeqQuantileNet
    windows: np.ndarray  # (n_rows, SEQ_WINDOW, N_CHANNELS) float16
    margins: np.ndarray  # (n_rows,) float32: forward-filled margin at cutoff
    key_index: dict[bytes, int]
    climatology: np.ndarray
    calibration: object | None = None
    volatility_scale: float = 1.0  # accepted for tool compatibility; unused
    horizons: tuple[int, ...] = HORIZON_GRID
    feature_names: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_NAMES))
    model_version: str = "bsai-seq/v1"

    def context(self) -> FeatureContext:
        return FeatureContext(climatology=self.climatology)

    def predict_grid(
        self,
        features: np.ndarray,
        remaining: np.ndarray,
        devices: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = features.shape[0]
        if rows == 0:
            return np.zeros((0, len(self.horizons)))
        torch.set_num_threads(int(min(3, torch.get_num_threads() or 3)))
        keys = make_keys(features, remaining)
        ids = np.empty(rows, dtype=np.int64)
        missing = 0
        for i, key in enumerate(keys):
            row = self.key_index.get(key, -1)
            ids[i] = row
            missing += row < 0
        if missing:
            raise KeyError(
                f"seq model could not identify {missing}/{rows} rows by feature "
                "fingerprint; the pack snapshot does not cover this population"
            )
        windows = self.windows[ids].astype(np.float32)
        margins = self.margins[ids].astype(np.float64)
        out = probability_grid(
            self.net, windows, margins, np.asarray(remaining, dtype=np.float64),
            self.horizons,
        )
        if self.calibration is not None:
            out = self.calibration.apply(out, remaining)
        return np.maximum.accumulate(np.clip(out, 0.0, 1.0), axis=1)

    def cdf_at(self, grid_values: np.ndarray, days: np.ndarray) -> np.ndarray:
        grid = np.asarray(self.horizons, dtype=float)
        days = np.asarray(days, dtype=float)
        if grid_values.shape[0] == 0:
            return np.zeros((0, days.shape[0]))
        anchored_x = np.concatenate([[0.0], grid])
        anchored_y = np.hstack([np.zeros((grid_values.shape[0], 1)), grid_values])
        out = np.empty((grid_values.shape[0], days.shape[0]))
        for row in range(grid_values.shape[0]):
            out[row] = np.interp(days, anchored_x, anchored_y[row])
        return out
