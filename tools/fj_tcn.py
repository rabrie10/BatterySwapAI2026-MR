"""A small causal TCN on raw trajectories: can the sequence say what the state cannot?

Six engineered representations have now improved ordering at matched margin and
damaged it across margins (`docs/FINAL_SEGMENT_EXPERIMENT.md`). The remaining
hypothesis is that the *hand-engineered state* -- 64 features, all read at the
cutoff -- has discarded the temporal information needed to know when a
wider-margin battery is in more danger than a tighter-margin one.

So this model is not given the 64 features. It is given the raw daily trajectory
and asked to predict the *future trajectory*, which is the same
data-rich supervision that made the Wiener law work: hundreds of thousands of
causal windows rather than the ~82 failure events.

    past 120 days of (voltage, shape, temperature, mask, staleness)
        -> small causal TCN
        -> 7 quantiles of the voltage change at 7, 14, 21, 28 and 42 days

Two gates, in order, both on held-out buildings:

1. **forecast** -- does it beat V8's own drift/scatter regressors and a
   persistence baseline at predicting future voltage, by horizon and by margin
   band? If not, stop; there is nothing to convert.
2. **ordering** -- turned into a 42-day crossing probability, does it beat V8's
   cross-margin concordance of 0.7359 by a margin that is not noise?

Leakage discipline: five building-disjoint folds; normalisation statistics,
model weights and every preprocessing constant are fitted inside the fold. A
window belongs to the fold of its device's building, and no held-out device
contributes a gradient or a statistic. Targets are censor-masked per horizon --
a horizon whose future day is unobserved is dropped from the loss for that
window rather than filled.

    python tools/fj_tcn.py --train
    python tools/fj_tcn.py --gate1
    python tools/fj_tcn.py --gate2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HISTORY = 120
HORIZONS = (7, 14, 21, 28, 42)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CHANNELS = 5
STRIDE = 3
V_SCALE = 0.1          # targets and the shape channel are expressed in units of this
EOL_THRESHOLD = 2.4
CHANNEL_NAMES = ("margin", "shape", "temperature", "mask", "staleness")


# ------------------------------------------------------------------ the data

class Corpus:
    """Every device's daily grid, concatenated, plus the legal anchor days.

    One flat array per channel with per-device offsets lets a batch be gathered
    with a single fancy index instead of 512 slices, and costs 385k * 5 floats.
    A window is legal only if it lies wholly inside one device, so the anchor
    must sit at least HISTORY - 1 days into that device's own history.

    ``stop`` is the day each device's battery reaches end of life, and an anchor
    must fall **strictly before** it. This matters more than its 2.6 % share of
    the windows suggests. A device's series continues for a median 86 days past
    its crossing, and 75.5 % of the windows anchored in that tail sit *below* the
    2.4 V barrier -- down to 2.037 V -- while **no** pre-EOL window and none of
    the 19,890 competition rows is below it at all (the minimum margin a scenario
    ever presents is exactly 0.0000). Training origins after EOL would therefore
    spend a small model's capacity on a voltage regime inference never visits,
    immediately adjacent to the decision boundary.

    The *targets* are deliberately not clipped: a window anchored before EOL
    whose 42-day horizon runs through and past the crossing is exactly the
    terminal decline the model is meant to learn.
    """

    def __init__(self, series: dict, stride: int = STRIDE,
                 stop: dict[str, int] | None = None) -> None:
        devices, raw, filled, temperature, mask, stale, offset = [], [], [], [], [], [], []
        cursor = 0
        for device in sorted(series):
            voltage, temp, _origin = series[device]
            voltage = np.asarray(voltage, dtype=np.float64)
            temp = np.asarray(temp, dtype=np.float64)
            good = np.isfinite(voltage)
            if good.sum() < HISTORY + max(HORIZONS):
                continue
            # Causal forward fill: index of the most recent observed day at or
            # before each day. Nothing after the day is read.
            last = np.maximum.accumulate(np.where(good, np.arange(good.size), -1))
            usable = last >= 0
            fill = np.full(good.size, np.nan)
            fill[usable] = voltage[last[usable]]
            age = np.where(usable, np.arange(good.size) - last, 0.0)
            temp_last = np.maximum.accumulate(
                np.where(np.isfinite(temp), np.arange(temp.size), -1))
            temp_fill = np.full(temp.size, np.nan)
            ok = temp_last >= 0
            temp_fill[ok] = temp[temp_last[ok]]

            devices.append(device)
            raw.append(voltage)
            filled.append(fill)
            temperature.append(temp_fill)
            mask.append(good.astype(np.float64))
            stale.append(age)
            offset.append(cursor)
            cursor += voltage.size

        self.devices = np.asarray(devices)
        self.offset = np.asarray(offset)
        self.length = np.asarray([r.size for r in raw])
        self.raw = np.concatenate(raw)
        self.filled = np.concatenate(filled)
        self.temperature = np.concatenate(temperature)
        self.mask = np.concatenate(mask)
        self.stale = np.concatenate(stale)

        anchors, owner = [], []
        self.stop = np.asarray([
            int((stop or {}).get(str(device), 1 << 30)) for device in self.devices])
        for index in range(self.devices.size):
            start, size = self.offset[index], self.length[index]
            days = np.arange(HISTORY - 1, size - 1, stride)
            days = days[days < self.stop[index]]
            local = start + days
            legal = np.isfinite(self.filled[local]) & np.isfinite(self.temperature[local])
            anchors.append(local[legal])
            owner.append(np.full(int(legal.sum()), index))
        self.anchor = np.concatenate(anchors)
        self.owner = np.concatenate(owner)

        # Censor-aware targets: NaN wherever the future day is unobserved or past
        # the end of that device's own series.
        limit = (self.offset + self.length)[self.owner]
        self.target = np.full((self.anchor.size, len(HORIZONS)), np.nan)
        for column, horizon in enumerate(HORIZONS):
            future = self.anchor + horizon
            inside = future < limit
            value = np.full(self.anchor.size, np.nan)
            value[inside] = self.raw[future[inside]]
            self.target[:, column] = value - self.filled[self.anchor]

    def batch(self, rows: np.ndarray, centre: float, spread: float) -> np.ndarray:
        """(B, C, L) float32, standardised with training-fold statistics only."""
        index = self.anchor[rows][:, None] - HISTORY + 1 + np.arange(HISTORY)[None, :]
        voltage = self.filled[index]
        anchor_value = self.filled[self.anchor[rows]][:, None]
        out = np.empty((rows.size, CHANNELS, HISTORY), dtype=np.float32)
        out[:, 0] = (voltage - EOL_THRESHOLD) / 0.5
        out[:, 1] = (voltage - anchor_value) / V_SCALE
        out[:, 2] = (self.temperature[index] - centre) / max(spread, 1e-6)
        out[:, 3] = self.mask[index]
        out[:, 4] = np.minimum(self.stale[index], 30.0) / 30.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------- the model

def make_model(torch):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            self.pad = (3 - 1) * dilation
            self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
            self.norm = nn.GroupNorm(4, width)
            self.act = nn.GELU()

        def forward(self, x):
            y = self.conv(nn.functional.pad(x, (self.pad, 0)))
            return x + self.act(self.norm(y))

    class TCN(nn.Module):
        """~30k parameters: six dilated causal blocks, receptive field 127 days."""

        def __init__(self, width: int = 24, latent: int = 32) -> None:
            super().__init__()
            self.stem = nn.Conv1d(CHANNELS, width, 1)
            self.blocks = nn.ModuleList(
                [Block(width, 2 ** k) for k in range(6)])
            self.head = nn.Sequential(
                nn.Linear(width, latent), nn.GELU(),
                nn.Linear(latent, len(HORIZONS) * len(QUANTILES)))

        def forward(self, x):
            h = self.stem(x)
            for block in self.blocks:
                h = block(h)
            out = self.head(h[:, :, -1])
            return out.view(-1, len(HORIZONS), len(QUANTILES))

    return TCN


def pinball(torch, prediction, target, available):
    """Quantile loss, masked per (window, horizon) so censoring is not imputed."""
    q = torch.tensor(QUANTILES, dtype=prediction.dtype).view(1, 1, -1)
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(q * error, (q - 1.0) * error)
    weight = available.unsqueeze(-1)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


# -------------------------------------------------------------------- --train

def anchor_days(corpus: "Corpus") -> np.ndarray:
    """Each window's anchor as a day index inside its own device's grid."""
    return corpus.anchor - corpus.offset[corpus.owner]


def assert_origins_precede_eol(corpus: "Corpus", stop: dict[str, int]) -> None:
    """No training example may start from a state a live battery cannot be in.

    A competition scenario only ever asks about an active battery, and the 19,890
    cached rows bear that out exactly: none sits at or after its device's
    crossing, and the smallest margin any of them presents is 0.0000. A window
    anchored after EOL is off-distribution by construction, so this is asserted
    rather than trusted.
    """
    day = anchor_days(corpus)
    limit = np.asarray([stop.get(str(d), 1 << 30) for d in corpus.devices])[corpus.owner]
    bad = int((day >= limit).sum())
    if bad:
        raise AssertionError(
            f"{bad} training windows are anchored at or after their device's EOL")


def _targets_through_eol(corpus: "Corpus", stop: dict[str, int]) -> np.ndarray:
    """Windows whose longest horizon reaches into or past the crossing."""
    day = anchor_days(corpus)
    limit = np.asarray([stop.get(str(d), 1 << 30) for d in corpus.devices])[corpus.owner]
    return (day + max(HORIZONS) >= limit) & np.isfinite(corpus.target[:, -1])


def fold_of_device(folds_path: Path) -> dict[str, int]:
    import joblib

    from batteryswap_public.utils import load_devices

    bundle = joblib.load(folds_path)
    seen: dict[int, int] = {}
    by_building = {
        str(name): seen.setdefault(id(model), len(seen))
        for name, model in bundle["by_building"].items()
    }
    devices = load_devices(Path("dataset/train") / "devices.csv")
    return {
        str(d): by_building[str(b)]
        for d, b in zip(devices["device_id"], devices["building_id"])
        if str(b) in by_building
    }


def train(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(11)
    started = time.time()

    from tools.fj_templates import crossing_index
    from tools.fj_terminality import load_series

    series = load_series(args.series)
    stop = crossing_index(series, args.dataset)
    corpus = Corpus(series, stride=args.stride, stop=stop)
    assert_origins_precede_eol(corpus, stop)
    assign = fold_of_device(args.folds)
    fold = np.asarray([assign.get(d, -1) for d in corpus.devices])
    window_fold = fold[corpus.owner]
    available = np.isfinite(corpus.target)
    print(f"{corpus.devices.size} devices, {corpus.anchor.size} windows "
          f"(stride {args.stride}, history {HISTORY} d); "
          f"{len(stop)} EOL devices, every origin strictly before its crossing")
    past = _targets_through_eol(corpus, stop)
    print(f"  {int(past.sum())} windows have a 42-day target that runs through or "
          f"past EOL -- kept on purpose, that is the terminal decline")
    print("target coverage per horizon: " + "  ".join(
        f"{h}d {available[:, i].mean():.1%}" for i, h in enumerate(HORIZONS)))
    for group in sorted(set(window_fold.tolist())):
        if group < 0:
            continue
        print(f"  fold {group}: {int((window_fold == group).sum()):6d} windows, "
              f"{int((fold == group).sum()):3d} devices")

    TCN = make_model(torch)
    payload = {"history": HISTORY, "horizons": list(HORIZONS),
               "quantiles": list(QUANTILES), "stride": args.stride}
    states, stats = {}, {}
    for group in sorted(g for g in set(window_fold.tolist()) if g >= 0):
        rows = np.flatnonzero((window_fold != group) & (window_fold >= 0))
        # Normalisation from training-fold windows only.
        sample = corpus.temperature[corpus.anchor[rows]]
        centre = float(np.nanmean(sample))
        spread = float(np.nanstd(sample))
        stats[group] = (centre, spread)

        model = TCN(width=args.width)
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs * max(rows.size // args.batch, 1))
        rng = np.random.default_rng(100 + group)
        model.train()
        for epoch in range(args.epochs):
            order = rng.permutation(rows)
            total, seen = 0.0, 0
            for start in range(0, order.size - args.batch + 1, args.batch):
                picked = order[start:start + args.batch]
                x = torch.from_numpy(corpus.batch(picked, centre, spread))
                y = torch.from_numpy(
                    np.nan_to_num(corpus.target[picked] / V_SCALE).astype(np.float32))
                m = torch.from_numpy(available[picked].astype(np.float32))
                loss = pinball(torch, model(x), y, m)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                schedule.step()
                total += float(loss.detach()) * picked.size
                seen += picked.size
            print(f"  fold {group} epoch {epoch + 1}/{args.epochs} "
                  f"pinball {total / max(seen, 1):.5f}  ({time.time() - started:.0f}s)")
        states[group] = {k: v.clone() for k, v in model.state_dict().items()}

    payload["states"] = states
    payload["stats"] = stats
    payload["width"] = args.width
    torch.save(payload, args.model)
    parameters = sum(p.numel() for p in TCN(width=args.width).parameters())
    print(f"\n{parameters} parameters per fold; wrote {args.model} "
          f"({time.time() - started:.0f}s)")


def load_bundle(args):
    import torch

    payload = torch.load(args.model, weights_only=False)
    TCN = make_model(torch)
    models = {}
    for group, state in payload["states"].items():
        model = TCN(width=payload["width"])
        model.load_state_dict(state)
        model.eval()
        models[group] = model
    return payload, models


def predict(torch, model, corpus, rows, centre, spread, batch=1024) -> np.ndarray:
    out = np.empty((rows.size, len(HORIZONS), len(QUANTILES)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            picked = rows[start:start + batch]
            x = torch.from_numpy(corpus.batch(picked, centre, spread))
            out[start:start + picked.size] = model(x).numpy()
    return out * V_SCALE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_segment.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/fj_tcn.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_tcn.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gate1", action="store_true")
    parser.add_argument("--gate2", action="store_true")
    args = parser.parse_args()
    if args.train:
        train(args)
        return
    if args.gate1:
        from tools.fj_tcn_gates import gate1
        gate1(args)
        return
    if args.gate2:
        from tools.fj_tcn_gates import gate2
        gate2(args)
        return
    parser.error("choose a stage: --train, --gate1, --gate2")


if __name__ == "__main__":
    main()
