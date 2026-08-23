"""The two gates the sequence model has to pass, in order.

**Gate 1 -- forecast.** Both models are asked the same question at the same rows:
given everything observable at a scenario cutoff, what is the voltage change over
the next 7 / 14 / 21 / 28 / 42 days? V8 answers with the drift and scatter
regressors that the first-passage law is built on (`bsai/wiener.py`); the TCN
answers with quantiles of the same quantity. Persistence -- "nothing changes" --
is carried as the floor. Scored by device-weighted MAE, median absolute error and
pinball loss, split by margin band, on held-out buildings only.

**Gate 2 -- ordering.** The forecast is turned into a causal 42-day crossing
score and judged on within-scenario cross-margin concordance against V8's 0.7359.
Absolute levels are not trusted: the score is used as a rank. Two readings are
reported, a quantile-interpolated probability and the Wiener-style standardised
crossing distance, plus the first-passage reading that takes the worst horizon
rather than only day 42.

Neither gate lets a held-out building influence anything: fold models,
normalisation constants and V8's own fold routing are all keyed on the building.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tools.fj_segment import Lab, context, device_bootstrap, device_weight, logit
from tools.fj_tcn import (
    EOL_THRESHOLD,
    HISTORY,
    HORIZONS,
    QUANTILES,
    Corpus,
    load_bundle,
    predict,
)

BANDS = (0.03, 0.05, 0.10, 0.20)
BAND_NAMES = ("0-0.03", "0.03-0.05", "0.05-0.10", "0.10-0.20", ">0.20")
Z90 = 1.2815515655446004


def _positions(corpus: Corpus, frame, dataset: Path) -> np.ndarray:
    """Absolute index into the concatenated corpus for every frame row."""
    from tools.fj_segment import cutoff_days
    from tools.fj_terminality import load_series

    cutoff = cutoff_days(frame, load_series(Path("outputs/fj_series.npz")), dataset)
    where = {device: index for index, device in enumerate(corpus.devices)}
    out = np.full(frame.scenario.size, -1, dtype=np.int64)
    for row in range(frame.scenario.size):
        index = where.get(str(frame.battery[row]))
        if index is None or cutoff[row] < HISTORY - 1:
            continue
        if cutoff[row] >= corpus.length[index]:
            continue
        out[row] = corpus.offset[index] + cutoff[row]
    return out


def _tcn_at(args, corpus, frame, position, fold) -> np.ndarray:
    """(rows, horizons, quantiles) predicted change, each row from its own fold."""
    import torch

    payload, models = load_bundle(args)
    torch.set_num_threads(args.threads)
    out = np.full((frame.scenario.size, len(HORIZONS), len(QUANTILES)), np.nan,
                  dtype=np.float32)
    for group, model in models.items():
        rows = np.flatnonzero((fold == group) & (position >= 0))
        if rows.size == 0:
            continue
        centre, spread = payload["stats"][group]
        # Query positions are scenario cutoffs, not the strided training anchors,
        # so a temporary corpus view is built whose anchors are exactly those.
        view = _view(corpus, position[rows])
        out[rows] = predict(torch, model, view, np.arange(rows.size), centre, spread)
    return out


class _view:
    """A Corpus whose anchor list is replaced, sharing the same flat channels."""

    def __init__(self, corpus: Corpus, anchors: np.ndarray) -> None:
        self.filled = corpus.filled
        self.temperature = corpus.temperature
        self.mask = corpus.mask
        self.stale = corpus.stale
        self.raw = corpus.raw
        self.anchor = np.asarray(anchors, dtype=np.int64)

    batch = Corpus.batch


def _v8_forecast(ctx, args) -> tuple[np.ndarray, np.ndarray]:
    """V8's own answer to the same question: expected drop and its sigma."""
    import joblib

    from bsai.wiener import MIN_SIGMA

    bundle = joblib.load(args.folds)
    frame = ctx["frame"]
    drop = np.full((frame.scenario.size, len(HORIZONS)), np.nan)
    sigma = np.full((frame.scenario.size, len(HORIZONS)), np.nan)
    for building in np.unique(frame.building):
        model = bundle["by_building"][building]
        rows = np.flatnonzero(frame.building == building)
        features = frame.features[rows]
        for column, horizon in enumerate(HORIZONS):
            design = np.hstack([
                features,
                np.full((rows.size, 1), float(horizon), dtype=np.float32)])
            drop[rows, column] = np.maximum(model.drift.predict(design), 0.0)
            sigma[rows, column] = (
                np.maximum(model.scatter.predict(design), MIN_SIGMA)
                * np.sqrt(np.pi / 2.0) * model.volatility_scale)
    return drop, sigma


def _actual(corpus, position) -> np.ndarray:
    """Realised voltage change at each horizon, NaN where unobserved."""
    out = np.full((position.size, len(HORIZONS)), np.nan)
    good = position >= 0
    base = np.full(position.size, np.nan)
    base[good] = corpus.filled[position[good]]
    limit = np.searchsorted(corpus.offset, position, side="right")
    end = (corpus.offset + corpus.length)[np.clip(limit - 1, 0, corpus.length.size - 1)]
    for column, horizon in enumerate(HORIZONS):
        future = position + horizon
        inside = good & (future < end)
        value = np.full(position.size, np.nan)
        value[inside] = corpus.raw[future[inside]]
        out[:, column] = value - base
    return out


def _pinball(prediction: np.ndarray, target: np.ndarray) -> float:
    q = np.asarray(QUANTILES)[None, :]
    error = target[:, None] - prediction
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def gate1(args) -> None:
    started = time.time()
    ctx = context(args)
    frame, fold, margin = ctx["frame"], ctx["fold"], ctx["margin"]

    from tools.fj_templates import crossing_index
    from tools.fj_terminality import load_series

    series = load_series(args.series)
    corpus = Corpus(series, stride=args.stride,
                    stop=crossing_index(series, args.dataset))
    position = _positions(corpus, frame, args.dataset)
    actual = _actual(corpus, position)
    tcn = _tcn_at(args, corpus, frame, position, fold)
    drop, sigma = _v8_forecast(ctx, args)

    median = QUANTILES.index(0.5)
    low, high = QUANTILES.index(0.1), QUANTILES.index(0.9)
    usable = (position >= 0) & np.isfinite(tcn[:, 0, 0])
    print(f"{int(usable.sum())} of {frame.scenario.size} frame rows scored "
          f"({np.unique(frame.battery[usable]).size} devices); "
          f"the rest have under {HISTORY} days of history at the cutoff")

    out: dict = {"horizon": {}, "band": {}}
    print(f"\n{'horizon':>8} {'n':>7} {'persist MAE':>12} {'V8 MAE':>9} {'TCN MAE':>9}"
          f" {'V8 medAE':>9} {'TCN medAE':>10} {'V8 pin':>8} {'TCN pin':>8}")
    for column, horizon in enumerate(HORIZONS):
        rows = np.flatnonzero(usable & np.isfinite(actual[:, column]))
        weight = device_weight(frame.battery, rows)
        weight = weight / weight.sum()
        truth = actual[rows, column]
        v8_mid, tcn_mid = -drop[rows, column], tcn[rows, column, median]
        wmae = lambda e: float(weight @ np.abs(e))  # noqa: E731
        wmed = lambda e: float(np.median(np.abs(e)))  # noqa: E731
        v8_q = np.stack([
            -drop[rows, column] + sigma[rows, column] * _z(q) for q in QUANTILES], 1)
        entry = {
            "n": int(rows.size),
            "devices": int(np.unique(frame.battery[rows]).size),
            "persistence_mae": round(wmae(truth), 5),
            "v8_mae": round(wmae(truth - v8_mid), 5),
            "tcn_mae": round(wmae(truth - tcn_mid), 5),
            "v8_medae": round(wmed(truth - v8_mid), 5),
            "tcn_medae": round(wmed(truth - tcn_mid), 5),
            "v8_pinball": round(_pinball(v8_q, truth), 5),
            "tcn_pinball": round(_pinball(tcn[rows, column], truth), 5),
            "tcn_q10_q90_coverage": round(float(
                np.mean((truth >= tcn[rows, column, low])
                        & (truth <= tcn[rows, column, high]))), 4),
        }
        out["horizon"][f"{horizon}d"] = entry
        print(f"{horizon:>7}d {rows.size:>7} {entry['persistence_mae']:>12.5f} "
              f"{entry['v8_mae']:>9.5f} {entry['tcn_mae']:>9.5f} "
              f"{entry['v8_medae']:>9.5f} {entry['tcn_medae']:>10.5f} "
              f"{entry['v8_pinball']:>8.5f} {entry['tcn_pinball']:>8.5f}")

    column = HORIZONS.index(42)
    band = np.searchsorted(np.asarray(BANDS), margin)
    print(f"\n42-day forecast by margin band (device-weighted MAE, volts)")
    print(f"{'band':>12} {'n':>7} {'devices':>8} {'persist':>9} {'V8':>9} {'TCN':>9} "
          f"{'TCN-V8':>9}")
    for level, name in enumerate(BAND_NAMES):
        rows = np.flatnonzero(usable & np.isfinite(actual[:, column]) & (band == level))
        if rows.size < 30:
            continue
        weight = device_weight(frame.battery, rows)
        weight = weight / weight.sum()
        truth = actual[rows, column]
        v8_e = float(weight @ np.abs(truth + drop[rows, column]))
        tcn_e = float(weight @ np.abs(truth - tcn[rows, column, median]))
        per = float(weight @ np.abs(truth))
        out["band"][name] = {
            "n": int(rows.size), "devices": int(np.unique(frame.battery[rows]).size),
            "persistence_mae": round(per, 5), "v8_mae": round(v8_e, 5),
            "tcn_mae": round(tcn_e, 5), "delta": round(tcn_e - v8_e, 5),
        }
        flag = "  <-- TCN better" if tcn_e < v8_e else ""
        print(f"{name:>12} {rows.size:>7} {np.unique(frame.battery[rows]).size:>8} "
              f"{per:>9.5f} {v8_e:>9.5f} {tcn_e:>9.5f} {tcn_e - v8_e:>+9.5f}{flag}")

    np.savez_compressed(args.report.with_suffix(".npz"),
                        tcn=tcn.astype(np.float32), drop=drop.astype(np.float32),
                        sigma=sigma.astype(np.float32), position=position,
                        actual=actual.astype(np.float32))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


def _z(q: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(q))


def _probability_from_quantiles(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """P(change <= threshold) from predicted quantiles, with exponential tails.

    Inside the knots this is a linear interpolation of the CDF. Outside them a
    plain clamp would send most of the population to exactly 0 and destroy the
    ordering, so the outermost gap sets an exponential decay rate instead.
    """
    levels = np.asarray(QUANTILES)
    knots = np.sort(values, axis=1)
    out = np.empty(values.shape[0])
    for row in range(values.shape[0]):
        x, y = knots[row], levels
        point = threshold[row]
        if point <= x[0]:
            scale = max(x[1] - x[0], 1e-4)
            out[row] = y[0] * np.exp((point - x[0]) / scale)
        elif point >= x[-1]:
            scale = max(x[-1] - x[-2], 1e-4)
            out[row] = 1.0 - (1.0 - y[-1]) * np.exp((x[-1] - point) / scale)
        else:
            out[row] = float(np.interp(point, x, y))
    return np.clip(out, 1e-9, 1 - 1e-9)


def gate2(args) -> None:
    started = time.time()
    ctx = context(args)
    frame, fold, margin = ctx["frame"], ctx["fold"], ctx["margin"]
    lab, base = ctx["lab"], ctx["base"]
    cache = np.load(args.report.with_suffix(".npz"))
    tcn, position = cache["tcn"].astype(float), cache["position"]
    median, low, high = QUANTILES.index(0.5), QUANTILES.index(0.1), QUANTILES.index(0.9)

    scored = (position >= 0) & np.isfinite(tcn[:, 0, 0])
    print(f"scored {int(scored.sum())} rows; "
          f"{int((lab.mask & ~scored).sum())} landmark rows fall back to V8")

    from scipy.stats import norm

    def assemble(probability: np.ndarray) -> np.ndarray:
        """Fall back to V8 wherever the TCN has no window.

        Every variant is converted to a *probability* before this point, so the
        fallback rows and the scored rows share one scale. Mixing a z-score with
        `logit(p_V8)` inside a scenario would corrupt the ordering rather than
        fall back to it.
        """
        out = logit(base).copy()
        out[scored] = logit(np.clip(probability, 1e-9, 1 - 1e-9))[scored]
        return out

    variants: dict[str, np.ndarray] = {}
    column = HORIZONS.index(42)
    # (a) quantile-interpolated crossing probability at 42 days
    probability = np.zeros(frame.scenario.size)
    rows = np.flatnonzero(scored)
    probability[rows] = _probability_from_quantiles(tcn[rows, column], -margin[rows])
    variants["quantile p42"] = assemble(probability)
    # (b) Wiener-style standardised crossing distance at 42 days, read as a
    #     Gaussian tail so it lands on the same scale as everything else
    scale = np.maximum((tcn[:, column, high] - tcn[:, column, low]) / (2 * Z90), 1e-4)
    variants["z-score 42"] = assemble(
        norm.cdf((-margin - tcn[:, column, median]) / scale))
    # (c) first passage: the worst horizon, not only day 42
    worst = np.full(frame.scenario.size, -np.inf)
    for col in range(len(HORIZONS)):
        s = np.maximum((tcn[:, col, high] - tcn[:, col, low]) / (2 * Z90), 1e-4)
        worst = np.maximum(worst, (-margin - tcn[:, col, median]) / s)
    variants["z-score first passage"] = assemble(norm.cdf(worst))

    reference = lab.report(base, fold)
    print(f"\nV8    all {reference['all']:.4f}  cross {reference['cross']:.4f}  "
          f"same {reference['same']:.4f}  folds "
          + " ".join(f"{reference[f'x{i}']:.3f}" for i in range(5)))
    results = {"V8": reference}
    for name, score in variants.items():
        table = lab.report(score, fold)
        won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
        results[name] = {**table, "folds_won_cross": won}
        flag = "  <-- beats V8 cross" if table["cross"] > reference["cross"] else ""
        print(f"  {name:<24} all {table['all']:.4f}  cross {table['cross']:.4f}  "
              f"same {table['same']:.4f}  folds won {won}/5{flag}")

    # A rank blend, reported because a standalone sequence score is not the only
    # way to spend one; still order-only, still judged on cross-margin.
    from tools.fj_fit import standardise_within

    best = max(variants, key=lambda k: results[k]["cross"])
    for weight in (0.25, 0.5, 1.0):
        blended = (standardise_within(frame.scenario, logit(base))
                   + weight * standardise_within(frame.scenario, variants[best]))
        table = lab.report(blended, fold)
        won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
        results[f"blend {best} w={weight}"] = {**table, "folds_won_cross": won}
        flag = "  <-- beats V8 cross" if table["cross"] > reference["cross"] else ""
        print(f"  {'blend w=' + str(weight):<24} all {table['all']:.4f}  "
              f"cross {table['cross']:.4f}  same {table['same']:.4f}  "
              f"folds won {won}/5{flag}")

    top = max((k for k in results if k != "V8"),
              key=lambda k: results[k]["cross"])
    score = variants[top] if top in variants else (
        standardise_within(frame.scenario, logit(base))
        + float(top.split("=")[-1]) * standardise_within(frame.scenario, variants[best]))
    boot = device_bootstrap(lab, score, base, "cross")
    results["bootstrap"] = {"candidate": top, **boot}
    print(f"\nbest: {top}  cross {results[top]['cross']:.4f} against V8's "
          f"{reference['cross']:.4f}")
    print(f"  device bootstrap of the cross-margin delta, 300 draws: "
          f"{boot['delta']:+.4f} [{boot['lo']:+.4f}, {boot['hi']:+.4f}], "
          f"P(delta > 0) = {boot['p_positive']:.2f}")
    print(f"  gate needs >= 0.75 cross-margin: "
          f"{'PASS' if results[top]['cross'] >= 0.75 else 'FAIL'}")

    path = args.report.with_name("fj_tcn_gate2.json")
    path.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {path} ({time.time() - started:.0f}s)")
