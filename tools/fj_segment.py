"""Analog cohorts: does the kind of device change what a voltage margin means?

Read `docs/FINAL_TERMINALITY.md`, `docs/FINAL_TEMPLATES.md` and
`docs/FINAL_FRAILTY.md` first. Four independent auxiliary representations have
now improved ordering at matched margin and damaged it across margins, because
each of them describes the *state*, and V8's margin already describes the state
better. The open question is not a better state description. It is whether the
mapping

    margin -> P(EOL within 42 days)

is the same function for every device, or whether there are regimes -- thermal,
baseline, dynamic -- in which the same 0.05 V means something different.

The experiment is deliberately low capacity and runs in three stages:

* `--build`   cache the device signature at all 19,890 (scenario, battery) rows;
* `--probe`   the scientific question on its own: fit P(due | margin) and
              P(due | margin, cohort) out of fold and see whether the cohort term
              survives;
* `--gate`    S1 (soft kNN analog cohort) and S2 (small in-fold segmentation),
              both deployed as order-only corrections to V8 and judged on
              **within-scenario cross-margin concordance** against V8's 0.7359.

Leakage discipline, re-created independently inside every one of the five
building folds and pinned by `tests/test_segment.py`:

* standardisation statistics, centroids and neighbour pools are fitted on
  training-fold devices only;
* a held-out building may *query* the training pool, never join it;
* neighbour outcomes come from training rows only;
* every signature reads the device's own past telemetry and nothing else.

    python tools/fj_segment.py --build
    python tools/fj_segment.py --probe
    python tools/fj_segment.py --gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fj_fit import label_and_mask  # noqa: E402
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402
from tools.fj_residual import v8_folds  # noqa: E402
from tools.fj_signature import NAMES, PLAIN, signature_at  # noqa: E402
from tools.fj_terminality import load_series  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
MARGIN_BIN = 0.01
TOP_K = 40


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------- the metric

class Lab:
    """The landmark population and the three concordances, fixed once.

    Landmarks are the top 40 rows per scenario by V8 probability among rows whose
    42-day fate is *observed* -- the same population `docs/FINAL_FRAILTY.md` and
    `docs/FINAL_TEMPLATES.md` report V8's 0.7280 / 0.7359 / 0.5846 on. A pair is
    cross-margin when the due row and the survivor sit in different 0.01 V bins,
    which is where the planner makes two of every three marginal choices.
    """

    def __init__(self, frame, base: np.ndarray, margin: np.ndarray) -> None:
        self.frame = frame
        self.base = base
        self.margin = margin
        self.bin = np.floor(margin / MARGIN_BIN).astype(int)
        _, self.usable = label_and_mask(frame)
        self.mask = np.zeros(base.shape[0], bool)
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero((frame.scenario == index) & self.usable)
            if rows.size:
                self.mask[rows[np.argsort(-base[rows], kind="stable")][:TOP_K]] = True
        positives, negatives = [], []
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero((frame.scenario == index) & self.mask)
            y = frame.due[rows]
            if y.sum() == 0 or (~y).sum() == 0:
                continue
            positive, negative = rows[y], rows[~y]
            positives.append(np.repeat(positive, negative.size))
            negatives.append(np.tile(negative, positive.size))
        self.positive = np.concatenate(positives)
        self.negative = np.concatenate(negatives)
        cross = self.bin[self.positive] != self.bin[self.negative]
        self.split = {
            "all": np.ones(cross.size, bool),
            "cross": cross,
            "same": ~cross,
        }

    def concordance(self, score: np.ndarray, keep: np.ndarray | None = None) -> float:
        positive = self.positive if keep is None else self.positive[keep]
        negative = self.negative if keep is None else self.negative[keep]
        if positive.size == 0:
            return float("nan")
        gap = score[positive] - score[negative]
        return float(((gap > 0).sum() + 0.5 * (gap == 0).sum()) / gap.size)

    def report(self, score: np.ndarray, fold: np.ndarray | None = None) -> dict:
        out = {kind: round(self.concordance(score, keep), 4)
               for kind, keep in self.split.items()}
        if fold is not None:
            # A fold's number is measured on pairs whose *both* rows are held-out
            # buildings, which is the definition `docs/FINAL_FRAILTY.md` reports
            # V8's 0.945 / 0.799 / 0.616 / 0.776 / 0.692 under. Scenarios mix
            # buildings, so keying on the due row alone would score a pair partly
            # on training buildings.
            for value in sorted(set(fold.tolist())):
                inside = (fold[self.positive] == value) & (fold[self.negative] == value)
                out[f"f{value}"] = round(self.concordance(score, inside), 4)
                out[f"x{value}"] = round(
                    self.concordance(score, inside & self.split["cross"]), 4)
        return out


def device_bootstrap(lab: Lab, score: np.ndarray, reference: np.ndarray,
                     kind: str = "cross", draws: int = 300, seed: int = 7) -> dict:
    """Resample *devices*, not rows: the 48 scenarios overlap by about 85 %."""
    keep = lab.split[kind]
    positive, negative = lab.positive[keep], lab.negative[keep]
    case = lab.frame.battery[positive]
    devices = np.unique(case)
    where = {d: np.flatnonzero(case == d) for d in devices}
    gap_new = score[positive] - score[negative]
    gap_old = reference[positive] - reference[negative]
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for draw in range(draws):
        picked = np.concatenate([where[d] for d in rng.choice(devices, devices.size)])
        new = ((gap_new[picked] > 0).sum() + 0.5 * (gap_new[picked] == 0).sum())
        old = ((gap_old[picked] > 0).sum() + 0.5 * (gap_old[picked] == 0).sum())
        deltas[draw] = (new - old) / picked.size
    return {
        "delta": round(float(deltas.mean()), 4),
        "lo": round(float(np.percentile(deltas, 2.5)), 4),
        "hi": round(float(np.percentile(deltas, 97.5)), 4),
        "p_positive": round(float((deltas > 0).mean()), 3),
    }


# ----------------------------------------------------------------- the build

def cutoff_days(frame, series: dict, dataset: Path) -> np.ndarray:
    """Index into each device's own smoothed grid at its scenario's start day."""
    scenarios = json.loads((dataset / "scenarios.json").read_text())
    starts = np.asarray([
        int((pd.Timestamp(s["start_time"]).normalize() - _EPOCH) / pd.Timedelta(days=1))
        for s in scenarios
    ])
    out = np.full(frame.scenario.size, -1, dtype=int)
    for row in range(frame.scenario.size):
        entry = series.get(str(frame.battery[row]))
        if entry is None:
            continue
        out[row] = min(max(starts[frame.scenario[row]] - entry[2], -1), entry[0].size - 1)
    return out


def build(args) -> None:
    started = time.time()
    frame = load_frame(args.frame)
    series = load_series(args.series)
    cutoff = cutoff_days(frame, series, args.dataset)
    print(f"{len(series)} smoothed device series, {frame.features.shape[0]} rows")

    cache: dict[tuple[str, int], np.ndarray] = {}
    out = np.full((frame.features.shape[0], len(NAMES)), np.nan)
    for row in range(frame.features.shape[0]):
        device = str(frame.battery[row])
        key = (device, int(cutoff[row]))
        value = cache.get(key)
        if value is None:
            entry = series.get(device)
            value = (np.full(len(NAMES), np.nan) if entry is None or cutoff[row] < 0
                     else signature_at(entry[0], entry[1], int(cutoff[row])))
            cache[key] = value
        out[row] = value

    coverage = np.isfinite(out).mean(axis=0)
    print(f"\n{len(cache)} distinct (device, cutoff) signatures")
    print("coverage and spread per signature feature:")
    for index, name in enumerate(NAMES):
        column = out[:, index]
        good = column[np.isfinite(column)]
        print(f"  {name:>20} {coverage[index]:6.1%}  p10 {np.percentile(good, 10):9.3f}"
              f"  median {np.median(good):9.3f}  p90 {np.percentile(good, 90):9.3f}")
    np.savez_compressed(args.out, features=out.astype(np.float32),
                        names=np.asarray(NAMES), cutoff=cutoff)
    print(f"\nwrote {args.out} ({time.time() - started:.0f}s)")


def load_signature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["features"].astype(float), data["cutoff"]


def context(args) -> dict:
    """Everything every stage needs, loaded once."""
    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    margin = frame.column("voltage") - 2.4
    signature, cutoff = load_signature(args.out)
    fold_of = v8_folds(args.folds)
    fold = np.asarray([fold_of[b] for b in frame.building])
    return {
        "frame": frame, "base": base, "margin": margin, "signature": signature,
        "cutoff": cutoff, "fold": fold, "lab": Lab(frame, base, margin),
        "columns": [NAMES.index(n) for n in PLAIN],
    }


# ------------------------------------------------- standardisation and folds

def device_weight(battery: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """One unit of weight per device, spread over the rows it contributed.

    The 48 scenarios overlap by about 85 %, so a device shows up as up to 48
    near-copies. Without this every statistic below is really a statement about
    the handful of devices that appear most often.
    """
    unique, inverse, counts = np.unique(battery[rows], return_inverse=True,
                                        return_counts=True)
    return 1.0 / counts[inverse]


class Standardiser:
    """Robust column statistics fitted on training-fold rows only."""

    def __init__(self, values: np.ndarray, weight: np.ndarray) -> None:
        self.centre = np.zeros(values.shape[1])
        self.scale = np.ones(values.shape[1])
        for column in range(values.shape[1]):
            good = np.isfinite(values[:, column])
            if good.sum() < 20:
                continue
            data = values[good, column]
            self.centre[column] = float(np.median(data))
            spread = float(np.percentile(data, 75) - np.percentile(data, 25))
            self.scale[column] = spread if spread > 1e-9 else 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        out = (values - self.centre) / self.scale
        return np.clip(np.nan_to_num(out, nan=0.0), -4.0, 4.0)


def spline_basis(margin: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear B-spline (hat) basis: low capacity, and readable band by band."""
    out = np.zeros((margin.size, knots.size))
    for index, knot in enumerate(knots):
        left = knots[index - 1] if index > 0 else knot - (knots[1] - knots[0])
        right = knots[index + 1] if index + 1 < knots.size else knot + (knots[-1] - knots[-2])
        rising = np.clip((margin - left) / max(knot - left, 1e-9), 0.0, 1.0)
        falling = np.clip((right - margin) / max(right - knot, 1e-9), 0.0, 1.0)
        out[:, index] = np.minimum(rising, falling)
    return out


def fit_logistic(design: np.ndarray, y: np.ndarray, weight: np.ndarray,
                 *, l2: float, offset: np.ndarray | None = None) -> np.ndarray:
    from scipy.optimize import minimize

    zero = np.zeros(design.shape[1])
    base = np.zeros(y.size) if offset is None else offset

    def objective(w):
        z = base + design @ w
        loss = np.logaddexp(0.0, z) - y * z
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        value = float((weight * loss).sum() / weight.sum() + l2 * w @ w)
        gradient = (design * (weight * (p - y))[:, None]).sum(axis=0) / weight.sum()
        return value, gradient + 2.0 * l2 * w

    return minimize(objective, zero, jac=True, method="L-BFGS-B",
                    options={"maxiter": 400}).x


def kmeans(values: np.ndarray, weight: np.ndarray, k: int, seed: int = 11,
           rounds: int = 60) -> np.ndarray:
    """Weighted k-means, deterministic k-means++ seeding. Centroids only."""
    rng = np.random.default_rng(seed)
    centres = values[rng.choice(values.shape[0], 1)]
    while centres.shape[0] < k:
        distance = ((values[:, None, :] - centres[None]) ** 2).sum(-1).min(axis=1)
        probability = distance * weight
        total = probability.sum()
        if total <= 0:
            centres = np.vstack([centres, values[rng.choice(values.shape[0])]])
            continue
        pick = rng.choice(values.shape[0], p=probability / total)
        centres = np.vstack([centres, values[pick]])
    for _ in range(rounds):
        label = ((values[:, None, :] - centres[None]) ** 2).sum(-1).argmin(axis=1)
        moved = centres.copy()
        for cluster in range(k):
            inside = label == cluster
            if inside.sum() == 0:
                continue
            moved[cluster] = (values[inside] * weight[inside, None]).sum(0) / weight[inside].sum()
        if np.allclose(moved, centres):
            break
        centres = moved
    return centres


# ------------------------------------------------------------------ --probe

def probe(args) -> None:
    """Does cohort membership move the margin -> risk mapping, out of fold?

    No V8 anywhere in this stage. Margin alone against margin plus a cohort
    term, fitted on training-fold landmarks and scored on the held-out
    buildings' landmarks. If the cohort term carries nothing transferable, the
    two out-of-fold likelihoods are the same and nothing after this can work.
    """
    started = time.time()
    ctx = context(args)
    frame, lab, fold = ctx["frame"], ctx["lab"], ctx["fold"]
    margin, signature = ctx["margin"], ctx["signature"][:, ctx["columns"]]
    rows = np.flatnonzero(lab.mask)
    y = frame.due[rows].astype(float)
    print(f"landmarks {rows.size}, due {int(y.sum())} "
          f"from {np.unique(frame.battery[rows][y > 0]).size} devices, "
          f"survivors from {np.unique(frame.battery[rows][y == 0]).size}")

    knots = np.asarray([0.0, 0.02, 0.04, 0.07, 0.12, 0.20, 0.35])
    weight = device_weight(frame.battery, rows)
    folds = sorted(set(fold.tolist()))
    results: dict[str, dict] = {}

    for k in (0, 3, 4, 5):
        oof = np.zeros(rows.size)
        oof_interaction = np.zeros(rows.size)
        labels = np.zeros(rows.size, int)
        for held in folds:
            inside = fold[rows] == held
            train, test = ~inside, inside
            if train.sum() < 50 or test.sum() < 5:
                continue
            scaler = Standardiser(signature[rows][train], weight[train])
            z_train = scaler(signature[rows][train])
            z_test = scaler(signature[rows][test])
            basis_train = spline_basis(margin[rows][train], knots)
            basis_test = spline_basis(margin[rows][test], knots)
            if k == 0:
                w = fit_logistic(basis_train, y[train], weight[train], l2=1e-3)
                oof[test] = basis_test @ w
                oof_interaction[test] = oof[test]
                continue
            # Cohorts are fitted on training devices, then held-out devices are
            # assigned to the nearest centroid. No held-out row moves a centroid.
            device, first = np.unique(frame.battery[rows][train], return_index=True)
            centres = kmeans(z_train[first], np.ones(first.size), k)
            assign_train = ((z_train[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
            assign_test = ((z_test[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
            labels[test] = assign_test
            dummy_train = np.eye(k)[assign_train][:, 1:]
            dummy_test = np.eye(k)[assign_test][:, 1:]
            design_train = np.column_stack([basis_train, dummy_train])
            design_test = np.column_stack([basis_test, dummy_test])
            w = fit_logistic(design_train, y[train], weight[train], l2=1e-3)
            oof[test] = design_test @ w
            wide_train = np.column_stack(
                [basis_train, dummy_train, dummy_train * margin[rows][train][:, None]])
            wide_test = np.column_stack(
                [basis_test, dummy_test, dummy_test * margin[rows][test][:, None]])
            w2 = fit_logistic(wide_train, y[train], weight[train], l2=1e-3)
            oof_interaction[test] = wide_test @ w2

        def score(z):
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
            loss = float((weight * (np.logaddexp(0.0, z) - y * z)).sum() / weight.sum())
            full = np.full(frame.due.size, -np.inf)
            full[rows] = z
            return loss, lab.report(full)

        loss, table = score(oof)
        loss2, table2 = score(oof_interaction)
        name = "margin only" if k == 0 else f"margin + {k} cohorts"
        results[name] = {"logloss": round(loss, 5), **table}
        print(f"  {name:<22} OOF logloss {loss:.5f}  all {table['all']:.4f}  "
              f"cross {table['cross']:.4f}  same {table['same']:.4f}")
        if k:
            results[f"{name} x margin"] = {"logloss": round(loss2, 5), **table2}
            print(f"  {'':<22} + margin interaction {loss2:.5f}  "
                  f"all {table2['all']:.4f}  cross {table2['cross']:.4f}  "
                  f"same {table2['same']:.4f}")

    reference = lab.report(ctx["base"], fold)
    results["V8 reference"] = reference
    print(f"\n  {'V8 (for scale)':<22} all {reference['all']:.4f}  "
          f"cross {reference['cross']:.4f}  same {reference['same']:.4f}")

    # The same nesting again, but anchored on V8 instead of on margin. Margin is
    # one number; V8 is margin plus drift plus scatter read off 64 features, some
    # of which (`temp_lifetime`, `age_days`, `beta_30`, `gap_fraction_90`) are
    # themselves regime descriptions. If the cohort term is worth something over
    # margin and nothing over V8, the regimes are real and already priced.
    print("\n  anchored on V8 -- does the cohort term survive what V8 already knows?")
    anchor = logit(ctx["base"])[rows]
    anchored: dict[str, dict] = {}
    for k in (0, 4, 5):
        for interact in (False, True):
            if k == 0 and interact:
                continue
            oof = anchor.copy()
            for held in folds:
                inside = fold[rows] == held
                train, test = ~inside, inside
                if train.sum() < 50 or test.sum() < 5:
                    continue
                if k == 0:
                    continue
                scaler = Standardiser(signature[rows][train], weight[train])
                z_train, z_test = scaler(signature[rows][train]), scaler(signature[rows][test])
                _, first = np.unique(frame.battery[rows][train], return_index=True)
                centres = kmeans(z_train[first], np.ones(first.size), k)
                a_train = ((z_train[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
                a_test = ((z_test[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
                design_train = np.eye(k)[a_train]
                design_test = np.eye(k)[a_test]
                if interact:
                    design_train = np.column_stack(
                        [design_train, design_train * margin[rows][train][:, None]])
                    design_test = np.column_stack(
                        [design_test, design_test * margin[rows][test][:, None]])
                w = fit_logistic(design_train, y[train], weight[train], l2=1e-3,
                                 offset=anchor[train])
                oof[test] = anchor[test] + design_test @ w
            loss = float((weight * (np.logaddexp(0.0, oof) - y * oof)).sum() / weight.sum())
            full = np.full(frame.due.size, -np.inf)
            full[rows] = oof
            table = lab.report(full)
            name = ("V8 only" if k == 0 else
                    f"V8 + {k} cohorts" + (" x margin" if interact else ""))
            anchored[name] = {"logloss": round(loss, 5), **table}
            print(f"    {name:<24} OOF logloss {loss:.5f}  all {table['all']:.4f}"
                  f"  cross {table['cross']:.4f}  same {table['same']:.4f}")
    results["anchored"] = anchored
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"probe": results}, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


# ------------------------------------------------------------------- --gate

def s1_scores(ctx, *, neighbours: int, bandwidth: float, floor: float = 4.0):
    """S1: the margin curve of this battery's K nearest analogues, out of fold.

    The returned quantity is deliberately a *difference of curves*, not a risk:

        delta = logit P(due | margin, cohort) - logit P(due | margin, everyone)

    Both terms are censor-aware kernel estimates at the query's own margin, both
    device-weighted, both fitted on training-fold rows alone. If cohort
    membership carries nothing about the mapping the two curves coincide and
    delta is zero everywhere, so the correction cannot move an ordering by
    accident -- it moves one only where the analogues genuinely disagree with the
    population about what this margin means.
    """
    frame, lab, fold = ctx["frame"], ctx["lab"], ctx["fold"]
    margin, signature = ctx["margin"], ctx["signature"][:, ctx["columns"]]
    delta = np.zeros(frame.due.size)
    cohort_of = np.full(frame.due.size, -1)
    query = np.flatnonzero(lab.mask)
    for held in sorted(set(fold.tolist())):
        train = np.flatnonzero((fold != held) & lab.usable)
        test = query[fold[query] == held]
        if train.size < 200 or test.size == 0:
            continue
        weight = device_weight(frame.battery, train)
        scaler = Standardiser(signature[train], weight)
        z_train = scaler(signature[train])
        z_test = scaler(signature[test])
        y = frame.due[train].astype(float)

        device = frame.battery[train]
        names = np.unique(device)
        centre = np.stack([z_train[device == d].mean(axis=0) for d in names])
        distance = ((z_test[:, None, :] - centre[None]) ** 2).sum(-1)
        near = np.argsort(distance, axis=1)[:, :neighbours]
        member = {d: np.flatnonzero(device == d) for d in names}

        gap = (margin[train][None, :] - margin[test][:, None]) / bandwidth
        kernel = np.exp(-0.5 * np.clip(gap, -8, 8) ** 2) * weight[None, :]
        population = ((kernel * y[None, :]).sum(1) + floor * 0.05) / (kernel.sum(1) + floor)
        for position in range(test.size):
            rows = np.concatenate([member[names[j]] for j in near[position]])
            local = kernel[position, rows]
            cohort = (local @ y[rows] + floor * 0.05) / (local.sum() + floor)
            delta[test[position]] = logit(np.array([cohort]))[0] - \
                logit(np.array([population[position]]))[0]
        cohort_of[test] = near[:, 0]
    return delta, cohort_of


def s2_scores(ctx, *, k: int, kappa: float, bands: tuple[float, ...] = ()):
    """S2: a small in-fold segmentation and a heavily shrunk logit offset.

    `logit(p_segment) = logit(p_V8) + beta_segment`, where `beta_segment` is the
    one-parameter logistic correction the training-fold rows of that segment
    ask for, shrunk by `n / (n + kappa)` in *devices* so a segment of four
    batteries cannot produce an extreme correction. With `bands`, the offset is
    keyed on (segment, margin band) instead -- the literal form of "this cohort
    reads 0.05 V differently".
    """
    frame, lab, fold = ctx["frame"], ctx["lab"], ctx["fold"]
    margin, signature = ctx["margin"], ctx["signature"][:, ctx["columns"]]
    base = ctx["base"]
    offset = np.zeros(frame.due.size)
    label = np.full(frame.due.size, -1)
    query = np.flatnonzero(lab.mask)
    edges = np.asarray(bands)
    for held in sorted(set(fold.tolist())):
        train = np.flatnonzero((fold != held) & lab.mask)
        test = query[fold[query] == held]
        if train.size < 100 or test.size == 0:
            continue
        weight = device_weight(frame.battery, train)
        scaler = Standardiser(signature[train], weight)
        z_train, z_test = scaler(signature[train]), scaler(signature[test])
        device = frame.battery[train]
        names, first = np.unique(device, return_index=True)
        centres = kmeans(z_train[first], np.ones(first.size), k)
        assign_train = ((z_train[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        assign_test = ((z_test[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        label[test] = assign_test

        band_train = np.searchsorted(edges, margin[train]) if edges.size else np.zeros(train.size, int)
        band_test = np.searchsorted(edges, margin[test]) if edges.size else np.zeros(test.size, int)
        y = frame.due[train].astype(float)
        anchor = logit(base[train])
        for cluster in range(k):
            for band in range(edges.size + 1):
                inside = (assign_train == cluster) & (band_train == band)
                if inside.sum() < 10:
                    continue
                beta = fit_logistic(np.ones((int(inside.sum()), 1)), y[inside],
                                    weight[inside], l2=1e-4,
                                    offset=anchor[inside])[0]
                devices = np.unique(device[inside]).size
                shrunk = beta * devices / (devices + kappa)
                hit = test[(assign_test == cluster) & (band_test == band)]
                offset[hit] = shrunk
    return offset, label


def reversal_table(ctx, score: np.ndarray, label: np.ndarray) -> dict:
    """When the cohort model overrides V8's margin ordering, is it right?"""
    lab, frame = ctx["lab"], ctx["frame"]
    base, margin = logit(ctx["base"]), ctx["margin"]
    signature = ctx["signature"]
    keep = lab.split["cross"]
    positive, negative = lab.positive[keep], lab.negative[keep]
    old = base[positive] - base[negative]
    new = score[positive] - score[negative]
    # A reversal is a *strict* disagreement: both models order the pair, and they
    # order it opposite ways. Counting ties as flips inflates the count and makes
    # "the cohort model was right" and "V8 was right" stop summing to one, which
    # is exactly the sanity check this table needs to keep.
    flipped = (old != 0) & (new != 0) & ((old > 0) != (new > 0))
    if flipped.sum() == 0:
        return {"cross_pairs": int(keep.sum()), "reversals": 0}
    correct = new[flipped] > 0
    out = {
        "cross_pairs": int(keep.sum()),
        "reversals": int(flipped.sum()),
        "reversal_rate": round(float(flipped.mean()), 4),
        "correct": round(float(correct.mean()), 4),
        "v8_correct_on_same_pairs": round(float((old[flipped] > 0).mean()), 4),
    }

    def stratify(name: str, value: np.ndarray, cuts: tuple[float, ...]) -> None:
        v = value[flipped]
        edges = np.searchsorted(np.asarray(cuts), v)
        rows = {}
        for level in range(len(cuts) + 1):
            inside = edges == level
            if inside.sum() < 15:
                continue
            rows[str(level)] = {
                "n": int(inside.sum()),
                "correct": round(float(correct[inside].mean()), 4),
            }
        out[name] = rows

    stratify("by_margin_gap",
             np.abs(margin[positive] - margin[negative]), (0.01, 0.03, 0.08))
    stratify("by_same_cohort",
             (label[positive] == label[negative]).astype(float), (0.5,))
    temp = signature[:, NAMES.index("t_mean_life")]
    stratify("by_temp_gap", np.abs(temp[positive] - temp[negative]), (1.0, 3.0))
    age = signature[:, NAMES.index("age_days")]
    stratify("by_age_gap", np.abs(age[positive] - age[negative]), (100.0, 300.0))
    plateau = signature[:, NAMES.index("v_plateau")]
    stratify("by_plateau_gap", np.abs(plateau[positive] - plateau[negative]),
             (0.01, 0.04))
    return out


def gate(args) -> None:
    started = time.time()
    ctx = context(args)
    lab, fold, base = ctx["lab"], ctx["fold"], ctx["base"]
    anchor = logit(base)
    reference = lab.report(base, fold)
    print(f"landmarks {int(lab.mask.sum())}, pairs {lab.positive.size} "
          f"({int(lab.split['cross'].sum())} cross-margin, "
          f"{int(lab.split['same'].sum())} same-margin)")
    print(f"\nV8  all {reference['all']:.4f}  cross {reference['cross']:.4f}  "
          f"same {reference['same']:.4f}   folds "
          + " ".join(f"{reference[f'f{i}']:.3f}" for i in range(5)))
    results = {"V8": reference}
    kept: list[tuple[str, dict, np.ndarray, np.ndarray]] = []
    full_strength = None

    print("\nS1 -- soft kNN analog cohort (K devices, margin bandwidth 0.02 V)")
    for neighbours in (15, 30, 50):
        delta, cohort = s1_scores(ctx, neighbours=neighbours, bandwidth=0.02)
        for strength in (0.125, 0.25, 0.5, 1.0):
            score = anchor + strength * delta
            table = lab.report(score, fold)
            won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
            name = f"S1 K={neighbours} lam={strength}"
            results[name] = {**table, "folds_won_cross": won}
            flag = "  <-- beats V8 cross" if table["cross"] > reference["cross"] else ""
            print(f"  {name:<18} all {table['all']:.4f}  cross {table['cross']:.4f}"
                  f"  same {table['same']:.4f}  cross folds won {won}/5{flag}")
            kept.append((name, table, score, cohort))
            if neighbours == 50 and strength == 1.0:
                full_strength = (name, table, score, cohort)

    print("\nS2 -- in-fold segmentation, shrunk logit offset (kappa = 25 devices)")
    for k in (3, 4, 5):
        for bands in ((), (0.05,), (0.03, 0.08)):
            offset, label = s2_scores(ctx, k=k, kappa=25.0, bands=bands)
            score = anchor + offset
            table = lab.report(score, fold)
            won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
            tag = "flat" if not bands else "x".join(str(b) for b in bands)
            name = f"S2 k={k} {tag}"
            results[name] = {**table, "folds_won_cross": won,
                             "moved": int((offset != 0).sum())}
            flag = "  <-- beats V8 cross" if table["cross"] > reference["cross"] else ""
            print(f"  {name:<18} all {table['all']:.4f}  cross {table['cross']:.4f}"
                  f"  same {table['same']:.4f}  cross folds won {won}/5{flag}")
            kept.append((name, table, score, label))

    print("\ndevice bootstrap of the cross-margin delta against V8, 300 draws")
    ranked = sorted(kept, key=lambda item: -item[1]["cross"])[:3]
    for name, table, score, label in ranked:
        boot = device_bootstrap(lab, score, base, "cross")
        results[name]["bootstrap_cross"] = boot
        print(f"  {name:<18} {boot['delta']:+.4f} "
              f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}]  "
              f"P(delta > 0) = {boot['p_positive']:.2f}")

    # The reversal question needs a candidate that actually overrides V8, so it
    # is asked of the strongest setting of the winning family as well as of the
    # shrunk one that scored best.
    print("\nreversal test -- when the cohort model overrides V8's cross-margin"
          " ordering, is it right?")
    for name, table, score, label in ranked[:1] + [full_strength]:
        entry = reversal_table(ctx, score, label)
        results[name]["reversal"] = entry
        print(f"  {name:<18} {entry['reversals']:5d} reversals of "
              f"{entry['cross_pairs']} cross-margin pairs "
              f"({entry['reversal_rate']:.1%}), correct on {entry['correct']:.1%}")
        for key in ("by_margin_gap", "by_same_cohort"):
            cells = ", ".join(f"{level}: {cell['correct']:.2f} (n={cell['n']})"
                              for level, cell in entry.get(key, {}).items())
            print(f"      {key:<16} {cells}")
    results["best"] = ranked[0][0]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


# -------------------------------------------------------------- --interpret

def interpret(args) -> None:
    """What the regimes are, what they are worth, and whether they are real.

    Three questions the gate table cannot answer on its own:

    * **What did the clustering find?** Centroids in raw units, sizes, and the
      EOL / censored counts each cohort carries.
    * **Does cohort change the meaning of a margin?** The literal PART 12 table:
      out-of-fold realised 42-day failure rate by (margin band, cohort), with the
      counts under it, so a cell backed by four devices is visible as such.
    * **Is the gate's small positive delta cohort information at all?** The
      signature is permuted among the devices present in each scenario, which
      destroys the device-to-signature link and preserves everything else, and
      S1 is re-run on the permuted signature. If the real delta sits inside the
      permuted distribution, the machinery produced it and not the cohorts.
    """
    started = time.time()
    ctx = context(args)
    frame, lab, fold = ctx["frame"], ctx["lab"], ctx["fold"]
    margin, base = ctx["margin"], ctx["base"]
    signature = ctx["signature"]
    plain = signature[:, ctx["columns"]]
    rows = np.flatnonzero(lab.mask)
    out: dict = {}

    # --- what the regimes are -------------------------------------------------
    weight = device_weight(frame.battery, rows)
    scaler = Standardiser(plain[rows], weight)
    z = scaler(plain[rows])
    device, first = np.unique(frame.battery[rows], return_index=True)
    centres = kmeans(z[first], np.ones(first.size), 5)
    label = ((z[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    print("five regimes fitted on all 1,708 landmarks (illustrative; the gate "
          "refits them inside every fold)")
    header = ["t_mean", "t_amp", "v_plateau", "v_slope", "beta", "v_std_detr",
              "age_days", "obs_frac"]
    print(f"  {'cohort':>8} {'rows':>5} {'devs':>5} {'EOL':>4} {'cens':>5} "
          f"{'due42':>6} " + " ".join(f"{h:>10}" for h in header))
    profiles = {}
    for cluster in range(5):
        inside = label == cluster
        picked = rows[inside]
        devices = np.unique(frame.battery[picked])
        eol = int(np.unique(frame.battery[picked][frame.due[picked]]).size)
        values = [float(np.median(signature[picked, NAMES.index(h if h != "t_amp"
                  else "t_amp_life")])) if h in ("t_amp",) else
                  float(np.median(signature[picked, NAMES.index({
                      "t_mean": "t_mean_life", "v_plateau": "v_plateau",
                      "v_slope": "v_slope_life", "beta": "beta_life",
                      "v_std_detr": "v_std_detr_life", "age_days": "age_days",
                      "obs_frac": "obs_frac"}[h])]))
                  for h in header]
        rate = float(frame.due[picked].mean())
        profiles[f"cohort_{cluster}"] = {
            "rows": int(inside.sum()), "devices": int(devices.size),
            "eol_devices": eol, "censored_devices": int(devices.size - eol),
            "due42_rate": round(rate, 4),
            **{h: round(v, 4) for h, v in zip(header, values)},
        }
        print(f"  {cluster:>8} {int(inside.sum()):>5} {devices.size:>5} {eol:>4} "
              f"{devices.size - eol:>5} {rate:>6.3f} "
              + " ".join(f"{v:>10.3f}" for v in values))
    out["regimes"] = profiles

    # --- does cohort change what a margin means? ------------------------------
    print("\nrealised 42-day failure rate by margin band and cohort "
          "(cohort assigned out of fold; n = landmark rows)")
    oof_label = np.full(frame.due.size, -1)
    for held in sorted(set(fold.tolist())):
        train = rows[fold[rows] != held]
        test = rows[fold[rows] == held]
        if train.size < 100 or test.size == 0:
            continue
        local = Standardiser(plain[train], device_weight(frame.battery, train))
        zt = local(plain[train])
        names_in, first_in = np.unique(frame.battery[train], return_index=True)
        centre = kmeans(zt[first_in], np.ones(first_in.size), 5)
        oof_label[test] = ((local(plain[test])[:, None] - centre[None]) ** 2).sum(-1).argmin(1)
    edges = np.asarray([0.02, 0.05, 0.10])
    band = np.searchsorted(edges, margin)
    names = ["<0.02", "0.02-0.05", "0.05-0.10", ">0.10"]
    table = {}
    # Row rates are reported next to *device-weighted* rates on purpose. The six
    # repeat false positives of `docs/FINAL_FP_ANALYSIS.md` sit at small margin in
    # up to 48 scenarios each, so a row-weighted low-margin cell is largely a
    # statement about them; one unit of weight per device is the honest number,
    # and the two disagree enough here to reverse the reading.
    print(f"  {'band':>10} " + " ".join(f"{'c' + str(c):>15}" for c in range(5)))
    print(f"  {'':>10} " + " ".join(f"{'row / device':>15}" for _ in range(5)))
    for level, title in enumerate(names):
        cells, line = {}, []
        for cluster in range(5):
            picked = rows[(band[rows] == level) & (oof_label[rows] == cluster)]
            if picked.size < 10:
                line.append(f"{'-':>15}")
                continue
            per_device = device_weight(frame.battery, picked)
            weighted = float((per_device * frame.due[picked]).sum() / per_device.sum())
            rate = float(frame.due[picked].mean())
            cells[f"c{cluster}"] = {
                "n": int(picked.size),
                "devices": int(np.unique(frame.battery[picked]).size),
                "due_devices": int(np.unique(frame.battery[picked][frame.due[picked]]).size),
                "rate": round(rate, 4),
                "device_weighted": round(weighted, 4),
                "v8_device_weighted": round(
                    float((per_device * base[picked]).sum() / per_device.sum()), 4),
            }
            line.append(f"{rate:>6.3f} /{weighted:>6.3f} ")
        table[title] = cells
        print(f"  {title:>10} " + " ".join(line))
    out["margin_by_cohort"] = table
    ordered = [
        (title, min(c["device_weighted"] for c in cells.values()),
         max(c["device_weighted"] for c in cells.values()))
        for title, cells in table.items() if len(cells) >= 2
    ]
    for title, low, high in ordered:
        print(f"    {title:>10}  device-weighted spread {low:.3f} to {high:.3f}")
    inverted = any(
        high_band[2] > low_band[1]
        for low_band, high_band in zip(ordered, ordered[1:])
    )
    out["cross_margin_inversion"] = bool(inverted)
    print(f"  a worse cohort at a wider margin out-ranks a better cohort at a "
          f"tighter one: {'yes' if inverted else 'NO'}")

    # --- is the gate delta cohort information? --------------------------------
    print("\npermutation control: the same S1 on a signature shuffled among the "
          "devices present in each scenario")
    real_delta, _ = s1_scores(ctx, neighbours=50, bandwidth=0.02)
    anchor = logit(base)
    real = lab.concordance(anchor + 0.125 * real_delta, lab.split["cross"]) \
        - lab.concordance(anchor, lab.split["cross"])
    print(f"  measured S1 K=50 lam=0.125 cross-margin delta: {real:+.4f}")
    shuffled = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        permuted = signature.copy()
        for index in np.unique(frame.scenario):
            block = np.flatnonzero(frame.scenario == index)
            permuted[block] = signature[rng.permutation(block)]
        fake = dict(ctx)
        fake["signature"] = permuted
        delta, _ = s1_scores(fake, neighbours=50, bandwidth=0.02)
        value = lab.concordance(anchor + 0.125 * delta, lab.split["cross"]) \
            - lab.concordance(anchor, lab.split["cross"])
        shuffled.append(value)
        print(f"    seed {seed}: {value:+.4f}")
    shuffled = np.asarray(shuffled)
    out["permutation"] = {
        "real": round(float(real), 4),
        "shuffled_mean": round(float(shuffled.mean()), 4),
        "shuffled_max": round(float(shuffled.max()), 4),
        "shuffled_sd": round(float(shuffled.std()), 4),
        "exceeded_by": int((shuffled >= real).sum()),
        "draws": int(shuffled.size),
    }
    print(f"  shuffled: mean {shuffled.mean():+.4f}, sd {shuffled.std():.4f}, "
          f"max {shuffled.max():+.4f}; {int((shuffled >= real).sum())} of "
          f"{shuffled.size} match or beat the real signature")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--series", type=Path, default=Path("outputs/fj_series.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_segment.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_segment.json"))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args()
    if args.interpret:
        interpret(args)
        return
    if args.build:
        build(args)
        return
    if args.probe:
        probe(args)
        return
    if args.gate:
        gate(args)
        return
    parser.error("choose a stage: --build, --probe, --gate")


if __name__ == "__main__":
    main()
