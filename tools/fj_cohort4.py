"""Cohort 4 as a continuous signature: one parameter, and only where it acts.

`docs/FINAL_SEGMENT_EXPERIMENT.md` closed general segmentation and left exactly
one cell standing: devices that are cold, strongly seasonal and badly observed
over their whole life are under-priced by V8 **7.9x** above 0.10 V of margin, on
22 distinct failure devices in 10 buildings and 4 of the 5 folds. V8's shipped
`staleness` and `gap_fraction_90` are median 0 on the decision population, so it
cannot see the whole-life version of that.

This asks the cheap question before any large one: can that cell be written as a
*continuous, transferable* device score rather than a cluster id, and is it worth
anything to **cross-margin ordering** where it acts?

    score = logit(p_V8) + alpha * s(device) * gate(margin)

`s` is built from whole-life causal signature features only -- no identity, no
outcome, no future. Three families are tested separately so a win can be
attributed, plus a zero-parameter equal-weight combination and, as a ceiling
rather than a candidate, a direction fitted inside the fold:

* `obs`      whole-history observation fraction, inverted;
* `cold`     lifetime mean temperature, inverted, and cold-day fraction;
* `amp`      lifetime seasonal amplitude;
* `sum3`     the equal-weight standardised sum of the three, **no fitted weights**;
* `fitted`   an in-fold logistic direction over the three, reported as an upper
             bound on what this family could ever be worth.

`alpha` is chosen by **nested** selection: inside each outer building fold, the
four training groups are cross-validated against each other to pick alpha, and
the held-out group is scored with a value it never saw. The gate is the hard
`margin > 0.10` the finding was stated at, with a smooth sigmoid variant so the
result cannot rest on a cliff.

Deployment is order-only: V8's own per-scenario probability multiset is reassigned
by the new rank, so risk mass is unchanged by construction. Concordance is
invariant to that remap (it is monotone within scenario); it is asserted here
because it is what a planner run would have to ship.

    python tools/fj_cohort4.py --gate
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

from tools.fj_fit import fit_weights  # noqa: E402
from tools.fj_segment import (  # noqa: E402
    Lab,
    Standardiser,
    context,
    device_bootstrap,
    device_weight,
    logit,
)
from tools.fj_signature import NAMES  # noqa: E402

HIGH_MARGIN = 0.10
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# Each family is (signature feature, sign) pairs; sign +1 means "larger value is
# more cohort-4 like". Cohort 4 is cold (low t_mean_life), strongly seasonal
# (high t_amp_life) and badly observed (low obs_frac).
FAMILIES = {
    "obs": (("obs_frac", -1.0),),
    "cold": (("t_mean_life", -1.0), ("t_cold_frac", +1.0)),
    "amp": (("t_amp_life", +1.0),),
    "sum3": (("obs_frac", -1.0), ("t_mean_life", -1.0), ("t_amp_life", +1.0)),
}


def family_columns(name: str) -> tuple[list[int], np.ndarray]:
    pairs = FAMILIES[name]
    return [NAMES.index(f) for f, _ in pairs], np.asarray([s for _, s in pairs])


def build_score(ctx, family: str, train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The cohort-4 similarity for `rows`, standardised on `train` alone.

    Returned on a within-fold z scale and clipped, so one alpha means the same
    thing in every fold and no single device can dominate through an outlier.
    """
    columns, signs = family_columns(family)
    values = ctx["signature"][:, columns]
    scaler = Standardiser(values[train], device_weight(ctx["frame"].battery, train))
    return np.clip((scaler(values[rows]) * signs).mean(axis=1), -3.0, 3.0)


def fitted_score(ctx, train: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Ceiling arm: the same three inputs, direction fitted inside the fold.

    Fitted on high-margin training landmarks only, with a pairwise ranking loss
    so it learns an *order* rather than a level, and heavily regularised. This is
    reported to bound the family, not proposed as a candidate.
    """
    columns, signs = family_columns("sum3")
    values = ctx["signature"][:, columns]
    scaler = Standardiser(values[train], device_weight(ctx["frame"].battery, train))
    frame = ctx["frame"]
    high = train[ctx["margin"][train] > HIGH_MARGIN]
    design_all = scaler(values)
    positives, negatives = [], []
    for index in np.unique(frame.scenario[high]):
        block = high[frame.scenario[high] == index]
        due = block[frame.due[block]]
        alive = block[~frame.due[block]]
        if due.size == 0 or alive.size == 0:
            continue
        positives.append(np.repeat(due, alive.size))
        negatives.append(np.tile(alive, due.size))
    if not positives:
        return np.zeros(rows.size)
    pos, neg = np.concatenate(positives), np.concatenate(negatives)
    unique, inverse, counts = np.unique(frame.battery[pos], return_inverse=True,
                                        return_counts=True)
    weight = 1.0 / counts[inverse]
    w = fit_weights(design_all, pos, neg, weight * (pos.size / weight.sum()), l2=0.5)
    if np.allclose(w, 0.0):
        return np.zeros(rows.size)
    w = w / np.linalg.norm(w)
    return np.clip(design_all[rows] @ w * np.sign(w @ signs or 1.0), -3.0, 3.0)


def gate_weight(margin: np.ndarray, smooth: bool) -> np.ndarray:
    if not smooth:
        return (margin > HIGH_MARGIN).astype(float)
    return 1.0 / (1.0 + np.exp(-(margin - HIGH_MARGIN) / 0.02))


def order_only(ctx, score: np.ndarray) -> np.ndarray:
    """Reassign V8's own per-scenario probability multiset by the new order.

    The planner reads a probability, so a reordering has to be spent as one. The
    multiset and therefore the per-scenario risk mass are unchanged by
    construction; this returns the probabilities a planner run would receive.
    """
    frame, base = ctx["frame"], ctx["base"]
    out = base.copy()
    for index in np.unique(frame.scenario):
        rows = np.flatnonzero(frame.scenario == index)
        levels = np.sort(base[rows])[::-1]
        out[rows[np.argsort(-score[rows], kind="stable")]] = levels
    return out


def high_margin_split(lab: Lab, margin: np.ndarray) -> dict[str, np.ndarray]:
    """Cross-margin pairs sliced by where the correction is allowed to act."""
    cross = lab.split["cross"]
    positive_high = margin[lab.positive] > HIGH_MARGIN
    negative_high = margin[lab.negative] > HIGH_MARGIN
    return {
        "cross_hi_any": cross & (positive_high | negative_high),
        "cross_hi_due": cross & positive_high,
        "cross_hi_both": cross & positive_high & negative_high,
        "cross_lo_only": cross & ~positive_high & ~negative_high,
    }


def evaluate(ctx, lab, fold, score, extra):
    table = lab.report(score, fold)
    for name, keep in extra.items():
        table[name] = round(lab.concordance(score, keep), 4)
    return table


def credit(ctx, lab, score, reference, keep) -> dict:
    """Which devices the change is actually made of, and how concentrated it is."""
    positive, negative = lab.positive[keep], lab.negative[keep]
    old = np.sign(reference[positive] - reference[negative])
    new = np.sign(score[positive] - score[negative])
    moved = np.flatnonzero(old != new)
    if moved.size == 0:
        return {"pairs_moved": 0}
    gain = ((new[moved] > 0).astype(float) - (old[moved] > 0).astype(float))
    devices = lab.frame.battery[positive[moved]]
    net: dict[str, float] = {}
    for device, value in zip(devices, gain):
        net[device] = net.get(device, 0.0) + float(value)
    ordered = sorted(net.items(), key=lambda kv: -abs(kv[1]))
    total = sum(net.values())
    helped = sum(1 for v in net.values() if v > 0)
    return {
        "pairs_moved": int(moved.size),
        "net_pairs_gained": round(float(total), 1),
        "due_devices_touched": len(net),
        "due_devices_net_positive": helped,
        "due_devices_net_negative": sum(1 for v in net.values() if v < 0),
        "top5_share_of_net": round(
            float(sum(v for _, v in ordered[:5]) / total), 3) if total else None,
        "top5": [[d, round(v, 1)] for d, v in ordered[:5]],
    }


def nested_alpha(ctx, lab, fold, family, smooth, outer) -> float:
    """Pick alpha on the four training groups, cross-validated among themselves."""
    rows = np.flatnonzero(lab.mask)
    inner_groups = [g for g in sorted(set(fold.tolist())) if g != outer]
    totals = {alpha: 0.0 for alpha in ALPHAS}
    for inner in inner_groups:
        train = np.flatnonzero((fold != outer) & (fold != inner) & lab.mask)
        test = rows[fold[rows] == inner]
        if train.size < 100 or test.size == 0:
            continue
        similarity = np.zeros(ctx["frame"].due.size)
        target = np.flatnonzero(lab.mask)
        similarity[target] = (fitted_score(ctx, train, target) if family == "fitted"
                              else build_score(ctx, family, train, target))
        gate = gate_weight(ctx["margin"], smooth)
        keep = (lab.split["cross"]
                & (fold[lab.positive] == inner) & (fold[lab.negative] == inner))
        if keep.sum() < 50:
            continue
        for alpha in ALPHAS:
            candidate = logit(ctx["base"]) + alpha * similarity * gate
            totals[alpha] += lab.concordance(candidate, keep) * keep.sum()
    return max(ALPHAS, key=lambda a: totals[a])


def run(args) -> None:
    started = time.time()
    ctx = context(args)
    lab, fold, base, margin = ctx["lab"], ctx["fold"], ctx["base"], ctx["margin"]
    anchor = logit(base)
    extra = high_margin_split(lab, margin)
    reference = evaluate(ctx, lab, fold, base, extra)
    rows = np.flatnonzero(lab.mask)
    high = rows[margin[rows] > HIGH_MARGIN]
    print(f"landmarks {rows.size}, of which {high.size} sit above "
          f"{HIGH_MARGIN} V ({np.unique(ctx['frame'].battery[high]).size} devices, "
          f"{int(ctx['frame'].due[high].sum())} due rows from "
          f"{np.unique(ctx['frame'].battery[high][ctx['frame'].due[high]]).size} devices)")
    print(f"cross-margin pairs {int(lab.split['cross'].sum())}; of these "
          + ", ".join(f"{name} {int(keep.sum())}" for name, keep in extra.items()))
    print(f"\nV8   all {reference['all']:.4f}  cross {reference['cross']:.4f}  "
          f"same {reference['same']:.4f}  cross|hi-due {reference['cross_hi_due']:.4f}"
          f"  cross|hi-any {reference['cross_hi_any']:.4f}")
    results = {"V8": reference}

    for smooth in (False, True):
        tag = "sigmoid" if smooth else "hard"
        print(f"\ngate: {tag} at margin > {HIGH_MARGIN} V")
        gate = gate_weight(margin, smooth)
        for family in ("obs", "cold", "amp", "sum3", "fitted"):
            # Out-of-fold similarity, then the fixed alpha sweep, then the
            # nested pick. The sweep is reported so the nested number can be read
            # against what an oracle alpha would have taken.
            similarity = np.zeros(ctx["frame"].due.size)
            picked = np.zeros(ctx["frame"].due.size)
            chosen = {}
            for held in sorted(set(fold.tolist())):
                train = np.flatnonzero((fold != held) & lab.mask)
                test = rows[fold[rows] == held]
                if train.size < 100 or test.size == 0:
                    continue
                similarity[test] = (fitted_score(ctx, train, test) if family == "fitted"
                                    else build_score(ctx, family, train, test))
                alpha = nested_alpha(ctx, lab, fold, family, smooth, held)
                chosen[f"f{held}"] = alpha
                picked[test] = alpha
            for alpha in ALPHAS[1:]:
                table = evaluate(ctx, lab, fold, anchor + alpha * similarity * gate, extra)
                won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
                results[f"{tag} {family} a={alpha}"] = {**table, "folds_won_cross": won}
            nested = anchor + picked * similarity * gate
            table = evaluate(ctx, lab, fold, nested, extra)
            won = sum(table[f"x{i}"] > reference[f"x{i}"] for i in range(5))
            name = f"{tag} {family} nested"
            results[name] = {**table, "folds_won_cross": won, "alpha_by_fold": chosen}
            sweep = " ".join(
                f"{results[f'{tag} {family} a={a}']['cross_hi_due']:.4f}"
                for a in ALPHAS[1:])
            print(f"  {family:>7}  nested alpha {list(chosen.values())}  "
                  f"cross {table['cross']:.4f}  cross|hi-due {table['cross_hi_due']:.4f}"
                  f"  (sweep {sweep})  folds {won}/5")

    # The best nested arm by the metric the finding was stated in.
    candidates = [k for k in results if k.endswith("nested")]
    best = max(candidates, key=lambda k: results[k]["cross_hi_due"])
    print(f"\nbest nested arm by cross|hi-due: {best}")
    tag, family = best.split()[0], best.split()[1]
    smooth = tag == "sigmoid"
    similarity = np.zeros(ctx["frame"].due.size)
    picked = np.zeros(ctx["frame"].due.size)
    for held in sorted(set(fold.tolist())):
        train = np.flatnonzero((fold != held) & lab.mask)
        test = rows[fold[rows] == held]
        if train.size < 100 or test.size == 0:
            continue
        similarity[test] = (fitted_score(ctx, train, test) if family == "fitted"
                            else build_score(ctx, family, train, test))
        picked[test] = nested_alpha(ctx, lab, fold, family, smooth, held)
    score = anchor + picked * similarity * gate_weight(margin, smooth)
    table = results[best]
    print(f"  all {table['all']:.4f} (V8 {reference['all']:.4f})   "
          f"cross {table['cross']:.4f} (V8 {reference['cross']:.4f})   "
          f"same {table['same']:.4f} (V8 {reference['same']:.4f})")
    print("  folds (cross-margin, both rows held out): "
          + " ".join(f"f{i} {table[f'x{i}']:.3f}/{reference[f'x{i}']:.3f}"
                     for i in range(5)))
    for kind, label in (("cross", "all cross-margin"),
                        ("cross_hi_due", "cross-margin, due above 0.10 V"),
                        ("cross_hi_any", "cross-margin, either above 0.10 V")):
        keep = lab.split["cross"] if kind == "cross" else extra[kind]
        boot = device_bootstrap(lab, score, base, "cross") if kind == "cross" else \
            _bootstrap_on(lab, score, base, keep)
        results.setdefault("bootstrap", {})[kind] = boot
        print(f"  bootstrap {label:<32} {boot['delta']:+.4f} "
              f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}]  P(d>0) {boot['p_positive']:.2f}")

    from tools.fj_segment import reversal_table
    dummy = np.zeros(ctx["frame"].due.size, int)
    results["reversal"] = reversal_table(ctx, score, dummy)
    entry = results["reversal"]
    print(f"  reversals {entry['reversals']} of {entry['cross_pairs']} "
          f"cross-margin pairs ({entry['reversal_rate']:.1%}), correct on "
          f"{entry['correct']:.1%}")
    results["credit"] = credit(ctx, lab, score, base, extra["cross_hi_due"])
    give = results["credit"]
    print(f"  the change is {give['pairs_moved']} moved pairs, net "
          f"{give['net_pairs_gained']:+.0f}, over {give['due_devices_touched']} due "
          f"devices ({give['due_devices_net_positive']} helped, "
          f"{give['due_devices_net_negative']} hurt); top 5 devices carry "
          f"{give['top5_share_of_net']} of the net")

    remapped = order_only(ctx, score)
    for index in np.unique(ctx["frame"].scenario):
        block = ctx["frame"].scenario == index
        np.testing.assert_allclose(np.sort(remapped[block]), np.sort(base[block]),
                                   atol=1e-12)
        np.testing.assert_allclose(remapped[block].sum(), base[block].sum(), atol=1e-9)
    order_table = evaluate(ctx, lab, fold, remapped, extra)
    results["order_only"] = order_table
    print(f"  order-only remap: multiset and risk mass identical in all 48 "
          f"scenarios; cross {order_table['cross']:.4f}, "
          f"cross|hi-due {order_table['cross_hi_due']:.4f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


def _bootstrap_on(lab, score, reference, keep, draws: int = 300, seed: int = 7):
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
        new = (gap_new[picked] > 0).sum() + 0.5 * (gap_new[picked] == 0).sum()
        old = (gap_old[picked] > 0).sum() + 0.5 * (gap_old[picked] == 0).sum()
        deltas[draw] = (new - old) / picked.size
    return {
        "delta": round(float(deltas.mean()), 4),
        "lo": round(float(np.percentile(deltas, 2.5)), 4),
        "hi": round(float(np.percentile(deltas, 97.5)), 4),
        "p_positive": round(float((deltas > 0).mean()), 3),
        "devices": int(devices.size),
    }


def _delta_on(lab, score, reference, keep) -> float:
    positive, negative = lab.positive[keep], lab.negative[keep]
    if positive.size == 0:
        return float("nan")
    new = score[positive] - score[negative]
    old = reference[positive] - reference[negative]
    win = lambda g: ((g > 0).sum() + 0.5 * (g == 0).sum()) / g.size  # noqa: E731
    return float(win(new) - win(old))


def falsify(args) -> None:
    """Three ways the `obs_frac` result could be an artefact, priced.

    The gate arm rests on 140 moved pairs whose net is carried by five devices,
    which is the shape of every artefact this project has caught
    (`weeks_observed` in `docs/FINAL_FRAILTY.md`, the six repeats in
    `docs/FINAL_FP_ANALYSIS.md`). So:

    * **permutation** -- shuffle the signature among the devices present in each
      scenario and re-run, which destroys the device-to-signature link and keeps
      everything else. This is the control that closed general segmentation;
    * **jackknife** -- drop the largest contributing due devices, one at a time
      and cumulatively, and see what survives;
    * **mechanism** -- is `obs_frac` acting as a stale-voltage proxy, and do V8's
      own 90-day gap features already carry it?
    """
    started = time.time()
    ctx = context(args)
    lab, fold, base, margin = ctx["lab"], ctx["fold"], ctx["base"], ctx["margin"]
    anchor, frame = logit(base), ctx["frame"]
    extra = high_margin_split(lab, margin)
    keep = extra["cross_hi_due"]
    rows = np.flatnonzero(lab.mask)
    out: dict = {}

    def arm(signature: np.ndarray) -> np.ndarray:
        """The shipped arm -- hard gate, `obs`, nested alpha -- on a signature."""
        local = dict(ctx)
        local["signature"] = signature
        similarity = np.zeros(frame.due.size)
        picked = np.zeros(frame.due.size)
        for held in sorted(set(fold.tolist())):
            train = np.flatnonzero((fold != held) & lab.mask)
            test = rows[fold[rows] == held]
            if train.size < 100 or test.size == 0:
                continue
            similarity[test] = build_score(local, "obs", train, test)
            picked[test] = nested_alpha(local, lab, fold, "obs", False, held)
        return anchor + picked * similarity * gate_weight(margin, False)

    score = arm(ctx["signature"])
    real = _delta_on(lab, score, base, keep)
    real_all = _delta_on(lab, score, base, lab.split["cross"])
    print(f"measured: cross|hi-due {real:+.4f}, all cross-margin {real_all:+.4f}")

    print("\npermutation control -- obs_frac shuffled among the devices in each scenario")
    null = []
    for seed in range(12):
        rng = np.random.default_rng(400 + seed)
        permuted = ctx["signature"].copy()
        for index in np.unique(frame.scenario):
            block = np.flatnonzero(frame.scenario == index)
            permuted[block] = ctx["signature"][rng.permutation(block)]
        value = _delta_on(lab, arm(permuted), base, keep)
        null.append(value)
        print(f"  seed {seed:2d}: {value:+.4f}")
    null = np.asarray(null)
    beaten = int((null >= real).sum())
    out["permutation"] = {
        "real": round(real, 4), "mean": round(float(null.mean()), 4),
        "sd": round(float(null.std()), 4), "max": round(float(null.max()), 4),
        "exceeded_by": beaten, "draws": int(null.size),
    }
    print(f"  shuffled mean {null.mean():+.4f}, sd {null.std():.4f}, "
          f"max {null.max():+.4f}; {beaten} of {null.size} match or beat the real one")

    print("\njackknife -- who is the gain made of?")
    positive, negative = lab.positive[keep], lab.negative[keep]
    new = np.sign(score[positive] - score[negative])
    old = np.sign(base[positive] - base[negative])
    moved = np.flatnonzero(new != old)
    gain = (new[moved] > 0).astype(float) - (old[moved] > 0).astype(float)
    devices = frame.battery[positive[moved]]
    net: dict[str, float] = {}
    for device, value in zip(devices, gain):
        net[device] = net.get(device, 0.0) + float(value)
    ordered = sorted(net.items(), key=lambda kv: -kv[1])
    print("  top contributors (due device, net pairs gained):")
    for device, value in ordered[:6]:
        held = frame.battery[positive] == device
        print(f"    {device}  {value:+5.0f}  over {int(held.sum())} of its pairs")
    print("  worst:")
    for device, value in ordered[-3:]:
        print(f"    {device}  {value:+5.0f}")

    steps = {}
    for drop in (1, 2, 3, 5, 8):
        banned = {d for d, _ in ordered[:drop]}
        mask = keep.copy()
        mask[keep] = ~np.isin(frame.battery[positive], list(banned))
        value = _delta_on(lab, score, base, mask)
        steps[f"drop_top_{drop}"] = {
            "delta": round(value, 4), "pairs": int(mask.sum()),
            "due_devices": int(np.unique(frame.battery[lab.positive[mask]]).size),
        }
        print(f"  without the top {drop} device(s): cross|hi-due delta "
              f"{value:+.4f} over {int(mask.sum())} pairs")
    out["jackknife"] = {"net_by_device": {d: v for d, v in ordered}, "steps": steps}

    loo = []
    for device in np.unique(frame.battery[positive]):
        mask = keep.copy()
        mask[keep] = frame.battery[positive] != device
        loo.append(_delta_on(lab, score, base, mask))
    loo = np.asarray(loo)
    out["leave_one_device_out"] = {
        "min": round(float(loo.min()), 4), "max": round(float(loo.max()), 4),
        "median": round(float(np.median(loo)), 4),
        "still_positive": int((loo > 0).sum()), "devices": int(loo.size),
    }
    print(f"  leave-one-device-out over {loo.size} due devices: "
          f"min {loo.min():+.4f}, median {np.median(loo):+.4f}, max {loo.max():+.4f}; "
          f"positive in {int((loo > 0).sum())}/{loo.size}")

    print("\nmechanism -- what is obs_frac doing that V8 cannot see?")
    from bsai.features import FEATURE_NAMES
    obs = ctx["signature"][:, NAMES.index("obs_frac")]
    high = rows[margin[rows] > HIGH_MARGIN]
    due = high[frame.due[high]]
    alive = high[~frame.due[high]]
    weights = (device_weight(frame.battery, due), device_weight(frame.battery, alive))
    print(f"  above {HIGH_MARGIN} V, device-weighted medians:")
    for label, column in (("obs_frac (whole life)", obs),
                          ("staleness", frame.features[:, FEATURE_NAMES.index("staleness")]),
                          ("gap_fraction_90", frame.features[:, FEATURE_NAMES.index("gap_fraction_90")]),
                          ("margin", margin),
                          ("p_V8", base)):
        column = np.asarray(column, dtype=float)
        print(f"    {label:>22}  due {np.median(column[due]):8.4f}   "
              f"survivors {np.median(column[alive]):8.4f}")
    out["mechanism"] = {
        "obs_frac_due_median": round(float(np.median(obs[due])), 4),
        "obs_frac_survivor_median": round(float(np.median(obs[alive])), 4),
        "correlation_obs_margin": round(float(np.corrcoef(obs[rows], margin[rows])[0, 1]), 4),
        "correlation_obs_pv8": round(float(np.corrcoef(obs[rows], base[rows])[0, 1]), 4),
        "due_devices_above_threshold": int(np.unique(frame.battery[due]).size),
    }
    print(f"  corr(obs_frac, margin) = {out['mechanism']['correlation_obs_margin']:+.3f}, "
          f"corr(obs_frac, p_V8) = {out['mechanism']['correlation_obs_pv8']:+.3f}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.report} ({time.time() - started:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fj_segment.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/fj_cohort4.json"))
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--falsify", action="store_true")
    args = parser.parse_args()
    if args.gate:
        run(args)
        return
    if args.falsify:
        falsify(args)
        return
    parser.error("choose a stage: --gate, --falsify")


if __name__ == "__main__":
    main()
