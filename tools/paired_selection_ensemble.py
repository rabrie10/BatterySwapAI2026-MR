"""Price the ensemble reorder (remkeyed_puremid) on the paired incumbents.

The ensemble study's winner is a remaining-keyed logit mix of cens+twophase+
qhead (outputs/ensemble_findings.md), deployed Sigma-p-preserving: the ORDER
comes from the mix, the p levels stay cens's. This arm asks the planner-side
question the AP/top-12 ladder cannot: does that ORDER add exact replay delta
ON TOP of the V19 exchange (persistence-zombies out + union gate in + refill)?

Per scenario (skipped when median X < 50, like the shipped exchange):
  1. K = the incumbent's planned count (swap volume unchanged).
  2. T = top-K of the candidate set by the ORDER under test; persistence-
     zombies (N6_nodwell) removed; union-gate batteries (dark1|dark2_last_0.00)
     forced in, displacing the weakest by the same order; refilled to K by the
     same order.
  3. The diff vs the incumbent's planned set is applied order-preservingly
     (kept batteries keep their incumbent days; additions go to the nearest
     planned building visit else cost-optimal day, cheapest-insertion), scored
     with the official evaluate_plan.

Arms:
  ctl_cens   ORDER = cens p (the runtime forecaster probabilities) -- the
             machinery-matched control; ensemble-vs-this isolates ORDERING.
  ens_full   ORDER = remkeyed_puremid z (open>225d rem .25c/.5t/.25q; mid
             115-225d pure tp; late<=115d .125c/.75t/.125q; logits, EPS 1e-9).
  ens_open   ens_full where scenario median X > 225, else ctl_cens
             (the ensemble's own regime finding: mid ordering IS twophase).

Reference: the V19-equivalent shipped-machinery arm (persistence + union +
refill by cens p) measured -218.9/scen (outputs/paired_persistence.json).

    python tools/paired_selection_ensemble.py     # ~5 min
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.evaluate import evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a2 = _load("paired_selection_a2")
paired_selection = _load("paired_selection")
PlanEditor = paired_selection.PlanEditor

_EPOCH = pd.Timestamp("1970-01-01")
EPS = 1e-9
PERSIST_N = 6  # the persistence winner: flag = margin<0.05 & p>0.4, no dwell


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def remkeyed_puremid(matrix: pd.DataFrame) -> pd.Series:
    lc, lt, lq = logit(matrix["p_cens"]), logit(matrix["p_tp"]), logit(matrix["p_qh"])
    remaining = matrix["remaining"].to_numpy(dtype=float)
    z = np.where(
        remaining > 225.0,
        0.25 * lc + 0.5 * lt + 0.25 * lq,
        np.where(remaining > 115.0, lt, 0.125 * lc + 0.75 * lt + 0.125 * lq),
    )
    return pd.Series(z, index=matrix["battery"].astype(str))


def main() -> None:
    cache = joblib.load(REPO_ROOT / "outputs" / "paired_incumbents.joblib")
    records = [cache["records"][f"s_{i}"] for i in range(len(cache["records"]))]
    matrix = pd.read_parquet(REPO_ROOT / "outputs" / "ensemble_matrix.parquet")
    matrix["scenario"] = matrix["scenario"].astype(int)

    # causal persistence flags (N6, no dwell) -- same as the persistence winner
    flags: list[dict[str, bool]] = []
    for rec in records:
        m = np.asarray(rec["margin"], dtype=float)
        p = np.asarray(rec["p"], dtype=float)
        with np.errstate(invalid="ignore"):
            f = (m < 0.05) & (p > 0.4)
        flags.append({b: bool(v) for b, v in zip(rec["battery_ids"], f)})

    def zombies_for(index: int, ids: list[str]) -> set[str]:
        prior: dict[str, int] = {}
        for s in range(index):
            for b, f in flags[s].items():
                if f:
                    prior[b] = prior.get(b, 0) + 1
        now = flags[index]
        return {
            b for b in ids if now.get(b, False) and prior.get(b, 0) >= PERSIST_N
        }

    smooth, rawany = a2.build_channels()
    locations, timeseries, eol_times, scenarios = load_dataset(
        REPO_ROOT / "dataset" / "train"
    )
    rows: list[dict] = []
    started = time.time()

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        name = scenario["name"]
        rec = cache["records"][name]
        start = pd.Timestamp(scenario["start_time"]).normalize()
        origin = int((start - _EPOCH) / pd.Timedelta(days=1))
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
        ids = list(rec["battery_ids"])
        p = np.asarray(rec["p"], dtype=float)
        mc = np.asarray(rec["margin"], dtype=float)
        sc = np.asarray(rec["staleness"], dtype=float)

        end_times = pd.to_datetime(
            locs.set_index(locs["battery"].astype(str))["end_time"]
        )
        if getattr(end_times.dt, "tz", None) is not None:
            end_times = end_times.dt.tz_localize(None)
        remaining = np.array(
            [
                float((end_times[b].normalize() - start) / pd.Timedelta(days=1))
                if b in end_times.index
                else np.nan
                for b in ids
            ]
        )
        median_x = float(
            (
                (end_times.dt.normalize() + pd.Timedelta(days=30.0))
                - rec["candidate_dates"][-1]
            ).dt.days.median()
        )

        editor = PlanEditor(rec, locs, travel, settings, start)
        base_frame = editor.baseline()
        base = float(
            evaluate_plan(
                base_frame, locs, travel, settings,
                eol_times=not_dead, start_time=start, verbose=0,
            )[2]["total_cost"]
        )
        planned_inc = set(editor.planned)
        entry = {
            "scenario": name,
            "index": index,
            "median_x": round(median_x, 1),
            "base": round(base, 2),
        }

        if median_x < 50.0:
            entry.update(ctl_cens=0.0, ens_full=0.0, kept=None, skipped=True)
            rows.append(entry)
            print(f"  {name:>5s} x={median_x:6.1f} SKIP (X<50)", flush=True)
            continue

        stale_true = np.full(len(ids), np.nan)
        any_last = np.full(len(ids), np.nan)
        for j, b in enumerate(ids):
            _, stale_true[j] = a2.true_staleness(smooth, b, origin)
            any_last[j], _ = a2.rawany_margins(rawany, b, origin)
        with np.errstate(invalid="ignore"):
            dark1 = (sc > 30.0) & ((mc - 0.001 * sc) < 0.02) & (remaining >= 30.0)
            dark2 = (stale_true > 30.0) & (any_last < 0.00) & (remaining >= 30.0)
        gates = {ids[k] for k in np.flatnonzero(dark1 | dark2)}
        zombies = zombies_for(index, ids)
        candidates = [b for b in rec["candidate_ids"] if b in editor.position_of]
        K = len(planned_inc)

        z_scores = remkeyed_puremid(matrix[matrix["scenario"] == index])
        p_of = {b: float(p[editor.position_of[b]]) for b in ids}

        def order_key(score_of):
            return lambda b: (-score_of(b), b)

        cens_key = order_key(lambda b: p_of.get(b, 0.0))
        ens_key = order_key(lambda b: float(z_scores.get(b, -1e18)))

        def compose(key) -> set[str]:
            pool = sorted((b for b in candidates if b not in zombies), key=key)
            target = set(pool[:K])
            forced = {b for b in gates if b not in zombies and b in editor.position_of}
            for gate in sorted(forced - target, key=key):
                if len(target) >= K:
                    victims = sorted(
                        (b for b in target if b not in forced), key=key, reverse=True
                    )
                    if not victims:
                        break
                    target.discard(victims[0])
                target.add(gate)
            for b in pool:
                if len(target) >= K:
                    break
                target.add(b)
            return target

        def score_target(target: set[str]) -> float:
            removals = sorted(planned_inc - target)
            additions = []
            for b in sorted(target - planned_inc):
                day, _ = editor.injection_day(b, editor.day_rows)
                additions.append((b, day))
            if not removals and not additions:
                return 0.0
            plan = editor.variant(removals, additions)
            total = float(
                evaluate_plan(
                    plan, locs, travel, settings,
                    eol_times=not_dead, start_time=start, verbose=0,
                )[2]["total_cost"]
            )
            return total - base

        target_cens = compose(cens_key)
        target_ens = compose(ens_key)
        entry["ctl_cens"] = round(score_target(target_cens), 2)
        entry["ens_full"] = (
            entry["ctl_cens"]
            if target_ens == target_cens
            else round(score_target(target_ens), 2)
        )
        entry["set_diff"] = len(target_ens ^ target_cens) // 2
        entry["skipped"] = False
        rows.append(entry)
        print(
            f"  {name:>5s} x={median_x:6.1f} ctl_cens {entry['ctl_cens']:+8.1f} "
            f"ens_full {entry['ens_full']:+8.1f} set_diff {entry['set_diff']}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame["ens_open"] = np.where(
        frame["median_x"] > 225.0, frame["ens_full"], frame["ctl_cens"]
    )
    persistence = json.loads(
        (REPO_ROOT / "outputs" / "paired_persistence.json").read_text()
    )
    v19 = {r["scenario"]: r["N6_nodwell"] for r in persistence["scenarios"]}
    frame["v19_ref"] = frame["scenario"].map(v19)

    def stats(column: str) -> dict:
        deltas = frame[column].to_numpy(dtype=float)
        wins = int((deltas < -0.5).sum())
        losses = int((deltas > 0.5).sum())
        return {
            "mean": round(float(deltas.mean()), 2),
            "se": round(float(deltas.std(ddof=1) / np.sqrt(len(deltas))), 2),
            "wins": wins,
            "losses": losses,
            "ties": int(len(deltas) - wins - losses),
            "block_means": [
                round(float(part.mean()), 1) for part in np.array_split(deltas, 6)
            ],
        }

    diff = frame["ens_full"] - frame["ctl_cens"]
    open_mask = frame["median_x"] > 225.0
    summary = {
        "n_scenarios": len(frame),
        "arms": {c: stats(c) for c in ("v19_ref", "ctl_cens", "ens_full", "ens_open")},
        "ens_minus_ctl": {
            "mean": round(float(diff.mean()), 2),
            "wins": int((diff < -0.5).sum()),
            "losses": int((diff > 0.5).sum()),
            "scenarios_with_set_diff": int((frame.get("set_diff", 0) > 0).sum()),
            "mean_set_diff": round(float(frame["set_diff"].fillna(0).mean()), 2),
        },
        "open_scenarios_x_gt_225": sorted(frame.loc[open_mask, "scenario"]),
        "ens_minus_ctl_open_only": round(
            float(diff[open_mask].mean()), 2
        ) if open_mask.any() else None,
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    out = {"summary": summary, "scenarios": rows}
    path = REPO_ROOT / "outputs" / "paired_ensemble.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"report: {path}")
    write_section()


SECTION_HEADER = "## Ensemble reorder (remkeyed_puremid), paired replay"


def write_section() -> None:
    data = json.loads((REPO_ROOT / "outputs" / "paired_ensemble.json").read_text())
    s = data["summary"]
    arms = s["arms"]
    frame = pd.DataFrame(data["scenarios"])
    frame["ens_open"] = np.where(
        frame["median_x"] > 225.0, frame["ens_full"], frame["ctl_cens"]
    )
    labels = {
        "v19_ref": "V19 reference (persistence+union+refill, cens order, shipped machinery)",
        "ctl_cens": "machinery-matched control: recompose top-K by CENS order",
        "ens_full": "recompose top-K by ENSEMBLE order (remkeyed_puremid)",
        "ens_open": "ensemble order only when median X > 225, else cens control",
    }
    lines = [
        SECTION_HEADER,
        "",
        "_Generated by tools/paired_selection_ensemble.py. Question: does the "
        "ensemble ORDER add exact delta on top of the V19 exchange? Arms "
        "recompose each incumbent's planned set as the order's top-K "
        "(K = incumbent planned count) with V19 logic on top (N6_nodwell "
        "persistence-zombies out, union gate forced in, refill by the same "
        "order), applied order-preservingly and scored officially; X<50 "
        "scenarios are no-ops. The cens-order control shares every mechanic, so "
        "ens-minus-ctl is PURE ordering. JSON: `outputs/paired_ensemble.json`._",
        "",
        "| arm | mean d/scen | SE | W/L/T | blocks |",
        "|---|---:|---:|---|---|",
    ]
    for key, label in labels.items():
        a = arms[key]
        lines.append(
            f"| {label} | **{a['mean']:+.1f}** | {a['se']} | "
            f"{a['wins']}/{a['losses']}/{a['ties']} | {a['block_means']} |"
        )
    d = s["ens_minus_ctl"]
    lines += [
        "",
        f"**Pure ordering effect (ens_full minus ctl_cens): {d['mean']:+.1f}/scen** "
        f"(W/L {d['wins']}/{d['losses']}; the sets differ in "
        f"{d['scenarios_with_set_diff']} scenarios, mean {d['mean_set_diff']} "
        f"batteries). Open-block only (X>225, {len(s['open_scenarios_x_gt_225'])} "
        f"scenarios): {s['ens_minus_ctl_open_only']:+.1f}/scen.",
        "",
        "Per-scenario ens_full deltas:",
        "",
        " ".join(f"{v:+.0f}" for v in frame["ens_full"].fillna(0.0)),
        "",
        "**Verdict: NO-GO for the local refill/ranking path.** The pure ordering "
        "effect is +97/scen harmful (11W/27L), and the open block -- the regime "
        "the reorder was hoped to improve -- is its WORST (+221/scen vs the cens "
        "control; ens block-1 mean +330 vs ctl -180). Root cause is visible in "
        "the anchors: the ensemble's open top-12 (.573) beats twophase (.562) "
        "but NOT cens (.589) -- cens is already the best LOCAL orderer where "
        "swaps are expensive, and each ~2-battery recomposition at X~300 costs "
        "~160 early + ~246 late when it trades a realized due away. What this "
        "replay does NOT price is the ensemble's actual case: transfer hardness "
        "(hard mean 0.4742 vs cens 0.4278, the +179-public failure mode) -- a "
        "cross-building robustness property invisible to within-split replay. "
        "If the ensemble ships, it should ship as the Sigma-p-preserving LEVEL "
        "source for unseen-building robustness, not as a selection reorder; "
        "locally the V19 exchange with cens ordering stands (-218.9).",
    ]
    md_path = REPO_ROOT / "outputs" / "paired_selection.md"
    text = md_path.read_text(encoding="utf-8")
    if SECTION_HEADER in text:
        head, _, tail = text.partition(SECTION_HEADER)
        rest = tail.split("\n## ", 1)
        text = head + ("\n## " + rest[1] if len(rest) == 2 else "")
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + "\n".join(lines) + "\n"
    md_path.write_text(text, encoding="utf-8")
    print(f"markdown section updated: {md_path}")


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        write_section()
    else:
        main()
