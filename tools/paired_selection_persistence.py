"""Persistence-keyed zombie rule, priced replay-exact on the paired incumbents.

V16 re-audit #1 unblock: today's fingerprint (margin<0.05 & dwell>42 & p>0.4)
catches 1 of 5 documented floor-zombies and SWEEPS in-slot dues
(VISIBLE-DROPPED +101/scen: d_ccd3a65228a9 x6, d_4643496a7525 x5, ...). The
persistence rule demotes only on repeated evidence:

    flag(b, s)   = margin < 0.05 & p > 0.4 [& dwell > 42 in the dwell variants]
    demote(b, s) = flag(b, s) AND #{s' < s : flag(b, s')} >= N

computed from the forecaster's OWN per-cutoff quantities (the cached records
carry the production margin/dwell/p arrays for every scenario), so the rule is
causal by construction: scenario s sees flags from cutoffs strictly before s.
A battery with a recorded death before cutoff s is not in the fleet at s
(iterate_scenarios drops EOL'd devices), so "zero recorded deaths" is
automatic for every alive battery; first-flag batteries have prior count 0 and
are exempt. Production wiring: the forecaster processes scenarios
sequentially, so predict() can carry {device: prior_flag_count} across calls,
incrementing AFTER the current flags are emitted.

Every arm is the SHIPPED _selection_exchange (v2) on the cached incumbents
with gate_include = the A2 union winner (dark1 | dark2_last_0.00) and
slot_demote swapped per arm -- each arm IS the full combined pass
(persistence-zombies out + union-gate in + refill), delta vs the incumbent.

    python tools/paired_selection_persistence.py     # ~5 min
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

from batteryswap_solution.costs import CostTables
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

_spec = importlib.util.spec_from_file_location(
    "paired_selection_a2", REPO_ROOT / "tools" / "paired_selection_a2.py"
)
a2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2)

_EPOCH = pd.Timestamp("1970-01-01")
NS = (3, 6, 10)
FLOOR5 = [
    "d_b5b678a3f79f",
    "d_3d26e12378f1",
    "d_c9a2ce794b68",
    "d_d4b4272d5229",
    "d_d9d695df1683",
]


def main() -> None:
    cache = joblib.load(REPO_ROOT / "outputs" / "paired_incumbents.joblib")
    records = [cache["records"][f"s_{i}"] for i in range(len(cache["records"]))]

    # ---- causal flag histories from the forecaster's own cached quantities --
    flag_by_scen: dict[bool, list[dict[str, bool]]] = {True: [], False: []}
    for rec in records:
        ids = list(rec["battery_ids"])
        margin = np.asarray(rec["margin"], dtype=float)
        dwell = np.asarray(rec["dwell"], dtype=float)
        p = np.asarray(rec["p"], dtype=float)
        with np.errstate(invalid="ignore"):
            base_flag = (margin < 0.05) & (p > 0.4)
            flag_by_scen[False].append(
                {b: bool(f) for b, f in zip(ids, base_flag)}
            )
            flag_by_scen[True].append(
                {b: bool(f) for b, f in zip(ids, base_flag & (dwell > 42.0))}
            )

    def demote_vector(scenario_index: int, ids: list[str], n: int, use_dwell: bool):
        flags = flag_by_scen[use_dwell]
        prior: dict[str, int] = {}
        for s in range(scenario_index):
            for b, f in flags[s].items():
                if f:
                    prior[b] = prior.get(b, 0) + 1
        now = flags[scenario_index]
        return np.array(
            [bool(now.get(b, False)) and prior.get(b, 0) >= n for b in ids],
            dtype=bool,
        )

    smooth, rawany = a2.build_channels()
    locations, timeseries, eol_times, scenarios = load_dataset(
        REPO_ROOT / "dataset" / "train"
    )

    arms = ["fp_today", "none"] + [
        f"N{n}_{'dwell' if d else 'nodwell'}" for n in NS for d in (True, False)
    ]
    rows: list[dict] = []
    diag_rows: list[dict] = []
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
        stale_true = np.full(len(ids), np.nan)
        any_last = np.full(len(ids), np.nan)
        for j, b in enumerate(ids):
            _, stale_true[j] = a2.true_staleness(smooth, b, origin)
            any_last[j], _ = a2.rawany_margins(rawany, b, origin)
        with np.errstate(invalid="ignore"):
            dark1 = (sc > 30.0) & ((mc - 0.001 * sc) < 0.02) & (remaining >= 30.0)
            dark2 = (stale_true > 30.0) & (any_last < 0.00) & (remaining >= 30.0)
        union_gate = dark1 | dark2

        full_costs = CostTables(
            battery_ids=tuple(ids),
            candidate_dates=rec["candidate_dates"],
            service_cost=np.asarray(rec["service_cost"], dtype=float),
            defer_cost=np.asarray(rec["defer_cost"], dtype=float),
            event_pmf=np.asarray(rec["event_pmf"], dtype=float),
            horizon_event_probability=p,
        )
        full_limit = min(15, int(np.ceil(1.6 * float(p.sum()) + 1.0)))
        planner = CompetitionPlanner(
            config=PlannerConfig(
                optimizer=OptimizationConfig(max_planned_count=full_limit)
            )
        )
        incumbent = rec["plan"].copy()
        base = float(
            evaluate_plan(
                incumbent, locs, travel, settings,
                eol_times=not_dead, start_time=start, verbose=0,
            )[2]["total_cost"]
        )
        due = set(
            not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index.astype(str)
        )
        planned = set(
            incumbent.loc[
                pd.to_datetime(incumbent["day"]) <= horizon_end, "battery"
            ].astype(str)
        )

        entry = {"scenario": name, "index": index, "base": round(base, 2)}
        for arm in arms:
            if arm == "fp_today":
                demote = np.asarray(rec["slot_demote"], dtype=bool)
            elif arm == "none":
                demote = np.zeros(len(ids), dtype=bool)
            else:
                n = int(arm[1 : arm.index("_")])
                demote = demote_vector(index, ids, n, arm.endswith("_dwell"))
            summaries = pd.DataFrame(
                {"battery_id": ids, "slot_demote": demote, "gate_include": union_gate}
            )
            plan = planner._selection_exchange(
                incumbent.copy(), full_costs, a2.StubForecast(summaries),
                rec["candidate_dates"], rec["defer_day"], locs,
                set(rec["candidate_ids"]), full_limit,
            )
            total = float(
                evaluate_plan(
                    plan, locs, travel, settings,
                    eol_times=not_dead, start_time=start, verbose=0,
                )[2]["total_cost"]
            )
            entry[arm] = round(total - base, 2)
            demoted_planned = [ids[k] for k in np.flatnonzero(demote) if ids[k] in planned]
            diag_rows.append(
                {
                    "scenario": name,
                    "arm": arm,
                    "demoted_planned": len(demoted_planned),
                    "demoted_due": sum(1 for b in demoted_planned if b in due),
                    "floor5_demoted": sum(1 for b in demoted_planned if b in FLOOR5),
                }
            )
        rows.append(entry)
        print(
            f"  {name:>5s} base {base:8.1f} "
            + " ".join(f"{arm} {entry[arm]:+7.1f}" for arm in arms),
            flush=True,
        )

    frame = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)
    summary: dict = {
        "n_scenarios": len(frame),
        "arms": {},
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    for arm in arms:
        deltas = frame[arm].to_numpy(dtype=float)
        d = diag[diag["arm"] == arm]
        wins = int((deltas < -0.5).sum())
        losses = int((deltas > 0.5).sum())
        summary["arms"][arm] = {
            "mean": round(float(deltas.mean()), 2),
            "se": round(float(deltas.std(ddof=1) / np.sqrt(len(deltas))), 2),
            "wins": wins,
            "losses": losses,
            "ties": int(len(deltas) - wins - losses),
            "block_means": [
                round(float(part.mean()), 1) for part in np.array_split(deltas, 6)
            ],
            "demotions_per_scen": round(float(d["demoted_planned"].mean()), 2),
            "due_sweeps_total": int(d["demoted_due"].sum()),
            "floor5_demotions_total": int(d["floor5_demoted"].sum()),
        }
    out = {"summary": summary, "scenarios": rows, "diagnostics": diag_rows}
    path = REPO_ROOT / "outputs" / "paired_persistence.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"report: {path}")
    write_section()


SECTION_HEADER = "## Persistence-keyed zombie rule (paired replay, V16 unblock #1)"


def write_section() -> None:
    data = json.loads((REPO_ROOT / "outputs" / "paired_persistence.json").read_text())
    s = data["summary"]
    arms = s["arms"]
    frame = pd.DataFrame(data["scenarios"])
    best = min(
        (a for a in arms if a not in ("fp_today", "none")),
        key=lambda a: arms[a]["mean"],
    )
    best_deltas = frame[best].to_numpy(dtype=float)
    lines = [
        SECTION_HEADER,
        "",
        "_Generated by tools/paired_selection_persistence.py. Every arm is the "
        "full combined pass -- persistence-zombies out + union gate "
        "(dark1|dark2_last_0.00) in + refill -- through the SHIPPED "
        "`_selection_exchange` on the cached incumbents; deltas are exact paired "
        "differences vs the incumbent. flag = margin<0.05 & p>0.4 "
        "(& dwell>42 in the dwell variants); demote at scenario s iff flagged "
        "NOW and flagged in >= N cutoffs STRICTLY BEFORE s (causal; alive "
        "batteries have zero recorded deaths by construction, first flags are "
        "exempt). JSON: `outputs/paired_persistence.json`._",
        "",
        "| arm | mean d/scen | SE | W/L/T | demotions/scen | due sweeps (48 scen) | floor-5 demotions | blocks |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    order = ["fp_today", "none"] + [
        f"N{n}_{d}" for n in NS for d in ("dwell", "nodwell")
    ]
    for arm in order:
        a = arms[arm]
        label = {
            "fp_today": "shipped fingerprint (dwell>42, no persistence)",
            "none": "no demotion (gate+refill only)",
        }.get(arm, arm)
        lines.append(
            f"| {label} | **{a['mean']:+.1f}** | {a['se']} | "
            f"{a['wins']}/{a['losses']}/{a['ties']} | {a['demotions_per_scen']} "
            f"| {a['due_sweeps_total']} | {a['floor5_demotions_total']} "
            f"| {a['block_means']} |"
        )
    n_best = best[1 : best.index("_")]
    dwell_txt = "with dwell>42" if best.endswith("_dwell") else "WITHOUT dwell"
    lines += [
        "",
        f"Winner: **{best}** ({arms[best]['mean']:+.1f}/scen, "
        f"{arms[best]['due_sweeps_total']} due sweeps vs "
        f"{arms['fp_today']['due_sweeps_total']} for the shipped fingerprint). "
        "Per-scenario paired deltas:",
        "",
        " ".join(f"{v:+.0f}" for v in best_deltas),
        "",
        "**Exact causal spec for the integrator** (replaces the `slot_demote` "
        "fingerprint in bsai/forecaster.py):",
        "",
        "```",
        "flag_now = (margin < 0.05) & (p42 > 0.4)"
        + (" & (dwell > 42)" if best.endswith("_dwell") else "")
        + f"   # {dwell_txt}",
        f"slot_demote = flag_now & (prior_flag_count >= {n_best})",
        "# prior_flag_count: per-device count of PRIOR predict() calls whose",
        "# flag_now was True. The forecaster is stateful and processes scenarios",
        "# sequentially -- keep `self.flag_history: dict[str, int]` and increment",
        "# AFTER computing slot_demote (history = prior cutoffs only, never the",
        "# current one). Batteries with a recorded death never reach predict()",
        "# again (iterate/harness drops EOL'd devices), so 'zero deaths' is",
        "# automatic; a fresh split starts at zero counts -> no demotions in the",
        "# first N scenarios (graceful, matches this measurement).",
        "```",
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
