"""Integration fidelity for CompetitionPlanner._selection_exchange.

Replays the coordinator's integrated exchange on the SAME cached incumbents
the paired harness measured (outputs/paired_incumbents.joblib), scores it with
the official ``evaluate_plan``, and diffs it per scenario against the measured
arm 'A+B refilled, X-gated'. Because both sides run on one incumbent, every
disagreement is a RULE difference, not noise.

Variants scored per scenario (all against the cached incumbent):
  theirs      _selection_exchange exactly as integrated (production flags:
              the dip gate cannot fire at the operating config because
              row_raw_min3 is NaN when no raw variant/resurrection gate is
              active -- verified in bsai/forecaster.py).
  ordered     the SAME membership and day decisions as `theirs`, but applied
              order-preservingly (incumbent row order kept; additions take the
              exact-replay cheapest insertion slot). Isolates the final
              ``sort_values(["day", "battery"])``, which re-orders every
              worked day's route alphabetically.
  sortonly    the incumbent with no membership change, just re-sorted the way
              _selection_exchange returns plans. Prices route destruction
              alone.

    python tools/paired_selection_fidelity.py
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

from batteryswap_solution.costs import CostTables, select_candidates
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

_spec = importlib.util.spec_from_file_location(
    "paired_selection", REPO_ROOT / "tools" / "paired_selection.py"
)
paired_selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired_selection)
PlanEditor = paired_selection.PlanEditor
gate_table = paired_selection.gate_table
removal_order = paired_selection.removal_order


class StubForecast:
    """_selection_exchange only reads .summaries."""

    def __init__(self, summaries: pd.DataFrame) -> None:
        self.summaries = summaries


def score_total(plan, locs, travel, settings, not_dead, start) -> float:
    _, _, scores = evaluate_plan(
        plan, locs, travel, settings,
        eol_times=not_dead, start_time=start, verbose=0,
    )
    return float(scores["total_cost"])


def main() -> None:
    cache = joblib.load(REPO_ROOT / "outputs" / "paired_incumbents.joblib")
    report = json.loads((REPO_ROOT / "outputs" / "paired_selection.json").read_text())
    my_by_scenario = {
        entry["scenario"]: (entry["arm_ab_refill"] or {"total": 0.0})["total"]
        for entry in report["scenarios"]
    }
    gates_frame = gate_table(pd.read_parquet(REPO_ROOT / "outputs" / "research_rowfeat.parquet"))

    locations, timeseries, eol_times, scenarios = load_dataset(REPO_ROOT / "dataset" / "train")
    rows: list[dict] = []
    started = time.time()

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        name = scenario["name"]
        rec = cache["records"].get(name)
        if rec is None:
            continue
        start = pd.Timestamp(scenario["start_time"]).normalize()
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
        dates = rec["candidate_dates"]
        defer_day = rec["defer_day"]
        ids = list(rec["battery_ids"])
        p = np.asarray(rec["p"], dtype=float)
        margin = np.asarray(rec["margin"], dtype=float)
        staleness = np.asarray(rec["staleness"], dtype=float)
        demote = np.asarray(rec["slot_demote"], dtype=bool)

        full_costs = CostTables(
            battery_ids=tuple(ids),
            candidate_dates=dates,
            service_cost=np.asarray(rec["service_cost"], dtype=float),
            defer_cost=np.asarray(rec["defer_cost"], dtype=float),
            event_pmf=np.asarray(rec["event_pmf"], dtype=float),
            horizon_event_probability=p,
        )

        # ------- flags exactly as the integrated forecaster produces them ----
        with np.errstate(invalid="ignore"):
            dark_runtime = (staleness > 30.0) & ((margin - 0.001 * staleness) < 0.02)
        dark_runtime &= np.isfinite(margin)
        # dip: raw_min3 is NaN at the operating config (no raw variant, no
        # resurrection gate), so isfinite() keeps it permanently False.
        gate_include_prod = dark_runtime.copy()
        summaries_prod = pd.DataFrame(
            {
                "battery_id": ids,
                "horizon_probability": p,
                "slot_demote": demote,
                "gate_include": gate_include_prod,
            }
        )

        # cap exactly as plan() has mutated it by the exchange call:
        # min(15, ceil(1.6 * sum(p over the FULL fleet) + 1)).
        full_limit = min(15, int(np.ceil(1.6 * float(p.sum()) + 1.0)))
        planner = CompetitionPlanner(
            config=PlannerConfig(
                optimizer=OptimizationConfig(max_planned_count=full_limit)
            )
        )

        incumbent = rec["plan"].copy()
        base = score_total(incumbent, locs, travel, settings, not_dead, start)

        candidate_ids = {
            full_costs.battery_ids[i] for i in select_candidates(full_costs)
        }
        theirs_plan = planner._selection_exchange(
            incumbent.copy(), full_costs, StubForecast(summaries_prod),
            dates, defer_day, locs, candidate_ids, full_limit,
        )
        theirs = score_total(theirs_plan, locs, travel, settings, not_dead, start)

        # ------- diagnostics: what did each side decide? ---------------------
        editor = PlanEditor(rec, locs, travel, settings, start)
        planned_before = set(editor.planned)
        planned_after_theirs = set(
            theirs_plan.loc[pd.to_datetime(theirs_plan["day"]) <= horizon_end, "battery"].astype(str)
        )
        their_removed = planned_before - planned_after_theirs
        their_added = planned_after_theirs - planned_before
        their_day_of = {
            str(row.battery): pd.Timestamp(row.day).normalize()
            for row in theirs_plan.itertuples(index=False)
            if str(row.battery) in their_added
        }

        end_times = pd.to_datetime(locs["end_time"])
        if getattr(end_times.dt, "tz", None) is not None:
            end_times = end_times.dt.tz_localize(None)
        median_x = float(
            (end_times.dt.normalize() + pd.Timedelta(days=30.0) - dates[-1]).dt.days.median()
        )
        their_skip = median_x < 100.0
        my_skip = index >= 40

        # `ordered`: their membership+days, applied order-preservingly.
        if their_removed or their_added:
            additions = sorted((b, their_day_of[b]) for b in their_added)
            ordered_plan = editor.variant(sorted(their_removed), additions)
            ordered = score_total(ordered_plan, locs, travel, settings, not_dead, start)
        else:
            ordered = base
        # `sortonly`: incumbent re-sorted the way _selection_exchange returns.
        resorted = (
            incumbent.assign(battery=incumbent["battery"].astype(str))
            .sort_values(["day", "battery"], kind="stable")
            .reset_index(drop=True)
        )
        sortonly = score_total(resorted, locs, travel, settings, not_dead, start)

        # ------- my measured arm's decisions (recomputed, deterministic) -----
        demote_of = dict(zip(ids, demote))
        p_of = {b: float(p[editor.position_of[b]]) for b in ids}
        limit_mine = rec["limit"] if rec["limit"] is not None else 15
        grows = gates_frame[gates_frame["scenario"] == index]
        my_gates = []
        for row in grows[grows["gate_dark"] | grows["gate_dip"]].itertuples(index=False):
            if row.battery in editor.position_of and row.battery not in planned_before:
                my_gates.append((row.battery, "dark" if row.gate_dark else "dip"))
        my_gates.sort()
        zombies = [b for b in editor.planned if demote_of.get(b, False)]
        add_specs = [
            (b, *editor.injection_day(b, editor.day_rows)) for b, _ in my_gates
        ]
        n_after = len(editor.planned) - len(zombies) + len(add_specs)
        extra = removal_order(editor, exclude=set(zombies))[: max(n_after - limit_mine, 0)]
        my_removed = set(zombies) | set(extra)
        gate_ids = {b for b, _, _ in add_specs}
        pool = sorted(
            (
                b
                for b in rec["candidate_ids"]
                if b not in planned_before and not demote_of.get(b, False)
                and b not in gate_ids
            ),
            key=lambda b: (-p_of[b], b),
        )
        slots_free = limit_mine - (len(editor.planned) - len(my_removed) + len(add_specs))
        refill = pool[: max(slots_free, 0)]
        my_added = gate_ids | set(refill)
        my_day_of = {b: d for b, d, _ in add_specs}
        my_day_of.update({b: editor.injection_day(b, editor.day_rows)[0] for b in refill})
        if my_skip:
            my_removed, my_added, my_day_of = set(), set(), {}

        common_added = (their_added & my_added) if not their_skip else set()
        day_mismatch = sorted(
            b for b in common_added
            if pd.Timestamp(my_day_of[b]).normalize() != their_day_of[b]
        )
        # days whose membership is untouched but whose row order changed
        reordered_days = 0
        for day, members in editor.day_rows.items():
            if any(b in their_removed or b in their_added for b in members):
                continue
            after = theirs_plan.loc[
                pd.to_datetime(theirs_plan["day"]).dt.normalize() == day, "battery"
            ].astype(str).tolist()
            if after and after != members:
                reordered_days += 1

        entry = {
            "scenario": name,
            "index": index,
            "median_x": round(median_x, 1),
            "their_skip": bool(their_skip),
            "my_skip": bool(my_skip),
            "base": round(base, 2),
            "delta_theirs": round(theirs - base, 2),
            "delta_ordered": round(ordered - base, 2),
            "delta_sortonly": round(sortonly - base, 2),
            "delta_mine_xgated": 0.0 if my_skip else round(my_by_scenario.get(name, 0.0), 2),
            "their_removed": sorted(their_removed),
            "their_added": sorted(their_added),
            "my_removed": sorted(my_removed),
            "my_added": sorted(my_added),
            "added_only_mine": sorted(my_added - their_added) if not (their_skip or my_skip) else [],
            "added_only_theirs": sorted(their_added - my_added) if not (their_skip or my_skip) else [],
            "removed_only_mine": sorted(my_removed - their_removed) if not (their_skip or my_skip) else [],
            "day_mismatch_common_adds": day_mismatch,
            "reordered_untouched_days": reordered_days,
        }
        rows.append(entry)
        print(
            f"  {name:>5s} x={median_x:6.1f} theirs {entry['delta_theirs']:+8.1f} "
            f"ordered {entry['delta_ordered']:+8.1f} sortonly {entry['delta_sortonly']:+8.1f} "
            f"mine {entry['delta_mine_xgated']:+8.1f} reordered_days {reordered_days}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    summary = {
        "n_scenarios": len(frame),
        "mean_delta_theirs_as_integrated": round(float(frame["delta_theirs"].mean()), 2),
        "mean_delta_their_sets_order_preserved": round(float(frame["delta_ordered"].mean()), 2),
        "mean_delta_sortonly": round(float(frame["delta_sortonly"].mean()), 2),
        "mean_delta_mine_xgated": round(float(frame["delta_mine_xgated"].mean()), 2),
        "their_skipped_scenarios": sorted(frame.loc[frame["their_skip"], "scenario"]),
        "my_skipped_scenarios": sorted(frame.loc[frame["my_skip"], "scenario"]),
        "scenarios_with_day_mismatch": int((frame["day_mismatch_common_adds"].str.len() > 0).sum()),
        "mean_reordered_untouched_days": round(float(frame["reordered_untouched_days"].mean()), 2),
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    out = {"summary": summary, "scenarios": rows}
    path = REPO_ROOT / "outputs" / "paired_fidelity.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"report: {path}")
    write_section()


SECTION_HEADER = "## Integration fidelity (_selection_exchange, paired replay)"


def write_section() -> None:
    """Patch the fidelity section into outputs/paired_selection.md."""
    data = json.loads((REPO_ROOT / "outputs" / "paired_fidelity.json").read_text())
    s = data["summary"]
    lines = [
        SECTION_HEADER,
        "",
        "_Generated by tools/paired_selection_fidelity.py: the integrated "
        "`CompetitionPlanner._selection_exchange` replayed on the SAME cached "
        "incumbents and scored with the official evaluator -- every gap below is a "
        "rule difference, not noise. JSON: `outputs/paired_fidelity.json`._",
        "",
        "| variant | mean d/scen (48) |",
        "|---|---:|",
        f"| integrated `_selection_exchange` as written | **{s['mean_delta_theirs_as_integrated']:+.1f}** |",
        f"| same membership+days, order-preserved application | {s['mean_delta_their_sets_order_preserved']:+.1f} |",
        f"| incumbent merely re-sorted by (day, battery) | {s['mean_delta_sortonly']:+.1f} |",
        f"| measured arm (A+B refilled, X-gated s_40-47) | {s['mean_delta_mine_xgated']:+.1f} |",
        "",
        "**Verdict: NOT faithful.** The integration inverts the measured arm "
        "(+16.6 vs -119.6). Named rule differences, largest first:",
        "",
        "1. **Route destruction (integration bug).** `_selection_exchange` returns "
        "`sort_values([\"day\", \"battery\"])`, re-ordering EVERY worked day's route "
        "alphabetically (the incumbent order is the local search's routed order; "
        f"the evaluator prices row order). Re-sorting the incumbent alone costs "
        f"{s['mean_delta_sortonly']:+.1f}/scen; mean "
        f"{s['mean_reordered_untouched_days']} untouched-membership days per "
        "scenario come back re-ordered. Fix: keep incoming row order, insert "
        "additions into the day's sequence (cheapest-insertion), drop removals in "
        "place.",
        "2. **Dark-gate staleness is clamped to the grid end (flag-source bug).** "
        "`forecaster.predict` (and the summaries the exchange reads) computes "
        "staleness after `index = min(index, len(series)-1)`: a device whose "
        "smoothed grid ENDED months before the cutoff (the cold-room dark channel "
        "-- the gate's entire target) reads staleness ~0 and never fires. Verified: "
        "d_124f1e85339f at s_23 has grid overhang 95 d, runtime staleness 0, true "
        "staleness 95; runtime-vs-measured dark sets disagree on 36 of 60 union "
        "rows, and d_124f1e85339f (the measured arm's top earner, 11 injections at "
        "-80 each) is missed in every scenario. Fix: staleness += "
        "max(origin_index - (len(series)-1), 0) before the clamp.",
        "3. **The dip gate is dead at the operating config.** `row_raw_min3` is NaN "
        "unless a raw feature variant or a resurrection gate is active "
        "(`raw_cache.update` is gated), so `isfinite(raw3)` never passes. Measured "
        "dip-only was +1.9 (neutral) -- either wire `raw_cache.update` when "
        "`selection_exchange` is on, or delete the dip branch deliberately.",
        "4. **No displacement removals.** The integrated gate loop `break`s at the "
        "cap; the measured arm removed extra lowest-p planned batteries to make "
        "room (21 extra removals over the 32 both-active scenarios).",
        "5. **Refill placement + pool.** Integrated refills go to the cost-optimal "
        "day from the FULL fleet with a p>0.05 floor; the measured arm placed "
        "nearest-visit-first from the candidate set (measured placement split: "
        "visit -92.7 vs cost-optimal +37.6 per swap-in); 20 common additions land "
        "on different days (gate anchors are also computed post-zombie-removal "
        "instead of on the incumbent).",
        "6. **Cap basis.** `config.optimizer.max_planned_count` at the exchange "
        "call is the mutated FULL-fleet budget; the measured limit was the "
        "filtered-candidate budget (+-1 slot in budget-bound scenarios).",
        "7. **X-gate set.** median X<100 skips s_32-47 (16 scenarios) vs the "
        "measured projection's s_40-47 (8). The extra 8 skips forfeit only "
        "-8.3/scen net (s_33-35 losses roughly offset s_32/s_36 wins) -- "
        "defensible, but it is not the measured gate.",
        "",
        "**Corrected expectation for the integrated pass:** as written "
        f"{s['mean_delta_theirs_as_integrated']:+.1f}/scen (ships a regression); "
        "with the re-sort fixed (order-preserving application of the same "
        f"decisions) {s['mean_delta_their_sets_order_preserved']:+.1f}/scen; "
        "recovering the measured -120/scen additionally requires the staleness "
        "overhang fix (2) plus displacement (4) and visit-first refill placement "
        "(5).",
    ]
    md_path = REPO_ROOT / "outputs" / "paired_selection.md"
    text = md_path.read_text(encoding="utf-8")
    if SECTION_HEADER in text:
        head, _, tail = text.partition(SECTION_HEADER)
        # drop the old section up to the next H2 or EOF
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
