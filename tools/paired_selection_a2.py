"""A2: any-temperature dark gate, re-priced on the paired incumbents.

The shipped dark gate extrapolates through the smoothing blackout
(mext = margin - 0.001 x staleness) and, since the fidelity round, runs on
CLAMPED staleness (in-grid gaps only). Roadblock L2(vii): 85% of dark dues are
raw-FRESH in the any-temperature channel the 10-30 degC filter discards --
bsai/v12_rawany.py (RawAnyCache) reads that channel. A2 replaces the
extrapolation with an observation:

    dark2 = (true smoothed staleness > 30)          # official channel is dark
          & (any-temp margin < tau)                 # observed fresh voltage low
          & (remaining observation >= 30)
    tau in {0.00, 0.01, 0.02}, margin from raw_any_{min3,last} - 2.4,
    reading taken UNCLAMPED in a 14-day window ending at cutoff-1 (a device
    whose any-temp channel also ended is NaN -> cannot fire; this is what
    prevents the long-ended-device flood the overhang-corrected mext variant
    hit).

Every arm below is the SHIPPED CompetitionPlanner._selection_exchange (v2:
order-preserving, displacement at the cap, visit-first placement, candidate
refill pool with p>0.05, X<50 regime gate) applied to the cached paired
incumbents with only the gate_include flags swapped -- so the table is exactly
"what ships today" vs "ship with dark2", replay-exact, no reroll noise.

    python tools/paired_selection_a2.py            # ~6 min (hourly parquet load)
"""

from __future__ import annotations

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

from bsai.smoothing import SmoothingCache
from bsai.v12_rawany import RawAnyCache

_EPOCH = pd.Timestamp("1970-01-01")
TAUS = (0.00, 0.01, 0.02)
AXES = ("min3", "last")


class StubForecast:
    def __init__(self, summaries: pd.DataFrame) -> None:
        self.summaries = summaries


def build_channels() -> tuple[SmoothingCache, RawAnyCache]:
    print("loading hourly parquet ...", flush=True)
    raw = pd.read_parquet(
        REPO_ROOT / "dataset" / "train" / "battery_metrics.parquet",
        engine="fastparquet",
    )
    print(f"  {len(raw)} rows; building caches ...", flush=True)
    smooth = SmoothingCache()
    smooth.update(raw)
    rawany = RawAnyCache()
    rawany.update(raw)
    return smooth, rawany


def true_staleness(smooth: SmoothingCache, battery: str, origin_ordinal: int):
    """(margin, staleness) at the cutoff, WITHOUT the grid-end clamp.

    Smoothing is causal, so the full-data grid truncated at the cutoff equals
    the as-of-cutoff grid; staleness is measured from the cutoff itself, not
    from wherever the device's grid happens to end.
    """
    series = smooth.devices.get(battery)
    if series is None:
        return np.nan, np.nan
    index = int(origin_ordinal) - series.origin
    if index < 0:
        return np.nan, np.nan
    upto = min(index, len(series) - 1)
    values = series.smooth_voltage[: upto + 1]
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return np.nan, np.nan
    return float(values[valid[-1]]) - 2.4, float(index - valid[-1])


def rawany_margins(rawany: RawAnyCache, battery: str, cutoff_ordinal: int):
    """(last, min3) any-temp margins from an UNCLAMPED 14-day window ending at
    cutoff-1. NaN when the channel has no reading in that window (the device
    is not merely cold-filtered but gone)."""
    series = rawany.devices.get(battery)
    if series is None:
        return np.nan, np.nan
    index = int(cutoff_ordinal) - 1 - series.origin
    if index < 0:
        return np.nan, np.nan
    lo = max(0, index - 13)
    hi = min(index, len(series) - 1)
    if hi < lo:
        return np.nan, np.nan
    window = series.median[lo : hi + 1]
    valid = window[~np.isnan(window)]
    if valid.size == 0:
        return np.nan, np.nan
    return float(valid[-1]) - 2.4, float(np.min(valid[-3:])) - 2.4


def main() -> None:
    cache = joblib.load(REPO_ROOT / "outputs" / "paired_incumbents.joblib")
    smooth, rawany = build_channels()
    locations, timeseries, eol_times, scenarios = load_dataset(
        REPO_ROOT / "dataset" / "train"
    )

    variants = [("today", None, None)] + [
        (f"dark2_{axis}_{tau:.2f}", axis, tau) for axis in AXES for tau in TAUS
    ]
    rows: list[dict] = []
    flag_rows: list[dict] = []
    started = time.time()

    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        name = scenario["name"]
        rec = cache["records"].get(name)
        if rec is None:
            continue
        start = pd.Timestamp(scenario["start_time"]).normalize()
        origin_ordinal = int((start - _EPOCH) / pd.Timedelta(days=1))
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
        dates = rec["candidate_dates"]
        defer_day = rec["defer_day"]
        ids = list(rec["battery_ids"])
        p = np.asarray(rec["p"], dtype=float)
        margin_clamped = np.asarray(rec["margin"], dtype=float)
        stale_clamped = np.asarray(rec["staleness"], dtype=float)
        demote = np.asarray(rec["slot_demote"], dtype=bool)

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
        any_min3 = np.full(len(ids), np.nan)
        for j, b in enumerate(ids):
            _, stale_true[j] = true_staleness(smooth, b, origin_ordinal)
            any_last[j], any_min3[j] = rawany_margins(rawany, b, origin_ordinal)

        with np.errstate(invalid="ignore"):
            dark_today = (
                (stale_clamped > 30.0)
                & ((margin_clamped - 0.001 * stale_clamped) < 0.02)
                & (remaining >= 30.0)
            )
            dark2 = {
                (axis, tau): (
                    (stale_true > 30.0)
                    & ((any_min3 if axis == "min3" else any_last) < tau)
                    & (remaining >= 30.0)
                )
                for axis in AXES
                for tau in TAUS
            }

        full_costs = CostTables(
            battery_ids=tuple(ids),
            candidate_dates=dates,
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
        for label, axis, tau in variants:
            gate = dark_today if label == "today" else dark2[(axis, tau)]
            summaries = pd.DataFrame(
                {
                    "battery_id": ids,
                    "slot_demote": demote,
                    "gate_include": gate,
                }
            )
            variant_plan = planner._selection_exchange(
                incumbent.copy(),
                full_costs,
                StubForecast(summaries),
                dates,
                defer_day,
                locs,
                set(rec["candidate_ids"]),
                full_limit,
            )
            total = float(
                evaluate_plan(
                    variant_plan, locs, travel, settings,
                    eol_times=not_dead, start_time=start, verbose=0,
                )[2]["total_cost"]
            )
            entry[label] = round(total - base, 2)
            flagged = [ids[k] for k in np.flatnonzero(gate)]
            unplanned = [b for b in flagged if b not in planned]
            flag_rows.append(
                {
                    "scenario": name,
                    "variant": label,
                    "flags": len(flagged),
                    "unplanned_flags": len(unplanned),
                    "unplanned_due": sum(1 for b in unplanned if b in due),
                    "overlap_today_dark": len(
                        set(flagged) & {ids[k] for k in np.flatnonzero(dark_today)}
                    ),
                }
            )
        rows.append(entry)
        print(
            f"  {name:>5s} base {base:8.1f} today {entry['today']:+8.1f} "
            + " ".join(
                f"{label.split('dark2_')[1]} {entry[label]:+7.1f}"
                for label, _, _ in variants[1:]
            ),
            flush=True,
        )

    frame = pd.DataFrame(rows)
    flags = pd.DataFrame(flag_rows)
    summary: dict = {
        "n_scenarios": len(frame),
        "arms": {},
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    for label, _, _ in variants:
        deltas = frame[label].to_numpy(dtype=float)
        fsub = flags[flags["variant"] == label]
        wins = int((deltas < -0.5).sum())
        losses = int((deltas > 0.5).sum())
        summary["arms"][label] = {
            "mean": round(float(deltas.mean()), 2),
            "se": round(float(deltas.std(ddof=1) / np.sqrt(len(deltas))), 2),
            "wins": wins,
            "losses": losses,
            "ties": int(len(deltas) - wins - losses),
            "block_means": [
                round(float(part.mean()), 1)
                for part in np.array_split(deltas, 6)
            ],
            "flags_per_scen": round(float(fsub["flags"].mean()), 2),
            "unplanned_flags_per_scen": round(
                float(fsub["unplanned_flags"].mean()), 2
            ),
            "unplanned_due_rate": round(
                float(fsub["unplanned_due"].sum())
                / max(int(fsub["unplanned_flags"].sum()), 1),
                3,
            ),
        }
    out = {"summary": summary, "scenarios": rows, "flags": flag_rows}
    path = REPO_ROOT / "outputs" / "paired_a2.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"report: {path}")
    write_section()


SECTION_HEADER = "## A2: any-temperature dark gate (paired replay on the shipped exchange)"


def write_section() -> None:
    data = json.loads((REPO_ROOT / "outputs" / "paired_a2.json").read_text())
    s = data["summary"]
    arms = s["arms"]
    best_label = min(
        (label for label in arms if label != "today"),
        key=lambda label: arms[label]["mean"],
    )
    frame = pd.DataFrame(data["scenarios"])
    best_deltas = frame[best_label].to_numpy(dtype=float)
    lines = [
        SECTION_HEADER,
        "",
        "_Generated by tools/paired_selection_a2.py. Every arm is the SHIPPED "
        "`_selection_exchange` (v2: order-preserving, displacement, visit-first, "
        "X<50) on the cached incumbents with only `gate_include` swapped; deltas "
        "are exact paired differences vs the incumbent. dark2 reads the device's "
        "actual any-temperature voltage (bsai/v12_rawany.py) in an UNCLAMPED "
        "14-day window ending at cutoff-1 instead of extrapolating mext through "
        "the blackout; `true staleness` is measured from the cutoff, not the "
        "grid end._",
        "",
        "| arm | mean d/scen | SE | W/L/T | flags/scen | unplanned due rate | blocks |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for label in ["today"] + sorted(l for l in arms if l != "today"):
        a = arms[label]
        lines.append(
            f"| {label} | **{a['mean']:+.1f}** | {a['se']} | "
            f"{a['wins']}/{a['losses']}/{a['ties']} | {a['unplanned_flags_per_scen']} "
            f"| {a['unplanned_due_rate']} | {a['block_means']} |"
        )
    today_deltas = frame["today"].to_numpy(dtype=float)
    diff = best_deltas - today_deltas
    diff_wins = int((diff < -0.5).sum())
    diff_losses = int((diff > 0.5).sum())
    lines += [
        "",
        f"Best single axis: **{best_label}** ({arms[best_label]['mean']:+.1f}/scen); "
        f"vs today it is heterogeneous (isolated diff {diff.mean():+.1f}/scen, "
        f"W/L {diff_wins}/{diff_losses}): the clamped mext gate and dark2 catch "
        "DIFFERENT populations -- in-grid gaps on live series (today) vs fully "
        "dark grids with a fresh any-temp reading (dark2). They are complements, "
        "not substitutes.",
    ]
    union_path = REPO_ROOT / "outputs" / "paired_a2_union.json"
    if union_path.exists():
        u = json.loads(union_path.read_text())
        lines += [
            "",
            f"**UNION (today's dark | dark2_last_0.00): {u['mean']:+.1f}/scen** "
            f"(SE {u['se']}, W/L/T {u['wins']}/{u['losses']}/{u['ties']}, blocks "
            f"{u['blocks']}, {u['unpl_per_scen']} unplanned flags/scen at due rate "
            f"{u['due_rate']}) -- a {u['mean'] - arms['today']['mean']:+.1f}/scen "
            "upgrade over the shipping flags. Per-scenario deltas:",
            "",
            " ".join(f"{v:+.0f}" for v in u["per_scenario"].values()),
        ]
    lines += [
        "",
        "**Exact flag spec for the integrator** (the dark branch of "
        "`gate_include` in bsai/forecaster.py becomes the union):",
        "",
        "```",
        "# branch 1 -- keep as shipped (clamped staleness, in-grid gaps):",
        "dark1 = (stale_clamped > 30) & (margin - 0.001*stale_clamped < 0.02) \\",
        "        & (remaining >= 30)",
        "# branch 2 -- observed any-temperature channel (bsai/v12_rawany.py):",
        "stale_true = origin_index - last_valid_smoothed_position  # NO grid-end clamp",
        "any_margin = RawAnyCache last daily median - 2.4, from an UNCLAMPED",
        "             14-day window ending at cutoff-1   # NaN when channel is gone:",
        "             # RawAnyCache.features_at clamps like the smoother -- do not",
        "             # reuse it as-is; no fresh reading => gate cannot fire",
        "dark2 = (stale_true > 30) & (any_margin < 0.00) & (remaining >= 30)",
        "gate_include = dark1 | dark2   # dip branch stays dead at the base variant",
        "```",
        "",
        "Notes: tau=0.00 dominates 0.01/0.02 on both axes (precision 0.435 vs "
        "0.326-0.405) -- the gate should fire only when the unfiltered channel "
        "already reads AT/below the 2.40 EOL line. The earlier overhang-mext "
        "attempt flooded (+520 cross-run) because mext extrapolates through "
        "long-ended devices; dark2 cannot flood -- no fresh any-temp reading, no "
        "flag. RawAnyCache must be updated in `predict` when the exchange is "
        "enabled (it is currently gated on `needs_raw`, the same dead path as "
        "the dip gate).",
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
