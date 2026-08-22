"""Where does the planner's belief diverge from the evaluator's bill?

``tools/belief_v6.py`` measures one number: believed against realised total.
It reports a gap of roughly +1150 with correlation 0.613, which is the largest
measured unexploited quantity in the project -- but a single scalar cannot say
whether the objective is *wrong* or merely *optimistic*.

This tool splits both sides into the same buckets so they can be subtracted
term by term:

    early      timing cost of swapping before the battery was due
    late       timing cost of swapping (or being forced to swap) after
    operational travel, room and building changes, swap time, overtime, limits

and splits ``early`` further by where the belief thought the probability mass
was -- inside the window, after the window but still observed, or never
recorded -- because those three branches are priced by different formulas.

The realised side is split the same way, and additionally by whether the
battery was *genuinely* due inside the window. That last split is what
separates the two competing explanations:

* if believed and realised agree bucket by bucket for genuinely-due batteries
  but the realised early cost of the not-due ones is far above what was
  believed, then the objective is right and the *probabilities* are wrong;
* if some single believed term is systematically off for both groups, that
  term is a bug in the cost model.

    python tools/belief_components.py --folds outputs/v7_folds.joblib --limit 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batteryswap_public.evaluate import cost_components, evaluate_plan
from batteryswap_public.utils import iterate_scenarios, load_dataset, load_devices

from batteryswap_solution.costs import build_expected_cost_tables, select_candidates
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig
from batteryswap_solution.replay import build_replay_context, replay_operational_cost

from bsai.forecaster import HazardForecaster
from bsai.validation import OofHazardModel

OPERATIONAL = ("battery_swap", "building_change", "room_change", "travel",
               "overtime", "daily_limit", "weekly_limit")


def _zeroed(settings, **overrides):
    """A settings clone with some penalties knocked out.

    ``build_expected_cost_tables`` is linear in both daily penalties, so
    evaluating it twice -- once with ``late = 0`` and once with ``early = 0`` --
    recovers the two halves exactly without duplicating the formula here, which
    would drift the moment the real one changed.
    """
    return settings.model_copy(update=overrides)


def _due_samples(probability: np.ndarray, samples: int, seed: int) -> np.ndarray:
    """Reproduce the stratified emergency draws ``_local_search`` uses."""
    if samples <= 0:
        return np.empty((0, probability.size), dtype=bool)
    rng = np.random.default_rng(seed)
    uniforms = np.empty((samples, probability.size), dtype=float)
    strata = (np.arange(samples, dtype=float) + 0.5) / samples
    for index in range(probability.size):
        uniforms[:, index] = rng.permutation(strata)
    return np.unique(uniforms < probability[None, :], axis=0)


def _believed(plan, costs, early_costs, late_costs, locations, travel, settings,
              start, config) -> dict:
    """Decompose ``_expected_score`` over the whole fleet, not just candidates.

    The search itself only scores candidates, but the evaluator bills the fleet:
    a non-candidate that turns out to be due still shows up as an emergency
    visit. Scoring the full table keeps both sides over the same population.
    """
    date_to_index = {date: index for index, date in enumerate(costs.candidate_dates)}
    plan_days = plan.set_index("battery")["day"]

    early = late_planned = 0.0
    planned: set[str] = set()
    deferred_indices: list[int] = []
    for index, battery_id in enumerate(costs.battery_ids):
        day = pd.Timestamp(plan_days.loc[battery_id]).normalize()
        if day in date_to_index:
            column = date_to_index[day]
            early += float(early_costs.service_cost[index, column])
            late_planned += float(late_costs.service_cost[index, column])
            planned.add(battery_id)
        else:
            deferred_indices.append(index)

    late_deferred = float(late_costs.defer_cost[np.asarray(deferred_indices, dtype=int)].sum())

    probability = costs.horizon_event_probability
    samples = _due_samples(probability, config.robust_emergency_samples,
                           config.random_seed)
    context = build_replay_context(locations, travel, settings, start)
    operational = []
    for sample in samples:
        emergency = [
            battery_id
            for index, battery_id in enumerate(costs.battery_ids)
            if sample[index] and battery_id not in planned
        ]
        operational.append(
            replay_operational_cost(plan, locations, travel, settings, start,
                                    emergency_batteries=emergency,
                                    context=context)["total_cost"]
        )
    planned_mask = np.array(
        [battery_id in planned for battery_id in costs.battery_ids], dtype=bool
    )
    return {
        "early": early,
        "late": late_planned + late_deferred,
        "believed_due_planned": float(probability[planned_mask].sum()),
        "believed_due_deferred": float(probability[~planned_mask].sum()),
        "late_planned": late_planned,
        "late_deferred": late_deferred,
        "operational": float(np.mean(operational)),
        "planned": planned,
    }


def _early_branches(plan, costs, locations, settings, dates, forecast) -> dict:
    """Split the believed early cost by which probability branch produced it.

    ``service_cost`` adds three terms: the in-window event PMF, the mass that
    fails after the window but is still on record, and the mass that is never
    recorded and gets the evaluator's ``end_time + 30`` substitute. Only the
    first is a battery the planner thinks it is saving; the other two are what
    it believes a wasted swap costs.
    """
    date_to_index = {date: index for index, date in enumerate(costs.candidate_dates)}
    plan_days = plan.set_index("battery")["day"]
    tail = forecast.tail.set_index("battery_id").reindex(list(costs.battery_ids))
    observed_tail = tail["prob_observed_after_horizon"].to_numpy(dtype=float)
    mean_excess = tail["mean_excess_rul_days_given_observed_after_horizon"].to_numpy(dtype=float)
    unobserved = tail["prob_unobserved_eol"].to_numpy(dtype=float)
    early_rate = float(settings.early_replacement_penalty_daily)

    id_column = "battery_id" if "battery_id" in locations else "battery"
    end_times = pd.to_datetime(
        locations.set_index(locations[id_column].astype(str))["end_time"]
    )
    proxy = (
        end_times + pd.Timedelta(days=float(settings.unobserved_eol_days))
    ).dt.normalize()

    horizon_index = float(len(dates) - 1)
    offsets = np.arange(len(dates), dtype=float)
    out = {"in_window": 0.0, "observed_tail": 0.0, "unobserved": 0.0}
    for index, battery_id in enumerate(costs.battery_ids):
        day = pd.Timestamp(plan_days.loc[battery_id]).normalize()
        if day not in date_to_index:
            continue
        column = date_to_index[day]
        pmf = costs.event_pmf[index]
        out["in_window"] += early_rate * float(
            np.sum(pmf * np.maximum(offsets - column, 0.0))
        )
        out["observed_tail"] += float(
            observed_tail[index] * early_rate
            * (horizon_index - column + mean_excess[index])
        )
        proxy_offset = float(
            (proxy.loc[battery_id] - dates[0]) / pd.Timedelta(days=1)
        )
        out["unobserved"] += float(
            unobserved[index] * early_rate * max(proxy_offset - column, 0.0)
        )
    return out


def _realised(plan, locations, travel, settings, start, not_dead) -> dict:
    """Bill the plan, then attribute every hour to a planned or emergency visit.

    The evaluator appends emergency visits after the final planned day, one per
    day in sorted battery order, so the split is a scan for the first swap of a
    battery the plan missed and a walk back over the travel that reached it.
    """
    horizon_end = start + pd.Timedelta(days=float(settings.planning_window_days))
    transitions, _, scores = evaluate_plan(
        plan, locations, travel, settings, eol_times=not_dead,
        start_time=start, verbose=0,
    )
    served = set(plan.loc[plan["day"] <= horizon_end, "battery"].astype(str))
    due = set(not_dead[(not_dead.notna()) & (not_dead <= horizon_end)].index)
    emergency_ids = {str(value) for value in sorted(due - served)}

    first = None
    for index, transition in enumerate(transitions):
        action = transition["action"]
        if action["name"] == "swap-battery" and str(action["battery"]) in emergency_ids:
            first = index
            break
    if first is not None:
        while first > 0 and transitions[first - 1]["action"]["name"] in (
            "change-building", "change-room", "week-limit"
        ):
            first -= 1

    planned_costs = {component: 0.0 for component in cost_components}
    emergency_costs = {component: 0.0 for component in cost_components}
    for index, transition in enumerate(transitions):
        bucket = emergency_costs if (first is not None and index >= first) else planned_costs
        for component, value in transition["costs"].items():
            # ``compute_daily_costs`` mutates each transition's cost dict on the
            # way past, adding a 'day' key. Read only the real components.
            if component in bucket:
                bucket[component] += float(value)

    # Per-battery timing on the planned swaps, split by whether the battery was
    # genuinely going to be due inside the window. This is the axis that tells a
    # broken cost model apart from a mis-calibrated probability.
    eol = not_dead.copy()
    missing = eol.index[eol.isna()]
    loc = locations.set_index("battery")
    eol.loc[missing] = (
        loc.loc[missing, "end_time"]
        + pd.Timedelta(days=float(settings.unobserved_eol_days))
    ).dt.normalize()
    early_rate = float(settings.early_replacement_penalty_daily)
    late_rate = float(settings.late_replacement_penalty_daily)
    early_due = early_not_due = 0.0
    late_planned_realised = 0.0
    n_due = n_not_due = 0
    active = plan[plan["day"] <= horizon_end]
    for day, battery in zip(active["day"], active["battery"].astype(str)):
        delta = (pd.Timestamp(eol.loc[battery]) - pd.Timestamp(day)) / pd.Timedelta(days=1)
        if delta > 0:
            if battery in due:
                early_due += early_rate * delta
                n_due += 1
            else:
                early_not_due += early_rate * delta
                n_not_due += 1
        else:
            late_planned_realised += late_rate * abs(delta)
            n_due += 1

    return {
        "total": float(scores["total_cost"]),
        "early": float(scores["early_swap"]),
        "late": float(scores["late_swap"]),
        "operational": float(sum(scores[c] for c in OPERATIONAL)),
        "late_planned": late_planned_realised,
        "late_emergency": float(emergency_costs["late_swap"]),
        "operational_planned": float(sum(planned_costs[c] for c in OPERATIONAL)),
        "operational_emergency": float(sum(emergency_costs[c] for c in OPERATIONAL)),
        "early_on_due": early_due,
        "early_on_not_due": early_not_due,
        "n_swapped_due": n_due,
        "n_swapped_not_due": n_not_due,
        "n_missed": len(emergency_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/v8_belief_components.json"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--volatility-scale", type=float, default=1.0)
    args = parser.parse_args()

    devices = load_devices(args.dataset / "devices.csv")
    building_of = dict(zip(devices["device_id"], devices["building_id"]))
    bundle = joblib.load(args.folds)
    for fold_model in bundle["by_building"].values():
        fold_model.volatility_scale = args.volatility_scale
    forecaster = HazardForecaster(
        OofHazardModel(
            by_building=bundle["by_building"],
            building_of=building_of,
            climatology=bundle["climatology"],
        )
    )
    config = PlannerConfig(
        local_search_evaluations=80,
        uncertain_local_search_evaluations=35,
        optimizer=OptimizationConfig(solver_seconds=1.0),
    )
    planner = CompetitionPlanner(forecaster=forecaster, config=config)

    locations, timeseries, eol_times, scenarios = load_dataset(args.dataset)
    rows: list[dict] = []
    for index, (scenario, locs, cut, not_dead) in enumerate(
        iterate_scenarios(locations, timeseries, eol_times, scenarios)
    ):
        if index >= args.limit:
            break
        start = pd.Timestamp(scenario["start_time"])
        settings = scenario["settings"]
        travel = scenario["travel_costs"]
        plan = planner.plan(cut, locs, travel, settings)

        dates, _ = planner._planning_clock(start, settings)
        forecast = planner._forecast(cut, locs, start, dates)
        full = build_expected_cost_tables(forecast, locs, settings, dates)
        early_only = build_expected_cost_tables(
            forecast, locs, _zeroed(settings, late_replacement_penalty_daily=0.0),
            dates,
        )
        late_only = build_expected_cost_tables(
            forecast, locs, _zeroed(settings, early_replacement_penalty_daily=0.0),
            dates,
        )
        believed = _believed(plan, full, early_only, late_only, locs, travel,
                             settings, start, config)
        branches = _early_branches(plan, full, locs, settings, dates, forecast)
        realised = _realised(plan, locs, travel, settings, start, not_dead)

        believed_total = believed["early"] + believed["late"] + believed["operational"]
        entry = {
            "scenario": scenario["name"],
            "believed_total": round(believed_total, 1),
            "realised_total": round(realised["total"], 1),
            "gap_total": round(realised["total"] - believed_total, 1),
            "believed_early": round(believed["early"], 1),
            "realised_early": round(realised["early"], 1),
            "gap_early": round(realised["early"] - believed["early"], 1),
            "believed_late": round(believed["late"], 1),
            "realised_late": round(realised["late"], 1),
            "gap_late": round(realised["late"] - believed["late"], 1),
            "believed_operational": round(believed["operational"], 1),
            "realised_operational": round(realised["operational"], 1),
            "gap_operational": round(
                realised["operational"] - believed["operational"], 1
            ),
            "believed_late_planned": round(believed["late_planned"], 1),
            "believed_late_deferred": round(believed["late_deferred"], 1),
            "realised_late_planned": round(realised["late_planned"], 1),
            "realised_late_emergency": round(realised["late_emergency"], 1),
            "believed_early_in_window": round(branches["in_window"], 1),
            "believed_early_observed_tail": round(branches["observed_tail"], 1),
            "believed_early_unobserved": round(branches["unobserved"], 1),
            "realised_early_on_due": round(realised["early_on_due"], 1),
            "realised_early_on_not_due": round(realised["early_on_not_due"], 1),
            "n_planned": len(believed["planned"]),
            "believed_due_planned": round(believed["believed_due_planned"], 2),
            "believed_due_deferred": round(believed["believed_due_deferred"], 2),
            "n_swapped_due": realised["n_swapped_due"],
            "n_swapped_not_due": realised["n_swapped_not_due"],
            "n_missed": realised["n_missed"],
            "realised_operational_planned": round(realised["operational_planned"], 1),
            "realised_operational_emergency": round(
                realised["operational_emergency"], 1
            ),
        }
        rows.append(entry)
        print(
            f"  {scenario['name']:>5} total {believed_total:7.1f}->{realised['total']:7.1f} "
            f"({realised['total'] - believed_total:+7.1f})  "
            f"early {believed['early']:6.1f}->{realised['early']:6.1f}  "
            f"late {believed['late']:6.1f}->{realised['late']:6.1f}  "
            f"ops {believed['operational']:6.1f}->{realised['operational']:6.1f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    means = {
        column: round(float(frame[column].mean()), 1)
        for column in frame.columns
        if column != "scenario"
    }
    gap = float(frame["gap_total"].mean())
    shares = {
        "early": round(float(frame["gap_early"].mean()) / gap, 3),
        "late": round(float(frame["gap_late"].mean()) / gap, 3),
        "operational": round(float(frame["gap_operational"].mean()) / gap, 3),
    }
    summary = {
        "n_scenarios": len(frame),
        "means": means,
        "gap_share": shares,
        "correlation": round(
            float(frame["believed_total"].corr(frame["realised_total"])), 3
        ),
    }
    print()
    print(json.dumps(summary, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "scenarios": rows}, indent=2))


if __name__ == "__main__":
    main()
