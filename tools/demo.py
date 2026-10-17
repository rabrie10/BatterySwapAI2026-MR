"""Run the real planner on a small synthetic fleet; no data or model download.

The supplied failure distributions are illustrative, not learned predictions.
Use --model to also run the shipped Wiener model on synthetic telemetry.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from batteryswap_public.evaluate import EvaluationSettings, check_plan_valid, evaluate_plan
from batteryswap_solution.forecast import CONTRACT_VERSION, ForecastMetadata, RiskForecast
from batteryswap_solution.optimizer import OptimizationConfig
from batteryswap_solution.planner import CompetitionPlanner, PlannerConfig

START = pd.Timestamp("2026-03-02")
HORIZON = 14
PROFILES = [
    ("sensor-01", "b_office", 0.97, 3, 2.425, .0016),
    ("sensor-02", "b_office", 0.91, 5, 2.445, .0013),
    ("sensor-03", "b_lab", 0.83, 9, 2.470, .0011),
    ("sensor-04", "b_lab", 0.02, 11, 2.910, .0002),
    ("sensor-05", "b_depot", 0.01, 12, 3.020, .0001),
    ("sensor-06", "b_office", 0.03, 12, 2.850, .0003),
]


def inputs():
    locations = pd.DataFrame([
        {"battery": name, "building": building, "room": "r_" + building,
         "start_time": START - pd.Timedelta(days=180),
         "end_time": START + pd.Timedelta(days=180)}
        for name, building, *_ in PROFILES
    ])
    distance = {"b_depot": 0, "b_office": .35, "b_lab": .85}
    travel = pd.DataFrame([
        {"from": a, "to": b, "hours": .03 if a == b else abs(distance[a] - distance[b])}
        for a in distance for b in distance
    ])
    timestamps = pd.date_range(end=START, periods=90 * 24 + 1, freq="h")
    age = np.arange(len(timestamps)) / 24
    cycle = np.sin(2 * np.pi * age)
    frames = []
    for name, _, _, _, voltage, slope in PROFILES:
        frames.append(pd.DataFrame({
            "device_id": name, "end_time": timestamps,
            "voltage": voltage + slope * (90 - age) + .006 * cycle,
            "temperature": 21 + 2 * cycle,
        }))
    telemetry = pd.concat(frames).set_index(["device_id", "end_time"]).sort_index()
    settings = EvaluationSettings(base_location="b_depot", base_room="r_b_depot",
                                  planning_window_days=HORIZON)
    return telemetry, locations, travel, settings


class IllustrativeForecaster:
    def predict(self, battery_data, locations, *, prediction_origin, horizon_days,
                evaluation_observation_end):
        dates = pd.date_range(prediction_origin, periods=horizon_days + 1)
        curves, tails = [], []
        for name, _, probability, center, *_ in PROFILES:
            # A normalized smooth CDF with an explicit probability of survival.
            raw = 1 / (1 + np.exp(-(np.arange(len(dates)) - center) / 1.2))
            cdf = probability * (raw - raw[0]) / (raw[-1] - raw[0])
            curves.extend({"battery_id": name, "forecast_date": date, "failure_cdf": float(p)}
                          for date, p in zip(dates, cdf))
            tails.append({"battery_id": name, "prob_observed_after_horizon": 1 - probability,
                          "mean_excess_rul_days_given_observed_after_horizon": 120.,
                          "prob_unobserved_eol": 0.,
                          "prob_no_observed_eol_by_horizon": 1 - probability})
        return RiskForecast(ForecastMetadata(CONTRACT_VERSION, "synthetic-illustration/v1",
                            prediction_origin, dates[-1], horizon_days, evaluation_observation_end),
                            pd.DataFrame(curves), pd.DataFrame(tails), pd.DataFrame())


class RecordingForecaster:
    def __init__(self, inner):
        self.inner = inner
        self.forecast = None

    def predict(self, *args, **kwargs):
        self.forecast = self.inner.predict(*args, **kwargs)
        return self.forecast


def curve_svg(forecast):
    colors = ["#007d68", "#2367cc", "#9654c5", "#b24e1b", "#556477", "#9a4064"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 300" role="img" '
             'aria-label="Cumulative failure probability over fourteen days">',
             '<rect width="850" height="300" fill="#fff"/>']
    for p in [0, .25, .5, .75, 1]:
        y = 240 - p * 190
        parts.append(f'<path d="M60 {y}H620" stroke="#dce4ee"/>'
                     f'<text x="8" y="{y+5}" font-size="13">{p:.0%}</text>')
    for i, (name, group) in enumerate(forecast.curves.groupby("battery_id", sort=True)):
        group = group.sort_values("forecast_date")
        points = " ".join(f'{60+j*560/HORIZON:.1f},{240-float(p)*190:.1f}'
                          for j, p in enumerate(group.failure_cdf))
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
                     f'<text x="650" y="{65+i*28}" fill="{color}" font-size="15">{escape(name)}</text>')
    parts.append('<text x="60" y="272" font-size="14">Day 0</text>'
                 '<text x="570" y="272" font-size="14">Day 14</text></svg>')
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--model", action="store_true", help="use shipped model on synthetic telemetry")
    args = parser.parse_args()
    telemetry, locations, travel, settings = inputs()
    inner = IllustrativeForecaster()
    if args.model:
        from script import load_forecaster
        inner = load_forecaster()
        if inner is None:
            raise SystemExit("Shipped artifact unavailable. Run git lfs pull or omit --model.")
    forecaster = RecordingForecaster(inner)
    planner = CompetitionPlanner(forecaster=forecaster, config=PlannerConfig(
        local_search_evaluations=24, uncertain_local_search_evaluations=12,
        robust_emergency_samples=0, optimizer=OptimizationConfig(solver_seconds=1.0)))
    # A fallback must fail the demo instead of masquerading as a successful run.
    import logging
    class FailOnFallback(logging.Handler):
        def emit(self, record):
            raise RuntimeError(record.getMessage())
    guard = FailOnFallback(level=logging.ERROR)
    logger = logging.getLogger("batteryswap_solution.planner")
    logger.addHandler(guard)
    try:
        plan = planner.plan(telemetry, locations, travel, settings)
    finally:
        logger.removeHandler(guard)
    check_plan_valid(plan, locations, start_time=START)
    if forecaster.forecast is None:
        raise RuntimeError("The demo did not produce a forecast")
    _, _, score = evaluate_plan(plan, locations, travel, settings, eol_times=None,
                                start_time=START, verbose=0)
    cutoff = START + pd.Timedelta(days=HORIZON)
    scheduled = int((pd.to_datetime(plan.day) <= cutoff).sum())
    summary = {"synthetic": True, "forecast_source": "shipped-model" if args.model else "illustrative",
               "batteries": len(locations), "scheduled": scheduled, "deferred": len(plan)-scheduled,
               "operational_cost": float(score["total_cost"]),
               "note": "Operational costs only; no ground-truth EOL labels or forecast accuracy score."}
    args.output.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output / "schedule.csv", index=False)
    forecaster.forecast.curves.to_csv(args.output / "forecast.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = plan.merge(locations[["battery", "building"]], on="battery")
    table["decision"] = np.where(pd.to_datetime(table.day) <= cutoff, "Service", "Defer")
    table["day"] = pd.to_datetime(table.day).dt.strftime("%Y-%m-%d")
    table = table.sort_values(["decision", "day", "battery"], ascending=[False, True, True])
    description = ("Shipped Wiener model applied to synthetic voltage and temperature histories."
                   if args.model else "Illustrative failure distributions passed through the real optimization and routing pipeline.")
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BatterySwapAI | Synthetic planning demo</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#17283f;background:#edf2f7;margin:0}}
main{{max-width:1000px;margin:40px auto;padding:36px;background:white;border-radius:18px}}
h1{{font-size:38px;line-height:1.15}}h2{{font-size:23px;margin-top:32px}}
.tag{{color:#007d68;font-weight:700;letter-spacing:2px;font-size:13px}}
.metrics{{display:flex;gap:18px;flex-wrap:wrap}}.metric{{background:#101c30;color:white;padding:20px;flex:1;border-radius:12px}}
.metric strong{{display:block;font-size:30px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce4ee}}
.note{{background:#edf5ff;padding:16px;border-left:4px solid #2367cc}}svg{{width:100%;height:auto}}
@media(max-width:650px){{main{{margin:12px;padding:18px}}h1{{font-size:28px}}td,th{{padding:6px;font-size:13px}}}}
</style><main><p class="tag">BATTERYSWAPAI / RUNNABLE DEMO</p>
<h1>From failure risk to a service schedule.</h1><p>{description}</p>
<p class="note">Synthetic example. This is a planning walkthrough, not a competition benchmark.
Forecast mode: <strong>{summary['forecast_source']}</strong>.</p>
<div class="metrics"><div class="metric"><strong>{len(plan)}</strong>batteries</div>
<div class="metric"><strong>{scheduled}</strong>scheduled for service</div>
<div class="metric"><strong>{len(plan)-scheduled}</strong>deferred beyond the window</div></div>
<h2>1. Estimate failure risk</h2><p>Each curve gives the probability of a recorded failure by a given day.</p>
{curve_svg(forecaster.forecast)}<h2>2. Allocate work and batch visits</h2>
<p>The planner balances early and late service, travel, work hours, and capacity penalties.</p>
{table[['battery','building','day','decision']].to_html(index=False, border=0)}
<h2>3. Validate the plan</h2><p>Plan completeness and validity checked with the official evaluator.
Operational cost: <strong>{summary['operational_cost']:.2f}</strong> score units.
No true end-of-life labels are supplied, so this does not measure predictive accuracy or total failure-related cost.</p>
<p>Outputs: <a href="schedule.csv">schedule.csv</a> · <a href="forecast.csv">forecast.csv</a> ·
<a href="summary.json">summary.json</a></p></main></html>'''
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Open {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
