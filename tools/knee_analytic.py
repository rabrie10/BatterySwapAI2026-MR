"""Analytic expected-value evaluation of the KneeBoost floor, at evaluator prices.

No planner run. Both arms select batteries with the same expected-value rule

    swap  iff  p * 280  >  (1 - p) * 0.5 * X + 30,      X = remaining + 30

(280 ~ measured cost of a missed due: 10/day lateness plus the isolated
emergency visit; 0.5*X ~ the evaluator's early penalty against the imputed
EOL at end_time + 30 for a never-due battery; 30 ~ in-window timing + marginal
op + capacity externality). The arms differ only in p: raw OOF v7 versus
max(p, knee floor), floors fitted out-of-fold by building.

Realized outcomes are then priced with evaluator mechanics:
  caught due   0.5*max(d_eol - s, 0) + 10*max(s - d_eol, 0) + op_in
  missed due   10 * (emergency_day(queue) - d_eol) + em_op
  wasted swap  0.5*max(X_eff - s, 0) + op_in,  X_eff = observed d_eol else X
with the emergency queue recomputed per arm (one battery per day, sorted id,
starting the Sunday-padded day after the window), s the assumed in-window swap
day. Deltas are reported per scenario; the floor only ever adds swaps.

    python tools/knee_analytic.py

Writes outputs/knee_analytic.json. Cheap: frame only.
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
sys.path.insert(0, str(REPO_ROOT / "tools"))

from fit_knee import fold_assignment

from bsai.knee import KneeBoost, KneeBoostBanded

LATE_PRICE = 280.0
SWAP_OVERHEAD = 30.0
WINDOW = 42


def scenario_calendar(dataset: Path) -> dict[int, dict]:
    scen = json.loads((dataset / "scenarios.json").read_text())
    out = {}
    for i, s in enumerate(scen):
        start = pd.Timestamp(s["start_time"])
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        end_time = start + pd.to_timedelta(WINDOW, unit="D")
        end_day = end_time.normalize() + pd.to_timedelta(6 - end_time.weekday(), unit="D")
        out[i] = {
            "start": start.normalize(),
            "emergency_offset": float((end_day - start.normalize()) / pd.Timedelta(days=1)),
            "month": int(start.month),
        }
    return out


def select(p: np.ndarray, x: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    return eligible & (p * LATE_PRICE > (1.0 - p) * 0.5 * x + SWAP_OVERHEAD)


def price_scenario(
    rows: pd.DataFrame,
    selected: np.ndarray,
    emergency_offset: float,
    *,
    s_swap: float,
    op_in: float,
    em_op: float,
) -> dict[str, float]:
    due = rows.due.to_numpy()
    d_eol = rows.days_to_eol.to_numpy()
    x_eff = np.where(np.isfinite(d_eol), d_eol, rows.remaining.to_numpy() + 30.0)

    caught = due & selected
    missed = due & ~selected
    wasted = ~due & selected

    caught_cost = float(
        (0.5 * np.maximum(d_eol[caught] - s_swap, 0.0)
         + 10.0 * np.maximum(s_swap - d_eol[caught], 0.0)
         + op_in).sum()
    )
    # Emergency queue: sorted battery id, one per day.
    miss_rows = rows[missed].sort_values("battery")
    queue = np.arange(len(miss_rows), dtype=float)
    lateness = emergency_offset + queue - miss_rows.days_to_eol.to_numpy()
    missed_cost = float((10.0 * np.maximum(lateness, 0.0) + em_op).sum())
    wasted_cost = float(
        (0.5 * np.maximum(x_eff[wasted] - s_swap, 0.0) + op_in).sum()
    )
    return {
        "caught": int(caught.sum()),
        "missed": int(missed.sum()),
        "wasted": int(wasted.sum()),
        "caught_cost": caught_cost,
        "missed_cost": missed_cost,
        "wasted_cost": wasted_cost,
        "total": caught_cost + missed_cost + wasted_cost,
    }


def run_arm(
    frame: pd.DataFrame,
    p_base: np.ndarray,
    p_boost: np.ndarray,
    calendar: dict[int, dict],
    *,
    s_swap: float,
    op_in: float,
    em_op: float,
) -> dict:
    x = frame.remaining.to_numpy() + 30.0
    eligible = frame.remaining.to_numpy() >= 0.0
    sel_base = select(p_base, x, eligible)
    sel_boost = select(p_boost, x, eligible)

    per_scenario = []
    for s, rows in frame.groupby("scenario"):
        idx = rows.index.to_numpy()
        base = price_scenario(
            rows, sel_base[idx], calendar[s]["emergency_offset"],
            s_swap=s_swap, op_in=op_in, em_op=em_op,
        )
        boost = price_scenario(
            rows, sel_boost[idx], calendar[s]["emergency_offset"],
            s_swap=s_swap, op_in=op_in, em_op=em_op,
        )
        per_scenario.append(
            {
                "scenario": int(s),
                "base": base,
                "boost": boost,
                "net_gain": base["total"] - boost["total"],
                "catches_gained": boost["caught"] - base["caught"],
                "wasted_added": boost["wasted"] - base["wasted"],
            }
        )

    def block(lo, hi):
        rows = [r for r in per_scenario if lo <= r["scenario"] <= hi]
        return {
            "net_gain": round(float(np.mean([r["net_gain"] for r in rows])), 1),
            "catches_gained": round(float(np.mean([r["catches_gained"] for r in rows])), 2),
            "wasted_added": round(float(np.mean([r["wasted_added"] for r in rows])), 2),
        }

    gains = np.array([r["net_gain"] for r in per_scenario])
    return {
        "mean_base_total": round(float(np.mean([r["base"]["total"] for r in per_scenario])), 1),
        "mean_boost_total": round(float(np.mean([r["boost"]["total"] for r in per_scenario])), 1),
        "net_gain_per_scenario": round(float(gains.mean()), 1),
        "net_gain_std_err": round(float(gains.std(ddof=1) / np.sqrt(len(gains))), 1),
        "scenarios_worse": int((gains < 0).sum()),
        "catches_per_scenario": {
            "base": round(float(np.mean([r["base"]["caught"] for r in per_scenario])), 2),
            "boost": round(float(np.mean([r["boost"]["caught"] for r in per_scenario])), 2),
        },
        "missed_per_scenario": {
            "base": round(float(np.mean([r["base"]["missed"] for r in per_scenario])), 2),
            "boost": round(float(np.mean([r["boost"]["missed"] for r in per_scenario])), 2),
        },
        "wasted_per_scenario": {
            "base": round(float(np.mean([r["base"]["wasted"] for r in per_scenario])), 2),
            "boost": round(float(np.mean([r["boost"]["wasted"] for r in per_scenario])), 2),
        },
        "blocks": {
            "opening_0_15": block(0, 15),
            "mid_16_31": block(16, 31),
            "closing_32_47": block(32, 47),
        },
        "per_scenario": per_scenario,
    }


def oof_floors(
    frame: pd.DataFrame, fit_kwargs: dict, cls=KneeBoost
) -> tuple[np.ndarray, dict[int, object]]:
    floors = np.zeros(len(frame))
    per_fold: dict[int, object] = {}
    for fold in sorted(frame.fold.unique()):
        others = frame[frame.fold != fold]
        boost = cls.fit(
            others.margin.to_numpy(),
            others.beta30.to_numpy(),
            others.due.to_numpy(),
            others.remaining.to_numpy(),
            **fit_kwargs,
        )
        per_fold[int(fold)] = boost
        mask = (frame.fold == fold).to_numpy()
        floors[mask] = boost.floor_for(
            frame.margin.to_numpy()[mask],
            frame.beta30.to_numpy()[mask],
            frame.remaining.to_numpy()[mask],
        )
    return floors, per_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/frame_oof_raw_beta.parquet"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/train"))
    parser.add_argument("--out", type=Path, default=Path("outputs/knee_analytic.json"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.frame).reset_index(drop=True)
    frame["fold"] = fold_assignment(frame, args.folds)
    calendar = scenario_calendar(args.dataset)

    fit_kwargs = dict(remaining_gate=30.0, min_rows=50, min_events=10, shrink=25.0)
    floors, _ = oof_floors(frame, fit_kwargs)

    # As-pipeline probabilities: each fold's RemainingCalibration factor, the
    # stage that runs after the knee hook in predict_grid. Flooring the
    # calibrated p at the empirical rate is algebraically the compensated
    # pre-calibration floor.
    bundle = joblib.load(args.folds)
    order: dict[int, int] = {}
    calibration_of: dict[int, object] = {}
    for building in sorted(bundle["by_building"]):
        model = bundle["by_building"][building]
        key = id(model)
        if key not in order:
            order[key] = len(order)
            calibration_of[order[key]] = model.calibration
    factor = np.ones(len(frame))
    for fold, cal in calibration_of.items():
        mask = (frame.fold == fold).to_numpy()
        if cal is not None:
            factor[mask] = cal.factor_for(frame.remaining.to_numpy()[mask])
    p_raw = frame.p.to_numpy()
    p_cal = np.clip(p_raw * factor, 0.0, 1.0)

    report: dict = {
        "rule": "swap iff p*280 > (1-p)*0.5*(remaining+30) + 30",
        "fit_kwargs": fit_kwargs,
        "arms": {},
    }
    prices = dict(s_swap=7.0, op_in=1.5, em_op=6.0)
    report["prices"] = prices

    banded_kwargs = dict(min_rows=40, min_events=25, shrink=25.0)
    report["banded_kwargs"] = banded_kwargs
    banded_floors, banded_folds = oof_floors(frame, banded_kwargs, cls=KneeBoostBanded)
    lifted = banded_floors > p_cal
    print(f"banded OOF floors: {int(lifted.sum())} rows lifted above calibrated p "
          f"({lifted.sum()/48:.1f}/scen), dues among them "
          f"{int(frame.due.to_numpy()[lifted].sum())}, "
          f"realized {frame.due.to_numpy()[lifted].mean():.3f}\n")

    for name, base, boost in [
        ("raw", p_raw, np.maximum(p_raw, floors)),
        ("calibrated", p_cal, np.maximum(p_cal, floors)),
        ("raw_banded", p_raw, np.maximum(p_raw, banded_floors)),
        ("calibrated_banded", p_cal, np.maximum(p_cal, banded_floors)),
    ]:
        arm = run_arm(frame, base, boost, calendar, **prices)
        report["arms"][name] = arm
        print(f"== arm {name} ==")
        print(f"  base total/scen {arm['mean_base_total']}, boost {arm['mean_boost_total']}, "
              f"net gain {arm['net_gain_per_scenario']} +- {arm['net_gain_std_err']}")
        print(f"  catches {arm['catches_per_scenario']}  missed {arm['missed_per_scenario']}  "
              f"wasted {arm['wasted_per_scenario']}")
        print(f"  blocks {json.dumps(arm['blocks'])}")
        print(f"  scenarios worse: {arm['scenarios_worse']}/48")

    # Sensitivity on the candidate (banded floors over calibrated p): pricing
    # assumptions, floor haircuts, and fit-discipline variants.
    sensitivity = {}
    candidate = np.maximum(p_cal, banded_floors)
    for label, kwargs in {
        "swap_day_14": dict(s_swap=14.0, op_in=1.5, em_op=6.0),
        "swap_day_21": dict(s_swap=21.0, op_in=1.5, em_op=6.0),
        "op_in_4": dict(s_swap=7.0, op_in=4.0, em_op=6.0),
        "em_op_30": dict(s_swap=7.0, op_in=1.5, em_op=30.0),
    }.items():
        arm = run_arm(frame, p_cal, candidate, calendar, **kwargs)
        sensitivity[label] = {
            "net_gain": arm["net_gain_per_scenario"],
            "std_err": arm["net_gain_std_err"],
            "catches_gained": round(
                arm["catches_per_scenario"]["boost"] - arm["catches_per_scenario"]["base"], 2
            ),
            "wasted_added": round(
                arm["wasted_per_scenario"]["boost"] - arm["wasted_per_scenario"]["base"], 2
            ),
        }

    for label, factor in [("floors_x0.75", 0.75), ("floors_x0.5", 0.5)]:
        arm = run_arm(frame, p_cal, np.maximum(p_cal, banded_floors * factor), calendar, **prices)
        sensitivity[label] = {
            "net_gain": arm["net_gain_per_scenario"], "std_err": arm["net_gain_std_err"],
        }
    for label, fk in {
        "min_rows_50": dict(min_rows=50, min_events=25, shrink=25.0),
        "min_events_40": dict(min_rows=40, min_events=40, shrink=25.0),
        "shrink_60": dict(min_rows=40, min_events=25, shrink=60.0),
    }.items():
        f2, _ = oof_floors(frame, fk, cls=KneeBoostBanded)
        arm = run_arm(frame, p_cal, np.maximum(p_cal, f2), calendar, **prices)
        sensitivity[label] = {
            "net_gain": arm["net_gain_per_scenario"], "std_err": arm["net_gain_std_err"],
        }
    report["sensitivity_calibrated_banded_arm"] = sensitivity
    print("== sensitivity (calibrated banded arm) ==")
    for k, v in sensitivity.items():
        print(f"  {k}: {v}")

    for fold, boost in banded_folds.items():
        report.setdefault("banded_per_fold", {})[str(fold)] = [
            [list(r) for r in b.floors] for b in boost.boosts
        ]

    flat = report["arms"]["calibrated"]["net_gain_per_scenario"]
    banded = report["arms"]["calibrated_banded"]["net_gain_per_scenario"]
    report["verdict"] = {
        "flat_floor": "GO" if flat > 0 else ("MARGINAL" if flat > -30 else "NO-GO"),
        "banded_floor": "GO" if banded > 0 else ("MARGINAL" if banded > -30 else "NO-GO"),
    }
    print(f"\nflat floor net {flat}/scen -> {report['verdict']['flat_floor']}; "
          f"banded net {banded}/scen -> {report['verdict']['banded_floor']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
