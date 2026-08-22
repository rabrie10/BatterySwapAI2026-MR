"""Profile the three populations the residual reranker has to separate.

* **A -- what V8 actually swapped.** The planner's own served set, from
  ``tools/validate_v6.py --served-out``, split into the swaps justified by a
  real EOL inside the window (TP) and the wasted ones (FP). This is the
  population that pays the 764 points of early cost.
* **B -- what V9 would add.** Rows V9's blend lifts materially above V8. On
  public V9 planned one more swap per scenario and caught nothing extra, so
  these are hard negatives with a known verdict.
* **C -- V8's hard misses.** Due batteries the planner did not serve, split by
  whether V8 gave them any priority at all.

    python tools/fj_populations.py --served outputs/fj_v8_served.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bsai.features import FEATURE_NAMES  # noqa: E402
from tools.fj_derived import derived, room_of  # noqa: E402
from tools.fj_frame import decision_probability, grid_for, load_frame  # noqa: E402

PROFILE = [
    "margin", "p42", "dwell_2.45", "dwell_2.50", "dwell_2.60",
    "p07_over_p42", "p28_over_p42", "cdf_front_mass", "p_late_tail",
    "rel_margin_room", "rel_slope30_room", "rel_temp_room",
    "staleness", "gap_fraction_90",
]
RAW = [
    "voltage", "voltage_compensated", "slope_30", "slope_90", "slope_ratio_14_90",
    "curvature_30_120", "beta_30", "beta_rise", "v_std_30", "temp_now",
    "age_days", "observations", "knee_slope_vs_history", "knee_trend_residual",
    "crossing_30", "season_sin",
]


def summarise(name: str, columns: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for signal, values in columns.items():
        entry = {"signal": signal}
        for label, mask in masks.items():
            subset = values[mask]
            subset = subset[np.isfinite(subset)]
            entry[f"{label}_med"] = round(float(np.median(subset)), 4) if subset.size else np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def auc(label: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    good = np.isfinite(score)
    if label[good].sum() in (0, int(good.sum())):
        return float("nan")
    return float(roc_auc_score(label[good], score[good]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=Path("outputs/v9_frame.npz"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/v7_folds.joblib"))
    parser.add_argument("--v9-folds", type=Path, default=Path("outputs/v9_blend_folds.joblib"))
    parser.add_argument("--served", type=Path, default=Path("outputs/fj_v8_served.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/fj_populations.json"))
    args = parser.parse_args()

    frame = load_frame(args.frame)
    grid = grid_for(frame, args.folds)
    base = decision_probability(grid, frame.remaining)
    signals = derived(frame, grid, room_of())
    columns = {name: signals[name] for name in PROFILE}
    columns.update({
        name: frame.features[:, FEATURE_NAMES.index(name)].astype(float) for name in RAW
    })

    key = {(int(s), str(b)): i for i, (s, b) in enumerate(zip(frame.scenario, frame.battery))}
    served = np.zeros(frame.due.size, dtype=bool)
    for record in json.loads(args.served.read_text()):
        index = int(record["scenario_index"])
        for battery in record["served"]:
            row = key.get((index, str(battery)))
            if row is not None:
                served[row] = True

    n = frame.n_scenarios
    tp, fp = served & frame.due, served & ~frame.due
    miss = frame.due & ~served
    print(f"population A -- what V8 swapped: {served.sum()/n:.2f}/scenario, "
          f"TP {tp.sum()/n:.2f}, FP {fp.sum()/n:.2f} (precision {tp.sum()/served.sum():.3f})")
    print(f"population C -- misses: {miss.sum()/n:.2f}/scenario, "
          f"of which V8 p<0.02: {(miss & (base < 0.02)).sum()/n:.2f}")
    print()

    print("how concentrated is the waste?")
    devices, counts = np.unique(frame.battery[fp], return_counts=True)
    order = np.argsort(-counts)
    top = counts[order][:10].sum()
    print(f"  {devices.size} distinct devices carry {int(fp.sum())} wasted swaps; "
          f"the worst 10 carry {top} ({top/fp.sum():.1%})")
    for position in order[:10]:
        battery = devices[position]
        rows = frame.battery == battery
        print(f"    {battery}  wasted {counts[position]:2d}  "
              f"ever due {int(frame.due[rows].sum()):2d}  "
              f"median margin {np.median(signals['margin'][rows]):.4f}  "
              f"median p {np.median(base[rows]):.3f}  "
              f"median dwell45 {np.median(signals['dwell_2.45'][rows]):.0f}")
    print()

    # ---- V9-only additions ------------------------------------------------
    v9 = None
    if args.v9_folds.exists():
        v9_grid = grid_for(frame, args.v9_folds)
        v9 = decision_probability(v9_grid, frame.remaining)
        lift = np.log(np.clip(v9, 1e-9, 1)) - np.log(np.clip(base, 1e-9, 1))
        # The rows V9 promotes into the top 18 of a scenario that V8 kept out.
        promoted = np.zeros(frame.due.size, dtype=bool)
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero(frame.scenario == index)
            in_v8 = set(rows[np.argsort(-base[rows])][:18])
            in_v9 = set(rows[np.argsort(-v9[rows])][:18])
            for row in in_v9 - in_v8:
                promoted[row] = True
        print(f"population B -- V9 promotes {promoted.sum()/n:.2f} rows/scenario into "
              f"the top 18 that V8 excluded; their realised due rate is "
              f"{frame.due[promoted].mean():.3f} against "
              f"{frame.due[(base >= np.percentile(base, 90))].mean():.3f} in V8's own top decile")
        demoted = np.zeros(frame.due.size, dtype=bool)
        for index in np.unique(frame.scenario):
            rows = np.flatnonzero(frame.scenario == index)
            in_v8 = set(rows[np.argsort(-base[rows])][:18])
            in_v9 = set(rows[np.argsort(-v9[rows])][:18])
            for row in in_v8 - in_v9:
                demoted[row] = True
        print(f"                V9 drops {demoted.sum()/n:.2f} rows/scenario that V8 kept; "
              f"their realised due rate is {frame.due[demoted].mean():.3f}")
        print(f"                net exchange due rate: promoted "
              f"{frame.due[promoted].sum()} vs dropped {frame.due[demoted].sum()}")
        print()

    masks = {
        "TP": tp,
        "FP": fp,
        "miss": miss,
        "missVIS": miss & (base >= 0.05),
    }
    if v9 is not None:
        masks["V9add"] = promoted
    table = summarise("A", columns, masks)
    table["auc_TPvsFP"] = [
        round(auc(np.r_[np.ones(int(tp.sum())), np.zeros(int(fp.sum()))],
                  np.r_[columns[s][tp], columns[s][fp]]), 4)
        for s in table["signal"]
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "served_per_scenario": round(float(served.sum()) / n, 3),
        "tp_per_scenario": round(float(tp.sum()) / n, 3),
        "fp_per_scenario": round(float(fp.sum()) / n, 3),
        "miss_per_scenario": round(float(miss.sum()) / n, 3),
        "profile": json.loads(table.to_json(orient="records")),
    }, indent=1))


if __name__ == "__main__":
    main()
