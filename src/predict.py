"""
BatterySwapAI — generate scheduling-ready predictions from the trained baseline model.

Output format (one row per currently-active battery):
    battery_id | location_id | t_p10 | t_p25 | t_p50 | t_p75 | t_p90 | hazard_next_7d

- t_p{k}: number of days from NOW until this battery has a k% chance of having
  failed, CONDITIONAL on it still being alive today. Computed from the fitted
  survival curve as S(t_now + delta) / S(t_now) = 1 - k/100, solved for delta.
  If the curve never reaches that probability within the prediction horizon,
  the value is NaN (not "never" — just "further out than we can estimate reliably").
- hazard_next_7d: probability this battery fails within the next 7 days, given
  it's alive today: 1 - S(t_now + 7) / S(t_now).
- location: both building_id and room_id are included as separate columns,
  since both may matter for scheduling decisions.
- "Now" is taken as each device's own duration_days, which for currently-active
  (censored) devices already equals days-since-deployment as of the data cutoff.
  This is only valid as long as the feature table's censoring point represents
  "today" — re-run preprocessing against fresh data before relying on this for
  a live deployment.

Usage:
    python predict.py --model-path ../models/cox_baseline.pkl \
                       --data-path ../data/processed/feature_table.parquet \
                       --out-path ../outputs/schedule_predictions.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

COVARIATE_COLS = ["voltage_mean", "voltage_std", "voltage_slope", "temp_mean", "temp_std"]
PERCENTILES = {"t_p10": 0.90, "t_p25": 0.75, "t_p50": 0.50, "t_p75": 0.25, "t_p90": 0.10}
HORIZON_DAYS = 3000  # prediction grid ceiling — well past any observed duration in this dataset
LOCATION_COLS = ["building_id", "room_id"]


def load_active_devices(data_path: Path) -> pd.DataFrame:
    feature_table = pd.read_parquet(data_path)
    # only currently-active (censored) devices need scheduling — already-dead ones don't
    active = feature_table[feature_table["event"] == 0].copy()
    active["t_now"] = active["duration_days"]
    missing_loc = active[LOCATION_COLS].isna().any(axis=1).sum()
    if missing_loc:
        print(f"Warning: {missing_loc} active device(s) missing building_id/room_id.")
    return active


def predict_quantiles_and_hazard(cph, active: pd.DataFrame) -> pd.DataFrame:
    X = active[COVARIATE_COLS]
    time_grid = np.arange(0, HORIZON_DAYS, 1)
    surv_funcs = cph.predict_survival_function(X, times=time_grid)  # columns = row index of X, rows = time_grid

    results = []
    for i, (idx, row) in enumerate(active.iterrows()):
        t_now = row["t_now"]
        s_curve = surv_funcs.iloc[:, i].values  # S(t) for t in time_grid

        s_now = np.interp(t_now, time_grid, s_curve)
        if s_now <= 0:
            # already effectively at end of curve — can't condition further
            out = {k: np.nan for k in PERCENTILES}
            out["hazard_next_7d"] = np.nan
        else:
            s_cond = np.clip(s_curve / s_now, 0, 1)  # conditional survival from t_now onward
            out = {}
            for label, target in PERCENTILES.items():
                below = np.where((time_grid >= t_now) & (s_cond <= target))[0]
                if len(below) == 0:
                    out[label] = np.nan  # curve never reaches this probability within horizon
                else:
                    t_event = time_grid[below[0]]
                    out[label] = round(t_event - t_now, 1)

            s_now7 = np.interp(t_now + 7, time_grid, s_curve)
            hazard_7d = 1 - (s_now7 / s_now)
            out["hazard_next_7d"] = round(float(np.clip(hazard_7d, 0, 1)), 4)

        out["device_id"] = row["device_id"]
        results.append(out)

    return pd.DataFrame(results)


def build_output(active: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = active[["device_id"] + LOCATION_COLS].merge(predictions, on="device_id")
    merged = merged.rename(columns={"device_id": "battery_id"})
    col_order = ["battery_id"] + LOCATION_COLS + ["t_p10", "t_p25", "t_p50", "t_p75", "t_p90", "hazard_next_7d"]
    return merged[col_order]


def run(model_path: Path, data_path: Path, out_path: Path):
    with open(model_path, "rb") as f:
        cph = pickle.load(f)

    active = load_active_devices(data_path)
    print(f"Generating predictions for {len(active)} currently-active devices.")

    predictions = predict_quantiles_and_hazard(cph, active)
    output = build_output(active, predictions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(output.head())

    n_nan = output[["t_p10", "t_p25", "t_p50", "t_p75", "t_p90"]].isna().sum()
    print(f"\nNaN counts per quantile (curve didn't reach that probability within {HORIZON_DAYS}-day horizon):")
    print(n_nan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("../models/cox_baseline.pkl"))
    parser.add_argument("--data-path", type=Path, default=Path("../data/processed/feature_table.parquet"))
    parser.add_argument("--out-path", type=Path, default=Path("../outputs/schedule_predictions.csv"))
    args = parser.parse_args()
    run(args.model_path, args.data_path, args.out_path)