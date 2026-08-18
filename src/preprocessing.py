"""
BatterySwapAI — preprocessing pipeline.

Raw files -> one-row-per-device feature table ready for survival modeling.

Decisions baked in here (see EDA notebook for the reasoning/validation):
- reading_time / failure_time / deploy_start / window_end all localized to UTC
  to avoid tz-naive vs tz-aware comparison errors.
- rows with missing voltage/temperature dropped (8 rows, confirmed none fell
  inside any device's feature window, so this doesn't change slopes computed
  earlier — but dropped anyway for correctness going forward).
- duplicate (device_id, reading_time) rows dropped, keep first (none found,
  kept as a safeguard).
- duration/event derived per device: event=1 + duration=(failure_time - deploy_start)
  if the device has a recorded failure_time, else event=0 + duration=(window_end - deploy_start).
- completeness ratio computed per device (actual / expected hourly readings over
  its full observed span) and flagged (not dropped) below 0.9.
- features computed on a FIXED EARLY WINDOW (first WINDOW_DAYS after deploy_start)
  to avoid leakage and because the minimum observed survival time across the
  whole dataset (366 days) safely covers this window for every device.
- devices with fewer than 2 readings inside that window are dropped (10 devices
  in the original run) — simple/explicit exclusion rather than a fallback window.

Usage:
    python preprocessing.py --raw-dir ../data/raw/train --out-dir ../data/processed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 60
COMPLETENESS_FLAG_THRESHOLD = 0.9


def load_raw(raw_dir: Path):
    metrics = pd.read_parquet(raw_dir / "battery_metrics.parquet")
    eol = pd.read_csv(raw_dir / "eol_times.csv")
    devices = pd.read_csv(raw_dir / "devices.csv")

    metrics = metrics.rename(columns={"end_time": "reading_time"})
    eol = eol.rename(columns={"end_time": "failure_time"})
    devices = devices.rename(columns={"start_time": "deploy_start", "end_time": "window_end"})

    metrics["reading_time"] = pd.to_datetime(metrics["reading_time"]).dt.tz_localize("UTC")
    eol["failure_time"] = pd.to_datetime(eol["failure_time"]).dt.tz_localize("UTC")
    devices["deploy_start"] = pd.to_datetime(devices["deploy_start"])
    devices["window_end"] = pd.to_datetime(devices["window_end"], format="ISO8601")

    return metrics, eol, devices


def clean_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_clean = metrics.dropna(subset=["voltage", "temperature"]).copy()
    metrics_clean = metrics_clean.drop_duplicates(subset=["device_id", "reading_time"], keep="first")
    return metrics_clean


def build_survival_table(metrics_clean: pd.DataFrame, eol: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    surv = eol.merge(
        devices[["device_id", "deploy_start", "window_end", "building_id", "room_id"]],
        on="device_id", how="left",
    )
    surv["event"] = surv["failure_time"].notna().astype(int)
    surv["end_of_observation"] = surv["failure_time"].fillna(surv["window_end"])
    surv["duration_days"] = (surv["end_of_observation"] - surv["deploy_start"]).dt.total_seconds() / 86400

    device_span = metrics_clean.groupby("device_id", observed=True)["reading_time"].agg(["min", "max", "count"])
    device_span["expected_hourly_readings"] = (
        (device_span["max"] - device_span["min"]).dt.total_seconds() / 3600
    ).round().astype(int) + 1
    device_span["completeness"] = device_span["count"] / device_span["expected_hourly_readings"]

    survival_table = surv.merge(
        device_span[["min", "max", "count", "completeness"]],
        left_on="device_id", right_index=True, how="left",
    ).rename(columns={"min": "first_reading", "max": "last_reading", "count": "n_readings"})

    survival_table["low_completeness"] = survival_table["completeness"] < COMPLETENESS_FLAG_THRESHOLD

    assert survival_table["device_id"].is_unique, "duplicate device_id in survival_table"
    assert (survival_table["duration_days"] >= 0).all(), "negative duration found"

    return survival_table


def engineer_device_features(device_id, metrics_df, deploy_start, window_days=WINDOW_DAYS):
    cutoff = deploy_start + pd.Timedelta(days=window_days)
    d = metrics_df[
        (metrics_df["device_id"] == device_id)
        & (metrics_df["reading_time"] >= deploy_start)
        & (metrics_df["reading_time"] < cutoff)
    ].sort_values("reading_time")

    if len(d) < 2:
        return None

    elapsed_hours = (d["reading_time"] - deploy_start).dt.total_seconds() / 3600
    voltage_slope = np.polyfit(elapsed_hours, d["voltage"], 1)[0]

    return pd.Series({
        "n_readings_window": len(d),
        "voltage_mean": d["voltage"].mean(),
        "voltage_std": d["voltage"].std(),
        "voltage_slope": voltage_slope,
        "temp_mean": d["temperature"].mean(),
        "temp_std": d["temperature"].std(),
    })


def build_feature_table(survival_table: pd.DataFrame, metrics_clean: pd.DataFrame) -> pd.DataFrame:
    deploy_lookup = survival_table.set_index("device_id")["deploy_start"]

    feature_rows = []
    for device_id, deploy_start in deploy_lookup.items():
        feats = engineer_device_features(device_id, metrics_clean, deploy_start)
        if feats is not None:
            feats["device_id"] = device_id
            feature_rows.append(feats)

    features = pd.DataFrame(feature_rows)
    feature_table = survival_table.merge(features, on="device_id", how="left")

    n_before = len(feature_table)
    feature_cols = ["voltage_mean", "voltage_std", "voltage_slope", "temp_mean", "temp_std"]
    feature_table = feature_table.dropna(subset=feature_cols).reset_index(drop=True)
    n_dropped = n_before - len(feature_table)
    if n_dropped:
        print(f"Dropped {n_dropped} device(s) with <2 readings in the first {WINDOW_DAYS}-day window.")

    return feature_table


def run(raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, eol, devices = load_raw(raw_dir)
    metrics_clean = clean_metrics(metrics)

    survival_table = build_survival_table(metrics_clean, eol, devices)
    feature_table = build_feature_table(survival_table, metrics_clean)

    out_path = out_dir / "feature_table.parquet"
    feature_table.to_parquet(out_path, index=False)
    print(f"Saved feature_table: {feature_table.shape} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("../data/raw/train"))
    parser.add_argument("--out-dir", type=Path, default=Path("../data/processed"))
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)