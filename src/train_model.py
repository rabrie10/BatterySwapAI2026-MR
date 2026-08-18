"""
BatterySwapAI — baseline survival model training.

Loads the processed feature table (from preprocessing.py), fits a Cox
Proportional Hazards model, checks assumptions, and saves the fitted model
so a teammate can load it directly and call .predict_median() /
.predict_survival_function() per device for the scheduling side.

Usage:
    python train_model.py --data-path ../data/processed/feature_table.parquet --out-dir ../models
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter

MODEL_COLS = ["duration_days", "event", "voltage_mean", "voltage_std", "voltage_slope", "temp_mean", "temp_std"]


def load_model_input(data_path: Path) -> pd.DataFrame:
    feature_table = pd.read_parquet(data_path)
    model_df = feature_table[MODEL_COLS].copy()

    assert model_df.isna().sum().sum() == 0, "NaNs found in model input — check preprocessing output"

    return model_df


def train(model_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="duration_days", event_col="event")
    return cph


def evaluate(cph: CoxPHFitter, model_df: pd.DataFrame):
    print("\n=== Model summary ===")
    cph.print_summary()

    print(f"\nConcordance index: {cph.concordance_index_:.3f}")

    print("\n=== Proportional hazards assumption check ===")
    try:
        cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"Assumption check raised an issue — review before trusting coefficients: {e}")


def save_model(cph: CoxPHFitter, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cox_baseline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(cph, f)
    print(f"\nSaved model -> {model_path}")
    print(
        "Load with:\n"
        "  import pickle\n"
        "  with open('cox_baseline.pkl', 'rb') as f:\n"
        "      cph = pickle.load(f)\n"
        "  cph.predict_median(new_device_features_df)\n"
        "  cph.predict_survival_function(new_device_features_df)"
    )


def run(data_path: Path, out_dir: Path):
    model_df = load_model_input(data_path)
    print(f"Training on {model_df.shape[0]} devices, {model_df['event'].sum()} observed failures.")

    cph = train(model_df)
    evaluate(cph, model_df)
    save_model(cph, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("../data/processed/feature_table.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("../models"))
    args = parser.parse_args()
    run(args.data_path, args.out_dir)