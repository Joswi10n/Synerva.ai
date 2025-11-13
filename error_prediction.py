#!/usr/bin/env python3
"""
error_analysis.py

Given two CSV files—
  • predictions.csv: contains timestamps, zone, and predicted PM values
  • actuals.csv:     contains timestamps, zone, and actual PM values

this script merges them on (zone, timestamp) and prints the per‐sample absolute
errors for each PM channel, as well as overall MAE and RMSE.

Usage:
  python3 error_analysis.py --pred predictions.csv --act actuals.csv

Both CSVs must share these columns:
  zone,timestamp,pm05,pm1,pm025,pm04,pm10

In predictions.csv, suffix the PM columns with “_pred” (e.g. pm05_pred).
In actuals.csv, keep them un‐suffixed (pm05, pm1, …).

The script will output each row’s absolute errors, then a final summary:
  • MAE per channel
  • Overall MAE and RMSE (across all channels and samples)
"""

import argparse
import sys
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare predicted vs. actual PM values and report errors."
    )
    p.add_argument(
        "--pred", "-p",
        required=True,
        help="Path to predictions CSV (columns: zone,timestamp,pm05_pred,pm1_pred,pm025_pred,pm04_pred,pm10_pred)"
    )
    p.add_argument(
        "--act", "-a",
        required=True,
        help="Path to actuals CSV     (columns: zone,timestamp,pm05,pm1,pm025,pm04,pm10)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load predictions
    try:
        df_pred = pd.read_csv(args.pred)
    except Exception as e:
        sys.exit(f"Error reading predictions CSV: {e}")

    required_pred_cols = ["zone", "timestamp",
                          "pm05_pred", "pm1_pred", "pm025_pred", "pm04_pred", "pm10_pred"]
    if any(col not in df_pred.columns for col in required_pred_cols):
        sys.exit(f"Predictions CSV must contain columns: {required_pred_cols}")

    # 2) Load actuals
    try:
        df_act = pd.read_csv(args.act)
    except Exception as e:
        sys.exit(f"Error reading actuals CSV: {e}")

    required_act_cols = ["zone", "timestamp", "pm05", "pm1", "pm025", "pm04", "pm10"]
    if any(col not in df_act.columns for col in required_act_cols):
        sys.exit(f"Actuals CSV must contain columns: {required_act_cols}")

    # 3) Merge on zone + timestamp
    df_merged = pd.merge(
        df_pred,
        df_act,
        on=["zone", "timestamp"],
        how="inner",
        suffixes=("_pred", "_act")
    )

    if df_merged.empty:
        sys.exit("No matching (zone,timestamp) pairs found between predictions and actuals.")

    # 4) Compute absolute and squared errors
    channels = ["pm05", "pm1", "pm025", "pm04", "pm10"]
    abs_err_cols = []
    sq_err_cols = []

    for ch in channels:
        pred_col = f"{ch}_pred"
        act_col  = ch
        err_col  = f"{ch}_abs_err"
        se_col   = f"{ch}_sq_err"

        df_merged[err_col] = (df_merged[act_col] - df_merged[pred_col]).abs()
        df_merged[se_col]  = (df_merged[act_col] - df_merged[pred_col]) ** 2

        abs_err_cols.append(err_col)
        sq_err_cols.append(se_col)

    # 5) Print row‐by‐row errors
    print("\nPer‐sample absolute errors:\n")
    for idx, row in df_merged.iterrows():
        ts = row["timestamp"]
        z  = row["zone"]
        err_vals = "  ".join(f"{ch}_err={row[f'{ch}_abs_err']:.1f}" for ch in channels)
        print(f"{idx+1}. Zone={z}, Timestamp={ts}  {err_vals}")

    # 6) Compute MAE per channel, overall MAE, and RMSE
    mae_per_channel = {
        ch: df_merged[f"{ch}_abs_err"].mean()
        for ch in channels
    }
    overall_mae = np.mean(list(mae_per_channel.values()))

    rmse_per_channel = {
        ch: np.sqrt(df_merged[f"{ch}_sq_err"].mean())
        for ch in channels
    }
    overall_rmse = np.sqrt(
        np.mean([df_merged[f"{ch}_sq_err"].mean() for ch in channels])
    )

    # 7) Print summary
    print("\n\nSummary of errors:")
    print("-------------------")
    for ch in channels:
        print(f"{ch}  MAE = {mae_per_channel[ch]:.3f}   RMSE = {rmse_per_channel[ch]:.3f}")
    print(f"\nOverall MAE  (avg across all channels) = {overall_mae:.3f}")
    print(f"Overall RMSE (avg across all channels) = {overall_rmse:.3f}\n")

if __name__ == "__main__":
    main()
