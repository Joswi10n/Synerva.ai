#!/usr/bin/env python3
"""
evaluate_lstm.py  (vectorized windowing, with shape-check)

 - Fast window‐building via numpy.lib.stride_tricks.sliding_window_view
 - Model MAE vs. Persistence, MA & Linear baselines
 - Inference time/sample
 - Auto-detects and corrects (timesteps,features) axis swap
"""

import argparse, time
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from numpy.lib.stride_tricks import sliding_window_view

def add_time_features(df):
    hr = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60
    df["hour_sin"] = np.sin(2*np.pi*hr/24)
    df["hour_cos"] = np.cos(2*np.pi*hr/24)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2*np.pi*dow/7)
    df["dow_cos"] = np.cos(2*np.pi*dow/7)
    return df

def add_roll(df, cols, window):
    for c in cols:
        df[f"{c}_rollmean_{window}m"] = (
            df.groupby("zone")[c]
              .rolling(window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

def build_windows_vectorized(df, feat_cols, target_cols, sx, sy, window_size, horizon):
    X_all = sx.transform(df[feat_cols].values.astype(np.float32))
    Y_all = sy.transform(df[target_cols].values.astype(np.float32))
    X_sw = sliding_window_view(X_all, window_shape=window_size, axis=0)
    first_y_idx = window_size - 1 + horizon
    Y_windows = Y_all[first_y_idx:]
    X_windows = X_sw[:len(Y_windows)]
    return X_windows, Y_windows

def linear_baseline_preds(X_val, K, horizon):
    N, _, _ = X_val.shape
    preds = np.zeros((N,5), dtype=np.float32)
    t = np.arange(K)
    mt, vy = t.mean(), np.var(t)
    for f_idx in range(5):
        y_last = X_val[:, -K:, f_idx]
        my = y_last.mean(axis=1)
        cov = ((y_last - my[:,None])*(t-mt)).mean(axis=1)
        slope = cov / vy
        intercept = my - slope*mt
        preds[:,f_idx] = slope*(K + horizon - 1) + intercept
    return preds

def main(args):
    # 1) Load & resample raw data
    df = pd.read_csv(args.raw_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["zone","timestamp"]).reset_index(drop=True)
    BASE = ["pm05","pm1","pm025","pm04","pm10",
            "temperature","humidity","pressure",
            "co2","particle_size","occupancy",
            "wall_particles","floor_particles","person_particles"]
    chunks=[]
    for zone, zdf in df.groupby("zone"):
        r = (zdf.set_index("timestamp")[BASE]
             .resample("60s").mean().dropna().reset_index())
        r["zone"] = zone
        chunks.append(r)
    df1 = pd.concat(chunks,ignore_index=True).sort_values(["zone","timestamp"]).reset_index(drop=True)

    # 2) Feature engineering
    df2 = add_time_features(df1)
    df2 = add_roll(df2, ["pm05","pm1","pm025","pm04","pm10"], args.window)

    # 3) Load scalers and model
    sx = MinMaxScaler(); sx.scale_, sx.min_ = np.load(args.scaler_x_scale), np.load(args.scaler_x_min)
    sy = MinMaxScaler(); sy.scale_, sy.min_ = np.load(args.scaler_y_scale), np.load(args.scaler_y_min)
    model = load_model(args.model)
    feat_cols = [ln.strip() for ln in open(args.features)]
    targ_cols = ["pm05","pm1","pm025","pm04","pm10"]

    # 4) Build windows
    t0 = time.time()
    X, Y_norm = build_windows_vectorized(df2, feat_cols, targ_cols,
                                         sx, sy, args.window, args.horizon)
    print(f"Built {len(X)} windows in {time.time()-t0:.2f}s")

    # 5) Split into validation and unscale
    split = int(len(X)*0.8)
    X_val, Yn_val = X[split:], Y_norm[split:]
    Y_val = (Yn_val - sy.min_) / sy.scale_

    # --- Shape check & correction ---
    expected_feats = len(feat_cols)
    if X_val.shape[2] != expected_feats:
        print(f"⚠️  Detected shape {X_val.shape}, expecting (N,{args.window},{expected_feats}); swapping axes")
        X_val = X_val.transpose(0,2,1)
        print(f"✅  New X_val shape: {X_val.shape}")

    # 6) Inference timing
    t1 = time.time()
    Yp_norm = model.predict(X_val, verbose=0)
    inf_time = time.time() - t1
    Yp = (Yp_norm - sy.min_) / sy.scale_
    print(f"Inference: {inf_time:.2f}s total  ({inf_time/len(X_val):.4f}s/sample)\n")

    # 7) Compute errors
    mae_mod = mean_absolute_error(Y_val, Yp)
    mae_pers = mean_absolute_error(Y_val, X_val[:, -1, :5])
    print(f"Model MAE              : {mae_mod:.2f}")
    print(f"Persistence MAE        : {mae_pers:.2f}")
    for K in (5,10,30):
        ma = X_val[:, -K:, :5].mean(axis=1)
        print(f"MA({K}m) MAE            : {mean_absolute_error(Y_val,ma):.2f}")
    for K in (5,10,30):
        lt = linear_baseline_preds(X_val,K,args.horizon)
        print(f"LinearTrend({K}m) MAE   : {mean_absolute_error(Y_val,lt):.2f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv",        default="raw_v2.csv")
    parser.add_argument("--model",          default="best_lstm_model.keras")
    parser.add_argument("--scaler_x_scale", default="scaler_X_scale.npy")
    parser.add_argument("--scaler_x_min",   default="scaler_X_min.npy")
    parser.add_argument("--scaler_y_scale", default="scaler_Y_scale.npy")
    parser.add_argument("--scaler_y_min",   default="scaler_Y_min.npy")
    parser.add_argument("--features",       default="feature_columns.txt")
    parser.add_argument("--window",   type=int, default=60)
    parser.add_argument("--horizon",  type=int, default=30)
    args=parser.parse_args()
    main(args)

