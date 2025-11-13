#!/usr/bin/env python3

import argparse, time
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

def build_windows(df, feat_cols, targ_cols, window, horizon):
    # scale features & targets
    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_all = sx.fit_transform(df[feat_cols].values.astype(np.float32))
    Y_all = sy.fit_transform(df[targ_cols].values.astype(np.float32))
    # sliding windows
    X_sw = sliding_window_view(X_all, window_shape=window, axis=0)
    first_y = window-1 + horizon
    X_w = X_sw[:len(Y_all)-first_y]
    Y_w = Y_all[first_y:]
    return X_w, Y_w, sx, sy

def create_model(units1, units2, dropout, lr, input_shape, n_targets):
    m = Sequential()
    m.add(LSTM(units1, return_sequences=True, input_shape=input_shape))
    m.add(Dropout(dropout))
    m.add(LSTM(units2, return_sequences=False))
    m.add(Dropout(dropout))
    m.add(Dense(n_targets, activation="linear"))
    m.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m

def main(args):
    # Load and preprocess data
    df = pd.read_csv(args.raw_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["zone","timestamp"]).reset_index(drop=True)
    BASE = args.base_features
    # resample
    parts = []
    for z, grp in df.groupby("zone"):
        r = grp.set_index("timestamp")[BASE].resample("60s").mean().dropna().reset_index()
        r["zone"] = z
        parts.append(r)
    df1 = pd.concat(parts, ignore_index=True).sort_values(["zone","timestamp"]).reset_index(drop=True)
    df1 = add_time_features(df1)
    df1 = add_roll(df1, args.target_cols, args.window)

    feat_cols = [c for c in args.feature_file.read().splitlines()]
    targ_cols = args.target_cols

    X, Y, sx, sy = build_windows(df1, feat_cols, targ_cols, args.window, args.horizon)
    split = int(len(X)*0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    input_shape = X_train.shape[1], X_train.shape[2]
    n_targs = Y_train.shape[1]

    best_mae, best_params = np.inf, None

    # sweep
    for u1 in args.units1:
        for u2 in args.units2:
            for dr in args.dropouts:
                for lr in args.lrs:
                    for bs in args.batch_sizes:
                        print(f"\n→ Testing: units1={u1}, units2={u2}, dropout={dr}, lr={lr}, batch={bs}")
                        model = create_model(u1, u2, dr, lr, input_shape, n_targs)
                        cb = [
                            EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True, verbose=0),
                            ReduceLROnPlateau(monitor="val_mae", patience=2, factor=0.5, min_lr=1e-6, verbose=0)
                        ]
                        start = time.time()
                        hist = model.fit(
                            X_train, Y_train,
                            validation_data=(X_val, Y_val),
                            epochs=args.epochs,
                            batch_size=bs,
                            callbacks=cb,
                            verbose=0
                        )
                        val_mae = min(hist.history["val_mae"])
                        print(f"  → val_mae = {val_mae:.4f} (took {time.time()-start:.1f}s)")
                        if val_mae < best_mae:
                            best_mae, best_params = val_mae, (u1,u2,dr,lr,bs)

    print(f"\n✅ Best validation MAE: {best_mae:.4f}")
    print(f"   Hyperparameters: units1,units2,dropout,lr,batch = {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default="raw_v2.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base_features", nargs="+", default=[
        "pm05","pm1","pm025","pm04","pm10",
        "temperature","humidity","pressure",
        "co2","particle_size","occupancy",
        "wall_particles","floor_particles","person_particles"])
    parser.add_argument("--target_cols", nargs="+", default=["pm05","pm1","pm025","pm04","pm10"])
    parser.add_argument("--feature_file", type=argparse.FileType("r"), default="feature_columns.txt")
    parser.add_argument("--units1", nargs="+", type=int, default=[64,128])
    parser.add_argument("--units2", nargs="+", type=int, default=[64,128])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.2,0.3])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3,2e-4])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64,128])
    args = parser.parse_args()
    main(args)
