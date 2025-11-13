#!/usr/bin/env python3
"""
train_offline.py  (1-min resample, 60-bin window, 30-min horizon)

1) Load raw CSV (timestamp in ms → datetime)
2) Resample each zone to 1 min bins (mean over BASE_FEATURES)
3) Add time-of-day, occupancy, multi-scale rolling features
4) Build sliding windows (60 rows) → train Seq-LSTM (30 min ahead)
5) Save best model + scaler parameters + feature list
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
RAW_CSV         = "raw_v2.csv"
MODEL_FILENAME  = "best_lstm_model.keras"

WINDOW_SIZE     = 60      
HORIZON_MINUTES = 30      
EPOCHS          = 50
BATCH_SIZE      = 64

BASE_FEATURES = [
    "pm05","pm1","pm025","pm04","pm10",
    "temperature","humidity","pressure",
    "co2","particle_size","occupancy",
    "wall_particles","floor_particles","person_particles"
]
TARGET_COLS   = ["pm05","pm1","pm025","pm04","pm10"]

# ──────────────────────────────────────────────────────────────────────────────
# 2) FEATURE-ENGINEERING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def add_time_features(df):
    hr = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df

def add_occupancy_features(df):
    df["occ_diff"] = df.groupby("zone")["occupancy"].diff().fillna(0)
    df["occ_rollsum_60m"] = (
        df.groupby("zone")["occupancy"]
          .rolling(window=60, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )
    return df

def add_multi_scale_roll(df, cols, windows=(5, 15, 60)):
    for w in windows:
        grp = df.groupby("zone")[cols]
        rolled = grp.rolling(window=w, min_periods=1)
        means = rolled.mean().reset_index(level=0, drop=True)
        stds  = rolled.std(ddof=0).fillna(0).reset_index(level=0, drop=True)
        for c in cols:
            df[f"{c}_rollmean_{w}m"] = means[c]
            df[f"{c}_rollstd_{w}m"]  = stds[c]
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 3) TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────────────────────

def train_offline():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"'{RAW_CSV}' not found.")
    df = pd.read_csv(RAW_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["zone", "timestamp"]).reset_index(drop=True)

    # Resample to 1-min bins
    down = []
    for zone_label, zdf in df.groupby("zone"):
        zdf = zdf.set_index("timestamp")
        numeric = zdf[BASE_FEATURES]
        z1 = numeric.resample("60s").mean().dropna()
        z1["zone"] = zone_label
        z1 = z1.reset_index()
        down.append(z1)
    df = pd.concat(down, ignore_index=True).sort_values(["zone","timestamp"]).reset_index(drop=True)

    # Feature engineering
    df = add_time_features(df)
    df = add_occupancy_features(df)
    df = add_multi_scale_roll(df, TARGET_COLS, windows=(5,15,60))

    FEATURE_COLUMNS = (
        BASE_FEATURES +
        ["hour_sin","hour_cos","dow_sin","dow_cos","occ_diff","occ_rollsum_60m"] +
        [f"{c}_rollmean_{w}m" for c in TARGET_COLS for w in (5,15,60)] +
        [f"{c}_rollstd_{w}m"  for c in TARGET_COLS for w in (5,15,60)]
    )
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered features: {missing}")

    # Scale features & targets
    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    X_all = df[FEATURE_COLUMNS].values
    Y_all = df[TARGET_COLS].values

    Xs = scaler_X.fit_transform(X_all).astype(np.float32)
    Ys = scaler_Y.fit_transform(Y_all).astype(np.float32)

    np.save("scaler_X_scale.npy", scaler_X.scale_)
    np.save("scaler_X_min.npy",   scaler_X.min_)
    np.save("scaler_Y_scale.npy", scaler_Y.scale_)
    np.save("scaler_Y_min.npy",   scaler_Y.min_)

    scaled_df = pd.DataFrame(
        np.hstack((Xs, Ys)),
        columns=FEATURE_COLUMNS + [f"{c}_target" for c in TARGET_COLS]
    )
    scaled_df["zone"]      = df["zone"].values
    scaled_df["timestamp"] = df["timestamp"].values

    # Build sliding windows
    X_wins, Y_tars = [], []
    h = HORIZON_MINUTES
    for zone_label, zd in scaled_df.groupby("zone"):
        zd = zd.sort_values("timestamp").reset_index(drop=True)
        total = len(zd)
        for end in range(WINDOW_SIZE - 1, total - h):
            start = end - WINDOW_SIZE + 1
            X_seq = zd.loc[start:end, FEATURE_COLUMNS].values.astype(np.float32)
            Y_seq = zd.loc[end + h, [f"{c}_target" for c in TARGET_COLS]].values.astype(np.float32)
            X_wins.append(X_seq)
            Y_tars.append(Y_seq)

    X_wins = np.stack(X_wins).astype(np.float32)
    Y_tars = np.stack(Y_tars).astype(np.float32)

    # sanity check
    print(f"Built windows: X {X_wins.shape}, Y {Y_tars.shape}")
    print("Any NaNs in X:", np.isnan(X_wins).any(), "Any NaNs in Y:", np.isnan(Y_tars).any())

    # Train/val split
    N = X_wins.shape[0]
    s = int(0.8 * N)
    X_train, X_val = X_wins[:s], X_wins[s:]
    Y_train, Y_val = Y_tars[:s], Y_tars[s:]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # LSTM model
    n_feat = X_train.shape[2]
    n_targ = Y_train.shape[1]
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(WINDOW_SIZE, n_feat)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_targ, activation="linear")
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    model.summary()

    cb = [
        ModelCheckpoint(MODEL_FILENAME, monitor="val_loss", save_best_only=True, mode="min", verbose=2),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=2
    )
    print("✅ Training complete — model saved to", MODEL_FILENAME)

    with open("feature_columns.txt", "w") as f:
        for c in FEATURE_COLUMNS:
            f.write(c + "\n")

if __name__ == "__main__":
    train_offline()
