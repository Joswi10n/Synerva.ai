#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# ──────────────────────────────────────────────────────────────────────────────
RAW_CSV         = "raw_v2.csv"
MODEL_FILENAME  = "best_lstm_model_deeper.keras"

WINDOW_SIZE     = 60    
HORIZON_MINUTES = 30      
EPOCHS          = 100     
BATCH_SIZE      = 64

BASE_FEATURES = [
    "pm05","pm1","pm025","pm04","pm10",
    "temperature","humidity","pressure",
    "co2","particle_size","occupancy",
    "wall_particles","floor_particles","person_particles"
]
TARGET_COLS   = ["pm05","pm1","pm025","pm04","pm10"]

# ──────────────────────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hr = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df

def add_rolling_means(df: pd.DataFrame, cols: list, window: int = 60) -> pd.DataFrame:
    for c in cols:
        df[f"{c}_rollmean_{window}m"] = (
            df.groupby("zone")[c]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# ──────────────────────────────────────────────────────────────────────────────
def train_offline_deeper():
   
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"'{RAW_CSV}' not found.")
    df = pd.read_csv(RAW_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    
    down = []
    for zone, zone_df in df.groupby("zone"):
        zone_df = zone_df.set_index("timestamp")[BASE_FEATURES]
        zone_1min = zone_df.resample("60s").mean().dropna().reset_index()
        zone_1min["zone"] = zone
        down.append(zone_1min)
    df = pd.concat(down, ignore_index=True)

    
    df = add_time_features(df)
    df = add_rolling_means(df, ["pm05","pm1","pm025","pm04","pm10"], window=WINDOW_SIZE)

    FEATURE_COLUMNS = BASE_FEATURES + ["hour_sin","hour_cos","dow_sin","dow_cos"] + \
                      [f"{p}_rollmean_{WINDOW_SIZE}m" for p in ["pm05","pm1","pm025","pm04","pm10"]]

    
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_all = df[FEATURE_COLUMNS].values
    Y_all = df[TARGET_COLS].values
    X_scaled = scaler_X.fit_transform(X_all)
    Y_scaled = scaler_Y.fit_transform(Y_all)

    
    np.save("scaler_X_scale.npy", scaler_X.scale_)
    np.save("scaler_X_min.npy",   scaler_X.min_)
    np.save("scaler_Y_scale.npy", scaler_Y.scale_)
    np.save("scaler_Y_min.npy",   scaler_Y.min_)

    
    X_windows, Y_targets = [], []
    for _, zone_df in df.groupby("zone"):
        X_zone = pd.DataFrame(
            scaler_X.transform(zone_df[FEATURE_COLUMNS].values),
            columns=FEATURE_COLUMNS
        )
        Y_zone = scaler_Y.transform(zone_df[TARGET_COLS].values)
        for end in range(WINDOW_SIZE-1, len(X_zone)-HORIZON_MINUTES):
            start = end - WINDOW_SIZE + 1
            X_windows.append(X_zone.iloc[start:end+1].values.astype(np.float32))
            Y_targets.append(Y_zone[end + HORIZON_MINUTES].astype(np.float32))

    X_arr = np.stack(X_windows)
    Y_arr = np.stack(Y_targets)

    # 6) Train/validation split
    split = int(len(X_arr)*0.8)
    X_train, X_val = X_arr[:split], X_arr[split:]
    Y_train, Y_val = Y_arr[:split], Y_arr[split:]

    # 7) Build deeper LSTM model
    n_features = X_train.shape[2]
    n_targets  = Y_train.shape[1]
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, n_features)),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(n_targets, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

   
    callbacks = [
        ModelCheckpoint(MODEL_FILENAME, monitor="val_loss", save_best_only=True, verbose=2),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

   
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training vs. Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 11) Save feature list
    with open("feature_columns.txt", "w") as f:
        for col in FEATURE_COLUMNS:
            f.write(col + "\\n")

if __name__ == "__main__":
    train_offline_deeper()
 