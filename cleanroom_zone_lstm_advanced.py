#!/usr/bin/env python3
# cleanroom_zone_lstm_advanced.py
#
# Enhanced per-zone LSTM with tf.data pipeline, Attention, KerasTuner hyper-search,
# and horizon-aware drift detection + auto-retraining.

import os, argparse, json, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
import keras_tuner as kt

# ────────────────────────────────────────────────────────────
# CLI flags
# ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--window",  type=int, default=60, help="look-back window (min)")
parser.add_argument("--horizon", type=int, default=30, help="forecast horizon (min)")
parser.add_argument("--epochs",  type=int, default=100)
parser.add_argument("--csv",     default="raw_v2.csv")
args = parser.parse_args()

WINDOW_SIZE     = args.window
HORIZON_MINUTES = args.horizon
EPOCHS          = args.epochs
RAW_CSV         = args.csv
BATCH_SIZE      = 64

# base hyper-parameters (same for all zones unless tuner overrides)
BASE_UNITS1, BASE_UNITS2 = 32, 64
DROPOUT_RATE  = 0.25
LEARNING_RATE = 1e-3
DRIFT_THRESHOLD = 1.2      # retrain if val_mae rises 20 % above baseline

BASE_FEATURES = [
    "pm05","pm1","pm025","pm04","pm10",
    "temperature","humidity","pressure",
    "co2","particle_size","occupancy",
    "wall_particles","floor_particles","person_particles"
]
TARGET_COLS = ["pm05","pm1","pm025","pm04","pm10"]

# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
def add_time_features(df):
    hr  = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60.0
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*hr/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hr/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*dow/7.0)
    return df

def add_rolling_means(df, window):
    for c in TARGET_COLS:
        df[f"{c}_rollmean_{window}m"] = (
            df.groupby("zone")[c]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# ────────────────────────────────────────────────────────────
# model-builder
# ────────────────────────────────────────────────────────────
def build_attn_model(input_shape, out_dim, units1, units2, dropout, lr):
    inp = Input(shape=input_shape)
    x   = LSTM(units1, return_sequences=True)(inp)
    x   = Dropout(dropout)(x)
    x   = LSTM(units2, return_sequences=True)(x)
    x   = Dropout(dropout)(x)
    att = Attention()([x, x])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(att)
    out = Dense(out_dim, activation="linear")(ctx)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m

# hyper-model wrapper for KerasTuner
def hypermodel_builder(hp):
    u1 = hp.Int('units1', 16, 64, 16, default=BASE_UNITS1)
    u2 = hp.Int('units2', 32,128, 32, default=BASE_UNITS2)
    dr = hp.Float('dropout', 0.1, 0.5, 0.1, default=DROPOUT_RATE)
    lr = hp.Choice('lr', [1e-2,1e-3,1e-4], default=LEARNING_RATE)
    return build_attn_model((WINDOW_SIZE, len(FEAT_COLS)), len(TARGET_COLS),
                            u1, u2, dr, lr)

# ────────────────────────────────────────────────────────────
# data prep
# ────────────────────────────────────────────────────────────
if not os.path.exists(RAW_CSV):
    raise FileNotFoundError(RAW_CSV)
raw = pd.read_csv(RAW_CSV)
raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms")

frames = []
for z, g in raw.groupby("zone"):
    tmp = (g.set_index("timestamp")[BASE_FEATURES]
           .resample("60s").mean().dropna().reset_index())
    tmp["zone"] = z
    frames.append(tmp)
all_df = pd.concat(frames, ignore_index=True)
all_df = add_time_features(all_df)
all_df = add_rolling_means(all_df, WINDOW_SIZE)

TIME_FEATS = ["hour_sin","hour_cos","dow_sin","dow_cos"]
ROLL_FEATS = [f"{c}_rollmean_{WINDOW_SIZE}m" for c in TARGET_COLS]
FEAT_COLS  = BASE_FEATURES + TIME_FEATS + ROLL_FEATS

# ────────────────────────────────────────────────────────────
# loop over zones
# ────────────────────────────────────────────────────────────
for zone in all_df["zone"].unique():
    print(f"\n=== Zone {zone} ===")
    zdf = all_df[all_df["zone"] == zone].reset_index(drop=True)

    # scalers
    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xs = sx.fit_transform(zdf[FEAT_COLS])
    Ys = sy.fit_transform(zdf[TARGET_COLS])
    np.save(f"scaler_X_scale_{zone}.npy", sx.scale_)
    np.save(f"scaler_X_min_{zone}.npy",   sx.min_)
    np.save(f"scaler_Y_scale_{zone}.npy", sy.scale_)
    np.save(f"scaler_Y_min_{zone}.npy",   sy.min_)

    full_seq = WINDOW_SIZE + HORIZON_MINUTES
    ds_raw = tf.keras.preprocessing.timeseries_dataset_from_array(
        Xs, Ys, sequence_length=full_seq, sequence_stride=1,
        sampling_rate=1, batch_size=None
    ).map(lambda x, y: (x[:WINDOW_SIZE], y)).cache()

    total = len(zdf) - full_seq + 1
    val_n = int(total * 0.20)
    train_n = total - val_n

    val_ds   = ds_raw.take(val_n).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    train_ds = ds_raw.skip(val_n).batch(BATCH_SIZE).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch  = max(train_n // BATCH_SIZE, 1)
    validation_steps = max(val_n   // BATCH_SIZE, 1)

    # hyper-parameter search (skip if too little data)
    if total < BATCH_SIZE * 10:
        print(f"⚠️ Skipping HP search (only {total} samples)")
        best_model = build_attn_model((WINDOW_SIZE,len(FEAT_COLS)), len(TARGET_COLS),
                                      BASE_UNITS1, BASE_UNITS2, DROPOUT_RATE, LEARNING_RATE)
    else:
        tuner = kt.RandomSearch(
            hypermodel_builder, objective="val_mae", max_trials=5,
            directory="kt_dir", project_name=f"zone_{zone}"
        )
        tuner.search(train_ds, validation_data=val_ds, epochs=10,
                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        best_model = tuner.get_best_models(1)[0]

    # callbacks
    cbs = [
        ModelCheckpoint(f"best_lstm_{zone}.weights.h5", save_weights_only=True,
                        monitor="val_loss", save_best_only=True, verbose=2),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=f"logs/{zone}")
    ]

    history = best_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=cbs, verbose=2
    )

    # drift check
    cur_mae = history.history["val_mae"][-1]
    base_file = f"baseline_mae_{zone}.json"
    baseline  = json.load(open(base_file))["val_mae"] if os.path.exists(base_file) else cur_mae
    if cur_mae > baseline * DRIFT_THRESHOLD:
        print(f"🔄 Drift in {zone} – extra fit")
        best_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS//2,
                       steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                       callbacks=cbs, verbose=2)
        cur_mae = best_model.evaluate(val_ds, steps=validation_steps, verbose=0)[1]
    json.dump({"val_mae": cur_mae}, open(base_file, "w"))

    # NEW  ➜ save full model (architecture + weights)
    model_path = f"best_lstm_{zone}.keras"
    best_model.save(model_path)
    print(f"✅ Saved full model to {model_path}")

    # feature list
    with open(f"feature_columns_{zone}.txt", "w") as f:
        f.writelines(c + "\n" for c in FEAT_COLS)

print("\nAll zones processed.")
