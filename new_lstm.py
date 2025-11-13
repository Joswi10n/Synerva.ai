#!/usr/bin/env python3
# new_lstm.py
#
# Pull semicon sensor data from InfluxDB and train per‑sensor LSTM(+Attention).
# Robust to slow servers (probe first, explicit connect/read timeouts, optional chunking).
# FIX: drop non‑numeric columns (e.g., sensor_id) before resampling to avoid TypeError.

import os
import argparse
import sys
import re
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

try:
    from urllib3.util import Timeout as _Urllib3Timeout
except Exception:
    _Urllib3Timeout = None

# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--url",     required=True,  help="InfluxDB base URL, e.g. http://localhost:8086")
parser.add_argument("--token",   required=True,  help="InfluxDB API Token (or use $env:INFLUX_TOKEN)")
parser.add_argument("--org",     required=True,  help="InfluxDB org name")
parser.add_argument("--bucket",  required=True,  help="InfluxDB bucket name")

parser.add_argument("--sensor-tag", default="sensor_id", help="Tag key identifying a sensor (default: sensor_id)")
parser.add_argument("--range",      default="2h",         help="Lookback (Flux duration, e.g. 2h, 24h, 7d)")
parser.add_argument("--aggregate-every", default="60s",   help="Aggregate window before pivot (e.g. 60s, 5m)")

# Timeouts & query behavior
parser.add_argument("--timeout-sec",     type=int, default=120, help="Read timeout (seconds) for queries")
parser.add_argument("--connect-timeout", type=int, default=10,  help="Connect timeout (seconds)")
parser.add_argument("--probe", action="store_true", help="Run quick 1-row probe and exit")
parser.add_argument("--chunk-min", type=int, default=0,
                    help="If >0, split --range into N-minute chunks per query. 0 = no chunking")

# Model/training
parser.add_argument("--window",  type=int, default=60, help="look‑back window (samples)")
parser.add_argument("--horizon", type=int, default=30, help="forecast horizon (samples)")
parser.add_argument("--epochs",  type=int, default=3)
parser.add_argument("--batch-size", type=int, default=64)

# Features
parser.add_argument("--include-counts", action="store_true",
                    help="Include pm*_# count measurements as features.")

args = parser.parse_args()

URL            = args.url
TOKEN          = args.token or os.getenv("INFLUX_TOKEN", "")
ORG            = args.org
BUCKET         = args.bucket
SENSOR_TAG     = args.sensor_tag
RANGE          = args.range
AGG_EVERY      = args.aggregate_every
READ_TIMEOUT   = args.timeout_sec
CONNECT_TIMEOUT= args.connect_timeout
CHUNK_MIN      = args.chunk_min

WINDOW_SIZE    = args.window
HORIZON        = args.horizon
EPOCHS         = args.epochs
BATCH_SIZE     = args.batch_size

if not TOKEN:
    print("❌ No token provided. Pass --token or set INFLUX_TOKEN.", file=sys.stderr)
    sys.exit(1)

# ────────────────────────────────────────────────────────────
# Measurements & features (from your logs)
# ────────────────────────────────────────────────────────────
MEAS_BASE = [
    "pm1_0", "pm2_5", "pm4_0", "pm10_0",
    "temperature", "humidity", "pressure", "co2",
    "particleSize", "battery"
]
MEAS_COUNTS = ["pm0_5#", "pm1_0#", "pm2_5#", "pm4_0#", "pm10_0#"]
MEAS_ALL = MEAS_BASE + (MEAS_COUNTS if args.include_counts else [])

TARGET_COLS = [m for m in ["pm1_0", "pm2_5", "pm4_0", "pm10_0"] if m in MEAS_BASE]
FEATURE_BASE = MEAS_ALL[:]

# ────────────────────────────────────────────────────────────
# Flux helpers
# ────────────────────────────────────────────────────────────
def regex_union(names: List[str]) -> str:
    escaped = [re.escape(n) for n in names]
    return "^(" + "|".join(escaped) + ")$"

def build_flux(bucket: str, sensor_tag: str, meas: List[str], lookback: str, every: str,
               start_iso: str|None=None, stop_iso: str|None=None, limit_rows: int|None=None) -> str:
    name_re = regex_union(meas)
    keep_cols = ["_time", sensor_tag] + meas
    keep_str  = ", ".join([f"\"{c}\"" for c in keep_cols])

    if start_iso and stop_iso:
        range_line = f'range(start: {start_iso}, stop: {stop_iso})'
    else:
        range_line = f'range(start: -{lookback})'

    flux = f'''
from(bucket: "{bucket}")
  |> {range_line}
  |> filter(fn: (r) => r._measurement =~ /{name_re}/)
  |> aggregateWindow(every: {every}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time","{sensor_tag}"], columnKey: ["_measurement"], valueColumn: "_value")
  |> keep(columns: [{keep_str}])
'''.rstrip()

    if limit_rows and limit_rows > 0:
        flux += f'\n  |> limit(n:{limit_rows})'

    return flux

def query_table(client: InfluxDBClient, flux: str) -> pd.DataFrame:
    qapi = client.query_api()
    try:
        out = qapi.query_data_frame(flux)
    except ApiException as e:
        print("\n❌ InfluxDB error:")
        print(str(e))
        print("\nFlux was:\n")
        print(flux)
        raise

    if out is None:
        return pd.DataFrame()
    if isinstance(out, list):
        frames = [f for f in out if isinstance(f, pd.DataFrame) and len(f.columns) > 0]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return out

# ────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────
def build_attn_model(input_shape: Tuple[int, int], out_dim: int) -> Model:
    inp = Input(shape=input_shape)
    x   = LSTM(64, return_sequences=True)(inp)
    x   = Dropout(0.25)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.25)(x)
    att = Attention()([x, x])
    ctx = GlobalAveragePooling1D()(att)
    out = Dense(out_dim, activation="linear")(ctx)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m

# ────────────────────────────────────────────────────────────
# Time features & rolling means
# ────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hr  = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60.0
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*hr/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hr/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*dow/7.0)
    return df

def add_rolling(df: pd.DataFrame, window: int, targets: List[str], key_col: str) -> pd.DataFrame:
    for c in targets:
        roll = (
            df.groupby(key_col)[c]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
        df[f"{c}_rollmean_{window}"] = roll
    return df

# ────────────────────────────────────────────────────────────
# Connect (with explicit timeouts) & probe
# ────────────────────────────────────────────────────────────
print("Connecting to InfluxDB …")
timeout_arg = None
if _Urllib3Timeout is not None:
    timeout_arg = _Urllib3Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT)

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG, timeout=timeout_arg or READ_TIMEOUT)

# quick probe (very small result) to avoid long-running first call
probe_flux = build_flux(BUCKET, SENSOR_TAG, MEAS_BASE[:2], lookback="1m", every="1m", limit_rows=1)
print("Probing connectivity with a 1-row query …")
try:
    _probe = query_table(client, probe_flux)
except Exception:
    print("\n❌ Probe failed. Flux was:\n")
    print(probe_flux)
    sys.exit(1)

if args.probe:
    print("✅ Probe OK. Exiting due to --probe.")
    sys.exit(0)

# ────────────────────────────────────────────────────────────
# Full query (optionally chunked)
# ────────────────────────────────────────────────────────────
def parse_range_to_minutes(r: str) -> int:
    r = r.strip().lower()
    if r.endswith("h"):
        return int(float(r[:-1]) * 60)
    if r.endswith("m"):
        return int(float(r[:-1]))
    if r.endswith("d"):
        return int(float(r[:-1]) * 24 * 60)
    raise ValueError(f"Unsupported --range format: {r}")

def chunked_time_windows(now: datetime, total_min: int, chunk_min: int) -> List[Tuple[str, str]]:
    out = []
    stop = now
    remaining = total_min
    while remaining > 0:
        span = min(chunk_min, remaining)
        start = stop - timedelta(minutes=span)
        out.append((start.isoformat() + "Z", stop.isoformat() + "Z"))
        stop = start
        remaining -= span
    return list(reversed(out))  # oldest -> newest

frames = []

if CHUNK_MIN and CHUNK_MIN > 0:
    total_min = parse_range_to_minutes(RANGE)
    windows = chunked_time_windows(datetime.utcnow(), total_min, CHUNK_MIN)
    print(f"Querying in {len(windows)} chunks of ~{CHUNK_MIN} minutes …")
    for i, (start_iso, stop_iso) in enumerate(windows, 1):
        flux = build_flux(BUCKET, SENSOR_TAG, MEAS_ALL, lookback=RANGE, every=AGG_EVERY,
                          start_iso=start_iso, stop_iso=stop_iso)
        print(f"  → chunk {i}/{len(windows)}: {start_iso} .. {stop_iso}")
        df = query_table(client, flux)
        if df is not None and not df.empty:
            frames.append(df)
else:
    flux = build_flux(BUCKET, SENSOR_TAG, MEAS_ALL, lookback=RANGE, every=AGG_EVERY)
    print("Running Flux query …")
    df = query_table(client, flux)
    if df is not None and not df.empty:
        frames.append(df)

if not frames:
    raise RuntimeError(
        "No data returned after query. Try increasing --range, enabling --include-counts (if those are present), "
        "or verify bucket/measurements/tag names."
    )

raw_df = pd.concat(frames, ignore_index=True)

# ────────────────────────────────────────────────────────────
# Prep
# ────────────────────────────────────────────────────────────
if "_time" not in raw_df.columns:
    raise RuntimeError(f"No _time column in query result: {list(raw_df.columns)}")

raw_df = raw_df.rename(columns={"_time": "timestamp"})
if SENSOR_TAG not in raw_df.columns:
    raise RuntimeError(
        f"Tag '{SENSOR_TAG}' was not found in the result. "
        f"Validate the tag key or pass --sensor-tag <your_tag>."
    )

present_meas = [c for c in MEAS_ALL if c in raw_df.columns]
if not present_meas:
    raise RuntimeError("Measurements list resulted in zero columns after pivot. Check names/escape characters.")

keep_cols = ["timestamp", SENSOR_TAG] + present_meas
raw_df = raw_df[keep_cols].copy()
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
raw_df = raw_df.dropna(subset=["timestamp"])
raw_df = raw_df.sort_values(["timestamp", SENSOR_TAG]).reset_index(drop=True)

# Ensure numeric for all measurements
for c in present_meas:
    raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

# Forward fill per sensor
raw_df[present_meas] = raw_df.groupby(SENSOR_TAG)[present_meas].ffill()

# ────────────────────────────────────────────────────────────
# Resample to 60s  (FIX: drop non‑numeric columns before mean)
# ────────────────────────────────────────────────────────────
frames = []
for sid, g in raw_df.groupby(SENSOR_TAG):
    g = g.sort_values("timestamp").reset_index(drop=True)
    # keep only numeric columns + timestamp for resample
    num_cols = g.select_dtypes(include=["number"]).columns.tolist()
    # ensure we include all present_meas that might be numeric but typed as object for some reason
    num_cols = sorted(set(num_cols).union(set(present_meas)))
    cols_for_resample = ["timestamp"] + [c for c in num_cols if c in g.columns]
    g_num = g[cols_for_resample].copy()
    g_num = g_num.set_index("timestamp").resample("60s").mean(numeric_only=True).reset_index()
    g_num[SENSOR_TAG] = sid  # add tag back
    frames.append(g_num)

all_df = pd.concat(frames, ignore_index=True)

# ────────────────────────────────────────────────────────────
# Features & targets
# ────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hr  = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60.0
    dow = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*hr/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hr/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*dow/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*dow/7.0)
    return df

all_df = add_time_features(all_df)

def add_rolling(df: pd.DataFrame, window: int, targets: List[str], key_col: str) -> pd.DataFrame:
    for c in targets:
        if c in df.columns:
            roll = (
                df.groupby(key_col)[c]
                  .rolling(window=window, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
            df[f"{c}_rollmean_{window}"] = roll
    return df

all_df = add_rolling(all_df, WINDOW_SIZE, [c for c in TARGET_COLS if c in all_df.columns], SENSOR_TAG)

TIME_FEATS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ROLL_FEATS = [f"{c}_rollmean_{WINDOW_SIZE}" for c in TARGET_COLS if f"{c}_rollmean_{WINDOW_SIZE}" in all_df.columns]
FEAT_COLS = [c for c in FEATURE_BASE if c in all_df.columns] + TIME_FEATS + ROLL_FEATS
TARGETS   = [c for c in TARGET_COLS if c in all_df.columns]

print(f"Features used ({len(FEAT_COLS)}): {FEAT_COLS}")
print(f"Targets  used ({len(TARGETS)}): {TARGETS}")
if len(TARGETS) == 0:
    raise RuntimeError("None of the target columns are present in data. Check measurement names.")

# ────────────────────────────────────────────────────────────
# Train per sensor
# ────────────────────────────────────────────────────────────
def build_attn_model(input_shape: Tuple[int, int], out_dim: int) -> Model:
    inp = Input(shape=input_shape)
    x   = LSTM(64, return_sequences=True)(inp)
    x   = Dropout(0.25)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.25)(x)
    att = Attention()([x, x])
    ctx = GlobalAveragePooling1D()(att)
    out = Dense(out_dim, activation="linear")(ctx)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return m

for sid, zdf in all_df.groupby(SENSOR_TAG):
    print(f"\n=== Sensor {sid} ===")
    zdf = zdf.dropna(subset=TARGETS).reset_index(drop=True)
    if len(zdf) < (WINDOW_SIZE + HORIZON + 10):
        print(f"Skipping {sid}: not enough samples ({len(zdf)}).")
        continue

    sx, sy = MinMaxScaler(), MinMaxScaler()
    Xs = sx.fit_transform(zdf[FEAT_COLS])
    Ys = sy.fit_transform(zdf[TARGETS])

    np.save(f"scaler_X_scale_{sid}.npy", sx.scale_)
    np.save(f"scaler_X_min_{sid}.npy",   sx.min_)
    np.save(f"scaler_Y_scale_{sid}.npy", sy.scale_)
    np.save(f"scaler_Y_min_{sid}.npy",   sy.min_)

    full_len = WINDOW_SIZE + HORIZON
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        Xs, Ys, sequence_length=full_len, sequence_stride=1, sampling_rate=1, batch_size=None
    ).map(lambda x, y: (x[:WINDOW_SIZE], y)).cache()

    total = len(zdf) - full_len + 1
    val_n = max(int(total * 0.2), 1)
    train_n = max(total - val_n, 1)

    val_ds   = ds.take(val_n).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    train_ds = ds.skip(val_n).batch(BATCH_SIZE).shuffle(1024).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch  = max(train_n // BATCH_SIZE, 1)
    validation_steps = max(val_n   // BATCH_SIZE, 1)

    model = build_attn_model((WINDOW_SIZE, len(FEAT_COLS)), len(TARGETS))
    cbs = [
        ModelCheckpoint(f"best_lstm_{sid}.weights.h5", save_weights_only=True,
                        monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=f"logs/{sid}")
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=2
    )

    model.save(f"best_lstm_{sid}.keras")
    print(f"✅ Saved best_lstm_{sid}.keras")

print("\nAll sensors processed.")
