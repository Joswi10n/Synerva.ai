#!/usr/bin/env python3
"""
online_stream_lstm_influx.py

Same as before, but fixes the MinMax inverse‐scaling so that predictions
are the actual particle values rather than negative decimals.
"""

import os
import io
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

import avro.schema
import avro.io
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 300
HORIZON_MINUTES   = 30

BASE_FEATURES = [
    "pm05", "pm1", "pm025", "pm04", "pm10",
    "temperature", "humidity", "pressure",
    "co2", "particle_size", "occupancy",
    "wall_particles", "floor_particles", "person_particles"
]
TARGET_COLS     = ["pm05", "pm1", "pm025", "pm04", "pm10"]

BOOTSTRAP_SERVERS  = "localhost:9092"
SENSOR_TOPIC       = "sensor_events"
SCHEMA_SENSOR_PATH = "schema/sensor_event.avsc"

INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "3EvgHe4KJX_uyRjxLj6WV2xjyggQZPkwXhIlXnMEZbHxF0pWY6ItYqbW8icYdue_xemPqef4YHOnQWbK8cQZ1g=="
INFLUX_ORG    = "Sems.ai"
INFLUX_BUCKET = "Cleanroom.ai"

MODEL_FILENAME      = "best_lstm_model.keras"
SCALER_X_SCALE      = "scaler_X_scale.npy"
SCALER_X_MIN        = "scaler_X_min.npy"
SCALER_Y_SCALE      = "scaler_Y_scale.npy"
SCALER_Y_MIN        = "scaler_Y_min.npy"
FEATURE_COLUMNS_TXT = "feature_columns.txt"

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: TIME FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def compute_time_features(ts: pd.Timestamp):
    hr = ts.hour + ts.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hr / 24.0)
    hour_cos = np.cos(2 * np.pi * hr / 24.0)
    dow = ts.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    return hour_sin, hour_cos, dow_sin, dow_cos

def add_rolling_means(df: pd.DataFrame, cols: list, window: int = 60):
    for c in cols:
        df[f"{c}_rollmean_{window}m"] = (
            df.groupby("zone")[c]
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# ──────────────────────────────────────────────────────────────────────────────
# CLASS: ZoneRunner (only predict once per minute)
# ──────────────────────────────────────────────────────────────────────────────
class ZoneRunner:
    def __init__(self, zone_label, model, scaler_X, scaler_Y, feature_cols, write_api):
        self.zone         = zone_label
        self.model        = model
        self.scaler_X     = scaler_X
        self.scaler_Y     = scaler_Y
        self.feature_cols = feature_cols
        self.write_api    = write_api

        self.buffer_raw       = []
        self.last_pred_minute = None

    def append_sensor(self, record: dict):
        if record["zone"] != self.zone:
            return
        ts = pd.to_datetime(record["timestamp"], unit="ms")
        self.buffer_raw.append((ts, record))
        if len(self.buffer_raw) > WINDOW_SIZE:
            self.buffer_raw.pop(0)

    def try_predict(self):
        if len(self.buffer_raw) < WINDOW_SIZE:
            return False

        latest_ts = self.buffer_raw[-1][0]
        current_minute = latest_ts.floor("T")
        if self.last_pred_minute == current_minute:
            return False
        self.last_pred_minute = current_minute

        # Build DataFrame from buffer
        window_rows = []
        for ts, rec in self.buffer_raw:
            row = {col: rec[col] for col in BASE_FEATURES}
            row["timestamp"] = ts
            row["zone"] = self.zone
            window_rows.append(row)
        df_win = pd.DataFrame(window_rows)

        # Add time features
        tf = df_win["timestamp"].apply(compute_time_features)
        tf_arr = np.vstack(tf.values)
        df_win["hour_sin"] = tf_arr[:, 0]
        df_win["hour_cos"] = tf_arr[:, 1]
        df_win["dow_sin"]  = tf_arr[:, 2]
        df_win["dow_cos"]  = tf_arr[:, 3]

        # Add rolling means
        df_win = add_rolling_means(df_win, ["pm05", "pm1", "pm025", "pm04", "pm10"], window=60)

        # Build feature matrix
        feat_df = df_win[self.feature_cols]
        X_raw = feat_df.values.astype(np.float32)
        X_scaled = self.scaler_X.transform(X_raw)
        X_input = X_scaled.reshape((1, WINDOW_SIZE, X_scaled.shape[1]))

        # Predict (normalized)
        y_norm = self.model.predict(X_input, verbose=0)[0]

        # Correct inversion: use inverse_transform
        y_pred = self.scaler_Y.inverse_transform(y_norm.reshape(1, -1))[0]

        # Build InfluxDB Point
        last_ts_ms = int(latest_ts.timestamp() * 1000)
        pred_ts_ms = last_ts_ms + (HORIZON_MINUTES * 60_000)

        point = (
            Point("particle_prediction")
            .tag("zone", self.zone)
            .field("pm05_pred",   float(y_pred[0]))
            .field("pm1_pred",    float(y_pred[1]))
            .field("pm025_pred",  float(y_pred[2]))
            .field("pm04_pred",   float(y_pred[3]))
            .field("pm10_pred",   float(y_pred[4]))
            .tag("model_version", "lstm_online")
            .time(pred_ts_ms * 1_000_000, WritePrecision.NS)
        )

        self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

        ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pred_ts_ms / 1000))
        print(f"Wrote → zone={self.zone} time={ts_human} "
              f"pm05_pred={y_pred[0]:.1f} pm1_pred={y_pred[1]:.1f} "
              f"pm025_pred={y_pred[2]:.1f} pm04_pred={y_pred[3]:.1f} "
              f"pm10_pred={y_pred[4]:.1f}")
        return True

# ──────────────────────────────────────────────────────────────────────────────
# MAIN STREAMING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main_stream():
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"'{MODEL_FILENAME}' not found.")
    print("⏳  Loading trained model from", MODEL_FILENAME)
    model = load_model(MODEL_FILENAME)

    scaler_X = MinMaxScaler()
    scaler_X.scale_ = np.load(SCALER_X_SCALE)
    scaler_X.min_   = np.load(SCALER_X_MIN)

    scaler_Y = MinMaxScaler()
    scaler_Y.scale_ = np.load(SCALER_Y_SCALE)
    scaler_Y.min_   = np.load(SCALER_Y_MIN)

    if not os.path.exists(FEATURE_COLUMNS_TXT):
        raise FileNotFoundError(f"'{FEATURE_COLUMNS_TXT}' not found.")
    with open(FEATURE_COLUMNS_TXT, "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]

    expected_len = len(BASE_FEATURES) + 4 + len(TARGET_COLS)
    if len(feature_cols) != expected_len:
        raise ValueError(f"Expected {expected_len} feature columns, got {len(feature_cols)}")

    sensor_schema = avro.schema.parse(open(SCHEMA_SENSOR_PATH).read())
    sensor_reader = avro.io.DatumReader(writer_schema=sensor_schema)

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    runners = {
        zone: ZoneRunner(zone, model, scaler_X, scaler_Y, feature_cols, write_api)
        for zone in ["Z1", "Z2", "Z3", "Z4"]
    }

    consumer = KafkaConsumer(
        SENSOR_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v
    )

    print(f"▶️  Listening for Avro‐encoded sensor events on '{SENSOR_TOPIC}' … (Ctrl‐C to stop)")
    try:
        for msg in consumer:
            buf = io.BytesIO(msg.value)
            record = sensor_reader.read(avro.io.BinaryDecoder(buf))
            zone = record.get("zone")
            if zone in runners:
                runners[zone].append_sensor(record)
                runners[zone].try_predict()
            else:
                continue

    except KeyboardInterrupt:
        print("\n⏹️  Stopped streaming.")
    finally:
        consumer.close()
        write_api.__del__()
        influx_client.__del__()

if __name__ == "__main__":
    main_stream()
