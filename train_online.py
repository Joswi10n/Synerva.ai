#!/usr/bin/env python3
import os
import io
import json
import time
import pathlib
import numpy as np
import pandas as pd
from collections import deque

from dotenv import load_dotenv
from kafka import KafkaConsumer
import avro.schema, avro.io

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from influx_writer import write_ai_prediction

# ────────────────────────────────────────────────────────────
# .env loader (searches upward so it works from any CWD)
# ────────────────────────────────────────────────────────────
def _load_env():
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here / ".env",                # same folder as this file
        here.parent / ".env",         # one level up
        pathlib.Path.cwd() / ".env",  # current working directory
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            url = os.getenv("INFLUX_URL", "")
            org = os.getenv("INFLUX_ORG", "")
            bkt = os.getenv("INFLUX_BUCKET", "")
            tok = os.getenv("INFLUX_TOKEN", "")
            tok_mask = (tok[:6] + "…") if tok else "(none)"
            print(f"[env] loaded: url={url} org='{org}' bucket='{bkt}' token={tok_mask}", flush=True)
            return
    load_dotenv()  # fallback
    print("[env] .env not found in common locations; using process environment only.", flush=True)

_load_env()

# ────────────────────────────────────────────────────────────
# Config & Hyperparameters
# ────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP       = os.getenv("KAFKA_GROUP", "trainer")
TOPIC_EVENTS      = "sensor_events"
SCHEMA_PATH       = os.path.join("schema", "sensor_event.avsc")

WINDOW_BINS = 60           # look-back window (minutes)
HORIZON_SEC = 30 * 60      # forecast horizon (seconds)
TRAIN_BATCH = 8            # incremental-training batch size
VAL_HOLD    = 200          # hold-out size for periodic validation

# Must match offline training
BASE_UNITS1   = 32
BASE_UNITS2   = 64
DROPOUT_RATE  = 0.25
LEARNING_RATE = 1e-3

ZONES = ["Z1", "Z2", "Z3", "Z4"]
BASE_FEATURES = [
    "pm05","pm1","pm025","pm04","pm10",
    "temperature","humidity","pressure",
    "co2","particle_size","occupancy",
    "wall_particles","floor_particles","person_particles"
]
TARGETS = ["pm05","pm1","pm025","pm04","pm10"]

# ---- feature completeness handling -----------------------------------------
REQUIRED_FEATURES = [
    "pm05", "pm1", "pm025", "pm04", "pm10",
    "temperature", "humidity", "pressure", "co2",
    "particle_size", "occupancy",
    "wall_particles", "floor_particles", "person_particles",
]
FEATURE_DEFAULTS = {
    "pm05": 0.0, "pm1": 0.0, "pm025": 0.0, "pm04": 0.0, "pm10": 0.0,
    "temperature": 22.0, "humidity": 45.0, "pressure": 12.0, "co2": 850.0,
    "particle_size": 5.0, "occupancy": 0,
    "wall_particles": 0.0, "floor_particles": 0.0, "person_particles": 0.0,
}
# -----------------------------------------------------------------------------

# ────────────────────────────────────────────────────────────
# Feature-engineering helpers
# ────────────────────────────────────────────────────────────
def cyc(ts: pd.Timestamp):
    h   = ts.hour + ts.minute/60.0
    dow = ts.dayofweek
    return (
        np.sin(2*np.pi*h/24),
        np.cos(2*np.pi*h/24),
        np.sin(2*np.pi*dow/7),
        np.cos(2*np.pi*dow/7),
    )

def add_roll(df, cols, win=60):
    for c in cols:
        df[f"{c}_rollmean_{win}m"] = (
            df.groupby("zone")[c]
              .rolling(win, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

# ────────────────────────────────────────────────────────────
# Build attention-LSTM model (same as offline)
# ────────────────────────────────────────────────────────────
def build_attn_model(input_shape, out_dim, units1, units2, dropout, lr):
    inp = tf.keras.layers.Input(shape=input_shape)
    x   = tf.keras.layers.LSTM(units1, return_sequences=True)(inp)
    x   = tf.keras.layers.Dropout(dropout)(x)
    x   = tf.keras.layers.LSTM(units2, return_sequences=True)(x)
    x   = tf.keras.layers.Dropout(dropout)(x)
    att = tf.keras.layers.Attention()([x, x])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(att)
    out = tf.keras.layers.Dense(out_dim, activation="linear")(ctx)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m

# ────────────────────────────────────────────────────────────
# Zone class
# ────────────────────────────────────────────────────────────
class Zone:
    def __init__(self, z, model, sx, sy, feats):
        self.z     = z
        self.m     = model
        self.sx    = sx
        self.sy    = sy
        self.feats = feats
        self.buf      = deque(maxlen=WINDOW_BINS)
        self.pending  = deque()
        self.train_X  = []
        self.train_y  = []
        self.val_X    = deque(maxlen=VAL_HOLD)
        self.val_y    = deque(maxlen=VAL_HOLD)
        self.last_min = None

    def push(self, rec):
        ts = pd.to_datetime(rec["timestamp"], unit="ms").floor("min")
        self.buf.append((ts, rec))
        if len(self.buf) == WINDOW_BINS:
            y_true = np.array([rec[t] for t in TARGETS])[None, :]
            y_norm = self.sy.transform(y_true)[0]
            start_sec = int(self.buf[0][0].timestamp())
            self.pending.append((start_sec, self.build_X(), y_norm))
        if self.last_min != ts:
            self.last_min = ts
            self.predict_and_write(ts)

    def build_X(self):
        rows = []
        for ts, rec in self.buf:
            # Use defaults if a key is missing
            d = {c: rec.get(c, FEATURE_DEFAULTS[c]) for c in BASE_FEATURES}
            d["timestamp"] = ts
            d["zone"]      = self.z
            rows.append(d)
        df = pd.DataFrame(rows)
        cyc_feats = np.vstack(df["timestamp"].apply(cyc))
        for i, name in enumerate(["hour_sin","hour_cos","dow_sin","dow_cos"]):
            df[name] = cyc_feats[:, i]
        df = add_roll(df, TARGETS, WINDOW_BINS)
        X_raw = df[self.feats].values.astype(np.float32)
        return self.sx.transform(X_raw).astype(np.float32)

    def predict_and_write(self, ts):
        if len(self.buf) < WINDOW_BINS:
            return
        X = self.build_X()[None, ...]
        y_norm = self.m.predict(X, verbose=0)[0]
        y_pred = self.sy.inverse_transform(y_norm[None, :])[0]
        future_ts = ts + pd.Timedelta(seconds=HORIZON_SEC)

        print(
            f"Pred → {self.z} {future_ts:%H-%M}  "
            + "  ".join(f"{t}={v:.1f}" for t, v in zip(TARGETS, y_pred)),
            flush=True
        )

        # Fail-soft writes: never crash the loop on write errors
        for tgt, val in zip(TARGETS, y_pred):
            try:
                write_ai_prediction(
                    zone=self.z,
                    timestamp=future_ts.to_pydatetime(),
                    target=tgt,
                    predicted_value=float(val),
                    actual_value=None
                )
            except Exception as e:
                print(f"[WARN] Influx write failed for {self.z}/{tgt}: {e}", flush=True)

    def try_collect_gt(self, now_sec):
        while self.pending and (self.pending[0][0] + HORIZON_SEC) <= now_sec:
            _, Xn, yn = self.pending.popleft()
            self.train_X.append(Xn)
            self.train_y.append(yn)
            self.val_X.append(Xn)
            self.val_y.append(yn)
            if len(self.train_X) >= TRAIN_BATCH:
                self.train_step()

    def train_step(self):
        Xb = np.stack(self.train_X)
        yb = np.stack(self.train_y)
        self.train_X.clear()
        self.train_y.clear()
        hist = self.m.fit(Xb, yb, epochs=1, batch_size=TRAIN_BATCH, verbose=0)
        loss, mae = hist.history["loss"][0], hist.history["mae"][0]
        print(f"Train → {self.z}  loss={loss:.4f}  mae={mae:.4f}", flush=True)
        if len(self.val_X) == VAL_HOLD:
            Xv = np.stack(self.val_X)
            yv = np.stack(self.val_y)
            l, m = self.m.evaluate(Xv, yv, verbose=0)
            print(f"Val   → {self.z} {VAL_HOLD}ex  loss={l:.4f}  mae={m:.4f}", flush=True)
            self.val_X.clear()
            self.val_y.clear()

# ────────────────────────────────────────────────────────────
# Decoders: JSON first, then Avro fallback
# ────────────────────────────────────────────────────────────
def build_avro_reader(schema_path):
    if not os.path.exists(schema_path):
        return None, None
    schema = avro.schema.parse(open(schema_path, "r").read())
    reader = avro.io.DatumReader(schema)
    return schema, reader

def decode_message(raw_bytes, avro_reader):
    # Try JSON first
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        pass
    # Avro fallback
    schema, reader = avro_reader
    if reader is None:
        raise ValueError("Message is not JSON and no Avro schema is available.")
    try:
        decoder = avro.io.BinaryDecoder(io.BytesIO(raw_bytes))
        return reader.read(decoder)
    except Exception as e:
        raise ValueError(f"Failed to decode as Avro: {e}")

# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    avro_reader = build_avro_reader(SCHEMA_PATH)

    consumer = KafkaConsumer(
        TOPIC_EVENTS,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP,              # stable group id
        auto_offset_reset="earliest",      # replay old data if no committed offset
        enable_auto_commit=True,
        value_deserializer=lambda b: b     # keep raw; we'll decode manually
    )
    # One-time connectivity log
    try:
        topics = consumer.topics()
    except Exception:
        topics = set()
    print(f"[kafka] connected to {BOOTSTRAP_SERVERS}; topics: {topics}", flush=True)

    # Force an assignment, then seek to earliest to replay existing data
    consumer.poll(timeout_ms=1000)                         # trigger assignment
    parts = consumer.assignment()
    if parts:
        consumer.seek_to_beginning(*parts)
        print(f"[kafka] assignment: {list(parts)} — seeking to beginning", flush=True)
    else:
        print("[kafka] no assignment yet; will seek on first poll", flush=True)

    # load per-zone artifacts
    zones = {}
    for z in ZONES:
        feats_path = f"feature_columns_{z}.txt"
        if not os.path.exists(feats_path):
            raise FileNotFoundError(f"Missing {feats_path} (train_offline must run first)")

        feats = [ln.strip() for ln in open(feats_path, "r") if ln.strip()]
        sx = MinMaxScaler(); sx.scale_, sx.min_ = (
            np.load(f"scaler_X_scale_{z}.npy"),
            np.load(f"scaler_X_min_{z}.npy")
        )
        sy = MinMaxScaler(); sy.scale_, sy.min_ = (
            np.load(f"scaler_Y_scale_{z}.npy"),
            np.load(f"scaler_Y_min_{z}.npy")
        )

        keras_path   = f"best_lstm_{z}.keras"
        weights_path = f"best_lstm_{z}.weights.h5"
        if os.path.exists(keras_path):
            m = load_model(keras_path, compile=False)
            m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])
        elif os.path.exists(weights_path):
            m = build_attn_model(
                (WINDOW_BINS, len(feats)), len(TARGETS),
                BASE_UNITS1, BASE_UNITS2,
                DROPOUT_RATE, LEARNING_RATE
            )
            m.load_weights(weights_path)
        else:
            raise FileNotFoundError(f"No model found for zone {z}")

        zones[z] = Zone(z, m, sx, sy, feats)

    print("▶️ Online LSTM streaming — Ctrl-C to stop.", flush=True)
    seen = 0
    try:
        # Poll-based loop → heartbeats when idle
        while True:
            batch = consumer.poll(timeout_ms=2000)
            if not batch:
                print("…waiting for Kafka messages", flush=True)
                continue

            for tp, messages in batch.items():
                for msg in messages:
                    seen += 1
                    try:
                        rec = decode_message(msg.value, avro_reader)
                    except ValueError as e:
                        if seen <= 5:
                            print(f"⚠️  Skip undecodable message at offset {msg.offset}: {e}", flush=True)
                        continue

                    # minimal sanity
                    if not isinstance(rec, dict) or "zone" not in rec or "timestamp" not in rec:
                        if seen <= 5:
                            print(f"⚠️  Bad record shape at offset {msg.offset}: {type(rec)}", flush=True)
                        continue

                    # Ensure all expected features are present
                    for feat in REQUIRED_FEATURES:
                        if feat not in rec:
                            rec[feat] = FEATURE_DEFAULTS[feat]

                    if rec["zone"] in zones:
                        zones[rec["zone"]].push(rec)

            now_sec = int(time.time())
            for zone_obj in zones.values():
                zone_obj.try_collect_gt(now_sec)

            # heartbeat with buffers
            if seen and seen % 50 == 0:
                print(
                    "…consumed {} msgs; buffers: {}".format(
                        seen,
                        ", ".join(f"{z}:{len(zones[z].buf)}/{WINDOW_BINS}" for z in ZONES)
                    ),
                    flush=True
                )

    except KeyboardInterrupt:
        print("\n⏹️ Shutting down.", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n⏹️ Unexpected error: {e}", flush=True)
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
