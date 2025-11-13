#!/usr/bin/env python3
import argparse, os, glob, datetime as dt, re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Optional Influx imports (only used if --write is passed)
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except Exception:
    InfluxDBClient = Point = WritePrecision = SYNCHRONOUS = None  # allows print-only runs w/o the package

# ---------- configuration of features/targets must match training ----------
TARGETS = ["pm1_0", "pm2_5", "pm4_0", "pm10_0"]
BASE_FIELDS = ["pm1_0","pm2_5","pm4_0","pm10_0","temperature","humidity","pressure","co2","particleSize","battery"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # time features
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute/60.0
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow"] = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7.0)

    # rolling means over 60 samples (60s cadence ⇒ 60 min) used in training
    for f in ["pm1_0","pm2_5","pm4_0","pm10_0"]:
        if f in df.columns:
            df[f"{f}_rollmean_60"] = df[f].rolling(60, min_periods=1).mean()

    return df

def get_feat_cols(df_cols):
    feat_cols = [
        "temperature","humidity","pressure","co2","particleSize","battery",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "pm1_0","pm2_5","pm4_0","pm10_0",
        "pm1_0_rollmean_60","pm2_5_rollmean_60","pm4_0_rollmean_60","pm10_0_rollmean_60",
    ]
    return [c for c in feat_cols if c in df_cols]

def load_scalers(sensor):
    sx_scale = np.load(f"scaler_X_scale_{sensor}.npy")
    sx_min   = np.load(f"scaler_X_min_{sensor}.npy")
    sy_scale = np.load(f"scaler_Y_scale_{sensor}.npy")
    sy_min   = np.load(f"scaler_Y_min_{sensor}.npy")
    return sx_scale, sx_min, sy_scale, sy_min

def scale_X(X, sx_scale, sx_min):
    return (X - sx_min) / sx_scale

def descale_Y(Yn, sy_scale, sy_min):
    return Yn * sy_scale + sy_min

def _strip_hash_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any pm* columns that end with '#' to drop the hash."""
    ren = {}
    for c in df.columns:
        m = re.match(r"^(pm(?:1_0|2_5|4_0|10_0))#$", c)
        if m:
            ren[c] = m.group(1)
    if ren:
        df = df.rename(columns=ren)
    return df

def query_last_window(client, bucket, sensor_tag, sensor_id, every, lookback_minutes, window):
    # Allow optional '#' on pm fields (your data shows pm4_0# etc.)
    fields_regex = r"^(pm1_0#?|pm2_5#?|pm4_0#?|pm10_0#?|temperature|humidity|pressure|co2|particleSize|battery)$"
    lookback = f"{max(lookback_minutes, window)}m"
    flux = f'''
from(bucket:"{bucket}")
  |> range(start: -{lookback})
  |> filter(fn: (r) => r._field =~ /{fields_regex}/ and r["{sensor_tag}"] == "{sensor_id}")
  |> aggregateWindow(every: {every}, fn: mean)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","{ '","'.join(BASE_FIELDS) }","pm1_0#","pm2_5#","pm4_0#","pm10_0#"])
  |> sort(columns: ["_time"])
  |> tail(n: {window})
'''
    try:
        q = client.query_api().query_data_frame(flux)
    except Exception as e:
        print(f"⏭️  {sensor_id}: Influx query error: {e}")
        return pd.DataFrame()

    if isinstance(q, list):
        q = pd.concat(q, ignore_index=True) if q else pd.DataFrame()
    if q is None or q.empty:
        return pd.DataFrame()

    if "_time" in q.columns:
        q = q.rename(columns={"_time":"timestamp"})
    q["timestamp"] = pd.to_datetime(q["timestamp"], utc=True)

    # normalize pm columns with trailing '#'
    q = _strip_hash_cols(q)

    cols = ["timestamp"] + [c for c in BASE_FIELDS if c in q.columns]
    return q[cols]

def write_prediction(client, bucket, measurement, sensor_tag, sensor_id, ts, pred_fields: dict):
    p = Point(measurement).tag(sensor_tag, sensor_id).time(ts, WritePrecision.NS)
    for k, v in pred_fields.items():
        p = p.field(k, float(v))
    client.write_api(write_options=SYNCHRONOUS).write(bucket=bucket, record=p)

def infer_output_shape(model, n_targets):
    out = model.output_shape
    if len(out) == 2 and out[1] == n_targets:
        return ("single", 1)
    if len(out) == 3 and out[2] == n_targets:
        return ("sequence", out[1])  # horizon
    if len(out) == 2 and out[1] % n_targets == 0:
        return ("flatseq", out[1] // n_targets)
    return ("unknown", 1)

def main():
    ap = argparse.ArgumentParser()
    # Influx connection (still needed for querying history)
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--org", required=True)
    ap.add_argument("--bucket", required=True)

    ap.add_argument("--sensor-tag", default="sensor_id")
    ap.add_argument("--aggregate-every", default="60s", help="must match training cadence")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--horizon-min", type=int, default=5, help="minutes-ahead point to report")
    ap.add_argument("--lookback-min", type=int, default=180, help="how much to pull from Influx to ensure window rows")
    ap.add_argument("--min-fresh-min", type=int, default=0, help="require last sample within N minutes (0=disable)")
    ap.add_argument("--write-measurement", default="predictions")
    ap.add_argument("--sensors", default="auto", help="'auto' uses model files; or comma list e.g. NCD1,NCD2")

    # NEW: print-only by default; pass --write to also send to Influx
    ap.add_argument("--write", action="store_true", help="also write predictions back to InfluxDB")

    args = ap.parse_args()

    # Influx client (for reading; writing only if --write)
    client = InfluxDBClient(url=args.url, token=args.token, org=args.org, timeout=60000)

    # choose sensors
    if args.sensors == "auto":
        sensors = sorted([os.path.splitext(os.path.basename(p))[0].replace("best_lstm_","")
                          for p in glob.glob("best_lstm_*.keras")])
    else:
        sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]

    wrote = 0
    for sid in sensors:
        # 1) fetch last window
        df = query_last_window(client, args.bucket, args.sensor_tag, sid,
                               args.aggregate_every, args.lookback_min, args.window)
        if df.empty or len(df) < args.window:
            print(f"⏭️  {sid}: not enough data after query ({len(df)}/{args.window}).")
            continue

        # freshness gate (optional)
        last_ts = df["timestamp"].iloc[-1]
        if args.min_fresh_min > 0:
            age_min = (pd.Timestamp.utcnow() - last_ts).total_seconds()/60.0
            if age_min > args.min_fresh_min:
                print(f"⏭️  {sid}: data too old ({age_min:.1f} min > {args.min_fresh_min}).")
                continue

        # 2) features & ordering
        df = build_features(df)
        feat_cols = get_feat_cols(df.columns)
        X = df[feat_cols].tail(args.window).to_numpy()

        # 3) model + scalers
        try:
            m = load_model(f"best_lstm_{sid}.keras")
            sx_scale, sx_min, sy_scale, sy_min = load_scalers(sid)
        except Exception as e:
            print(f"⏭️  {sid}: missing model/scalers: {e}")
            continue

        # shape check
        try:
            Xn = scale_X(X, sx_scale, sx_min)[None, ...]  # (1, window, n_feat)
        except Exception as e:
            print(f"⏭️  {sid}: scaler/feature shape mismatch: {e}")
            continue

        mode, out_h = infer_output_shape(m, len(TARGETS))
        try:
            Yn = m.predict(Xn, verbose=0)
        except Exception as e:
            print(f"⏭️  {sid}: model inference error: {e}")
            continue

        # 4) choose the minute-ahead point to report
        idx = min(max(args.horizon_min, 1), out_h) - 1  # clamp
        if mode == "single":
            y_sel_n = Yn[0]
        elif mode == "sequence":
            y_sel_n = Yn[0, idx, :]
        elif mode == "flatseq":
            y_seq = Yn.reshape(1, out_h, len(TARGETS))
            y_sel_n = y_seq[0, idx, :]
        else:
            print(f"⏭️  {sid}: unknown model output shape {m.output_shape}.")
            continue

        try:
            y_sel = descale_Y(y_sel_n, sy_scale, sy_min)
        except Exception as e:
            print(f"⏭️  {sid}: descale error: {e}")
            continue

        fields = {f"{t}_pred": float(v) for t, v in zip(TARGETS, y_sel)}
        pred_ts = last_ts.to_pydatetime() + dt.timedelta(minutes=args.horizon_min)

        # ---- PRINT to terminal (always) ----
        nice = "  ".join(f"{k}={v:.2f}" for k, v in fields.items())
        print(f"Pred → {sid}  {pred_ts.strftime('%Y-%m-%d %H:%M')}  {nice}")

        # ---- OPTIONAL: write back if --write ----
        if args.write:
            if InfluxDBClient is None:
                print(f"⏭️  {sid}: --write requested but influxdb_client is not installed.")
            else:
                try:
                    p = Point(args.write_measurement).tag(args.sensor_tag, sid).time(pred_ts, WritePrecision.NS)
                    for k, v in fields.items():
                        p = p.field(k, float(v))
                    client.write_api(write_options=SYNCHRONOUS).write(bucket=args.bucket, record=p)
                    wrote += 1
                    print(f"✅ {sid}: wrote prediction → {fields}")
                except Exception as e:
                    print(f"⏭️  {sid}: write error: {e}")

    if args.write:
        print(f"\nDone. Wrote {wrote} prediction points to measurement '{args.write_measurement}'.")
    client.close()

if __name__ == "__main__":
    main()
