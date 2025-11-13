#!/usr/bin/env python3
import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ──────────────────────────────────────────────────────────────
# Environment loading (safe fallback if not using .env)
# ──────────────────────────────────────────────────────────────
URL    = os.getenv("INFLUX_URL", "http://localhost:8086")
TOKEN  = os.getenv("INFLUX_TOKEN", "HyPZ4k4S6X4n75N-NQ_cwlMUjlqktjpTjSyee0DFR21ztzSIM_2TNS00ly1j3j1l9MJFi_mdC8bJVTtonVWXwA==")
ORG    = os.getenv("INFLUX_ORG", "Sems.ai")
ORG_ID = os.getenv("ORG_ID", "e5a7cb77736ded78")     # prefer orgID if set
BUCKET = os.getenv("INFLUX_BUCKET", "Cleanroom.ai")

# ──────────────────────────────────────────────────────────────
# Client init — uses orgID if available
# ──────────────────────────────────────────────────────────────
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG_ID or ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_ai_prediction(zone, timestamp, target, predicted_value, actual_value=None):
    """
    Writes an AI prediction to InfluxDB.
    Falls back to environment defaults if called from train_online.py.
    """
    try:
        point = (
            Point("ai_prediction")
            .tag("zone", zone)
            .tag("target", target)
            .field("predicted_value", float(predicted_value))
        )

        if actual_value is not None:
            point.field("actual_value", float(actual_value))

        point.time(timestamp, WritePrecision.NS)
        write_api.write(bucket=BUCKET, org=ORG_ID or ORG, record=point)
        print(f"[influx] ✅ wrote {zone}/{target} = {predicted_value:.2f}", flush=True)

    except Exception as e:
        print(f"[influx] ❌ write failed for {zone}/{target}: {e}", flush=True)

# ──────────────────────────────────────────────────────────────
# Manual test mode
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[env] url={URL} bucket={BUCKET} org={ORG} orgID={ORG_ID}")
    point = (
        Point("sensor_event")
        .tag("zone", "Z1")
        .field("pm05", 2430.5)
        .field("pm1", 590.2)
        .field("temperature", 22.1)
        .field("humidity", 45.9)
        .time(datetime.utcnow(), WritePrecision.MS)
    )
    try:
        write_api.write(bucket=BUCKET, org=ORG_ID or ORG, record=point)
        print("✔️  Successfully wrote 1 test point to InfluxDB.")
    except Exception as e:
        print("❌  Write failed:", e)
