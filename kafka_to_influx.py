#!/usr/bin/env python3
import os, io, signal, sys, time
from typing import Dict, Any
from kafka import KafkaConsumer
import avro.schema, avro.io
from influxdb_client import InfluxDBClient, Point, WriteOptions

# --- Env (adjust as needed) ---
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS", "host.docker.internal:19092")
TOPICS            = os.getenv("TOPICS", "sensor_events,cleanroom_alerts").split(",")
SCHEMA_PATH       = os.getenv("SCHEMA_PATH", os.path.join("schema", "sensor_event.avsc"))

INFLUX_URL   = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "super-long-token")   # from your docker-compose
INFLUX_ORG   = os.getenv("INFLUX_ORG", "Sems.ai")
INFLUX_BUCKET= os.getenv("INFLUX_BUCKET", "CsvData")           # <- matches your dashboard

MEASUREMENT = "SensorData"  # dashboard queries this measurement

# --- Avro setup ---
SCHEMA = avro.schema.parse(open(SCHEMA_PATH, "r").read())
reader = avro.io.DatumReader(SCHEMA)

def avro_decode(binary: bytes) -> Dict[str, Any]:
    buf = io.BytesIO(binary)
    dec = avro.io.BinaryDecoder(buf)
    return reader.read(dec)

# --- Influx setup ---
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=WriteOptions(batch_size=500,
                                                        flush_interval=1000,
                                                        jitter_interval=0,
                                                        retry_interval=2000))

# fields we’ll write (everything numeric), + use "zone" as tag
FIELD_WHITELIST = {
    "pm05","pm1","pm025","pm04","pm10",
    "temperature","humidity","pressure","co2","particle_size",
    "occupancy","wall_particles","floor_particles","person_particles",
    "alert_pm05","alert_pm1","alert_pm025","alert_pm04","alert_pm10"
}

def to_point(rec: Dict[str, Any]) -> Point:
    zone = rec.get("zone", "unknown")
    ts_ms = int(rec.get("timestamp", time.time()*1000))
    p = Point(MEASUREMENT).tag("zone", zone)
    # write numeric fields
    for k, v in rec.items():
        if k in FIELD_WHITELIST and isinstance(v, (int, float, bool)):
            # cast bool -> int for easier thresholds
            if isinstance(v, bool): v = int(v)
            p = p.field(k, float(v))
    # write at original event time
    p = p.time(ts_ms, write_precision="ms")
    return p

def main():
    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        client_id="influx-writer",
        group_id="influx-writer-g1",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda b: b  # raw bytes; we Avro-decode manually
    )

    print(f"✅ Kafka→Influx started. Consuming {TOPICS} from {BOOTSTRAP_SERVERS}, writing to {INFLUX_BUCKET}/{MEASUREMENT}")
    running = True

    def stop(*_):
        nonlocal running
        running = False
        print("\n⏹️  Stopping...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        for msg in consumer:
            if not running: break
            try:
                rec = avro_decode(msg.value)
                pt = to_point(rec)
                write_api.write(bucket=INFLUX_BUCKET, record=pt)
            except Exception as e:
                # don’t crash on bad record
                print(f"⚠️  decode/write error: {e}", file=sys.stderr)
    finally:
        write_api.flush()
        write_api.close()
        client.close()
        consumer.close()
        print("✅ Closed cleanly.")

if __name__ == "__main__":
    main()
