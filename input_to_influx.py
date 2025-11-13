#!/usr/bin/env python3
"""
kafka_to_influx_sensor_events.py  (NO alert fields)

Listen to Kafka topic 'sensor_events', decode Avro, and push each message
to InfluxDB 2.x.  The alert_pmXX fields are deliberately excluded.

Install deps:
    pip install kafka-python avro-python3 influxdb-client
"""

import io, time
import avro.schema, avro.io
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ───────────── InfluxDB connection ─────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "ycGZJoOJD2KMYjip03BaoNnssUZVRy0F0Au9azYaMGO_uk6BHnET3C-NIcA_KgLzlRAaGrZ3gZVPzGmJaWOcXw=="
INFLUX_ORG    = "Sems.ai"
INFLUX_BUCKET = "Cleanroom.ai"

# ───────────── Kafka / Avro details ────────────
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_EVENTS      = "sensor_events"
SCHEMA_PATH       = "schema/sensor_event.avsc"

# 1) InfluxDB client
client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# 2) Avro decoder
schema        = avro.schema.parse(open(SCHEMA_PATH).read())
datum_reader  = avro.io.DatumReader(schema)

# 3) Kafka consumer
consumer = KafkaConsumer(
    TOPIC_EVENTS,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda v: v
)

print("▶️  Streaming sensor_events → InfluxDB  (alerts skipped)  Ctrl-C to stop")
try:
    for msg in consumer:
        buf   = io.BytesIO(msg.value)
        ev    = datum_reader.read(avro.io.BinaryDecoder(buf))

        pt = (
            Point("sensor_event")
            .tag("zone", ev["zone"])
            # particle concentrations
            .field("pm05",  float(ev["pm05"]))
            .field("pm1",   float(ev["pm1"]))
            .field("pm025", float(ev["pm025"]))
            .field("pm04",  float(ev["pm04"]))
            .field("pm10",  float(ev["pm10"]))
            # environment + meta
            .field("temperature",   float(ev["temperature"]))
            .field("humidity",      float(ev["humidity"]))
            .field("pressure",      float(ev["pressure"]))
            .field("co2",           float(ev["co2"]))
            .field("particle_size", float(ev["particle_size"]))
            .field("occupancy",     int(ev["occupancy"]))
            # rough source fractions
            .field("wall_particles",   float(ev["wall_particles"]))
            .field("floor_particles",  float(ev["floor_particles"]))
            .field("person_particles", float(ev["person_particles"]))
            # timestamp (ms → ns)
            .time(int(ev["timestamp"]) * 1_000_000, WritePrecision.NS)
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=pt)

        ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["timestamp"]/1000))
        print(f"✓ wrote zone={ev['zone']}  {ts_human}  pm05={ev['pm05']:.0f}")

except KeyboardInterrupt:
    print("\n⏹️  Stopped.")
finally:
    consumer.close()
    client.__del__()
