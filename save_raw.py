#!/usr/bin/env python3
import io
import csv
import signal
import avro.schema, avro.io
from kafka import KafkaConsumer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC             = "sensor_events"
SCHEMA_FP         = "schema/sensor_event.avsc"
HEADER = [
    "zone", "timestamp",
    "pm05", "pm1", "pm025", "pm04", "pm10",
    "temperature", "humidity", "pressure", "co2", "particle_size",
    "occupancy", "wall_particles", "floor_particles", "person_particles"
]
POLL_TIMEOUT_MS   = 500   # ms
# ──────────────────────────────────────────────────────────────────────────────

# Load Avro schema & prepare reader
schema = avro.schema.parse(open(SCHEMA_FP).read())
reader = avro.io.DatumReader(writer_schema=schema)

# Set up the Kafka consumer and seek to beginning
consumer = KafkaConsumer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    value_deserializer=lambda v: v
)
consumer.subscribe([TOPIC])
# force assignment & seek
consumer.poll(timeout_ms=0)
for tp in consumer.assignment():
    consumer.seek_to_beginning(tp)

rows = []
stop = False

def handle_sigint(signum, frame):
    global stop
    stop = True

signal.signal(signal.SIGINT, handle_sigint)

print("⏳  Collecting data… Ctrl-C to stop")
try:
    while not stop:
        records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)
        for tp, msgs in records.items():
            for msg in msgs:
                # skip empty
                if not msg.value:
                    continue
                buf = io.BytesIO(msg.value)
                try:
                    event = reader.read(avro.io.BinaryDecoder(buf))
                except Exception:
                    # skip malformed or incomplete
                    continue
                # build CSV row
                row = [event[field] for field in HEADER]
                rows.append(row)

except KeyboardInterrupt:
    pass
finally:
    consumer.close()

print(f"\n✔️  Collected {len(rows)} rows — saving to raw_v2.csv")
with open("raw_v2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print("Done.")
