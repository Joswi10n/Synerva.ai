#!/usr/bin/env python3
"""
send_fake.py – stream fake cleanroom sensor data into Redpanda/Kafka
"""

import os, json, random, time, io
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
# Use comma-separated list in BOOTSTRAP_SERVERS, e.g. "localhost:19092,localhost:9092"
BOOTSTRAP_SERVERS = os.getenv("BOOTSTRAP_SERVERS", "localhost:19092")
TOPIC_EVENTS = os.getenv("TOPIC_EVENTS", "sensor_events")

ZONES = ["Z1", "Z2", "Z3", "Z4"]

LIMITS = {
    "pm05": 3520,
    "pm1": 832,
    "pm025": 150,
    "pm04": 30,
    "pm10": 18,
}

def _ks(k):
    return k.encode("utf-8") if isinstance(k, str) else k

def _vs(v):
    if isinstance(v, (bytes, bytearray)):
        return v
    if not isinstance(v, str):
        v = str(v)
    return v.encode("utf-8")

# ─────────────────────────────────────────────
# Kafka Producer
# ─────────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers=[s.strip() for s in BOOTSTRAP_SERVERS.split(",") if s.strip()],
    acks="all",
    linger_ms=50,                 # small batching without much latency
    retries=10,                   # retry transient errors
    request_timeout_ms=15000,     # 15s request timeout
    max_in_flight_requests_per_connection=5,
    key_serializer=_ks,
    value_serializer=_vs,
    security_protocol="PLAINTEXT",
)

print(f"▶️  Streaming cleanroom data → topic='{TOPIC_EVENTS}' on {BOOTSTRAP_SERVERS} (Ctrl-C to stop)")

def on_delivery(rec_metadata, exc):
    if exc is not None:
        print(f"[producer] ❌ delivery failed: {exc}")
    else:
        # Uncomment to see every ack:
        # print(f"[producer] ✅ {rec_metadata.topic}@{rec_metadata.partition} offset={rec_metadata.offset}")
        pass

# ─────────────────────────────────────────────
# Data loop
# ─────────────────────────────────────────────
try:
    while True:
        ts = int(time.time() * 1000)  # ms epoch
        futures = []
        for zone in ZONES:
            record = {
                "zone": zone,
                "timestamp": ts,
                "pm05": random.uniform(1500, 2200),
                "pm1": random.uniform(400, 800),
                "pm025": random.uniform(100, 200),
                "pm04": random.uniform(10, 35),
                "pm10": random.uniform(5, 20),
                "temperature": random.uniform(20, 23),
                "humidity": random.uniform(40, 50),
                "pressure": random.uniform(11, 13),
                "co2": random.uniform(800, 900),
                "particle_size": random.uniform(4, 6),
                "occupancy": random.choice([0, 1]),
                "wall_particles": random.uniform(1100, 1600),
                "floor_particles": random.uniform(650, 900),
                "person_particles": random.uniform(450, 650),
            }
            # alerts
            for k in ["pm05", "pm1", "pm025", "pm04", "pm10"]:
                record[f"alert_{k}"] = record[k] > LIMITS[k]
            record["limits"] = LIMITS

            buffer = io.StringIO()
            json.dump(record, buffer)
            fut = producer.send(TOPIC_EVENTS, key=zone, value=buffer.getvalue())
            # optional: attach callback for visibility
            fut.add_callback(lambda md, z=zone: on_delivery(md, None))
            fut.add_errback(lambda exc: on_delivery(None, exc))
            futures.append(fut)

            print(record)

        # flush once per tick so we don’t block every record
        for f in futures:
            try:
                f.get(timeout=5)
            except KafkaError as e:
                print(f"[producer] ❌ send failed: {e}")

        # small wait between ticks
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⏹️  Stopped streaming.")
finally:
    producer.flush(10)
    producer.close()
