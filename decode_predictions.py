#!/usr/bin/env python3
import io
import time
from collections import deque

from kafka import KafkaConsumer
import avro.schema, avro.io

from rich.console import Console
from rich.table import Table
from rich.live import Live

# ──────────────────────────────────────────────────────────────────────────────
TOPIC      = "particle_predictions"
BOOTSTRAP  = "localhost:9092"
POLL_MS    = 500     # how long to wait each loop for new messages
DRAW_RATE  = 1.0     # seconds between table refresh
ZONES      = ["Z1","Z2","Z3","Z4"]
FIELDS     = ["pm05_pred","pm1_pred","pm025_pred","pm04_pred","pm10_pred"]
# ──────────────────────────────────────────────────────────────────────────────

# 1) Load Avro schema
schema = avro.schema.parse(open("schema/particle_prediction.avsc").read())
reader = avro.io.DatumReader(writer_schema=schema)

# 2) Start consumer with a fresh group so it only sees new messages
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP,
    group_id="decode-lstm-clean",        # NEW: fresh consumer group
    auto_offset_reset="latest",          # start at end on first run
    enable_auto_commit=True,
    value_deserializer=lambda v: v
)

# 3) Buffers to hold the latest prediction per zone
buffers = {
    z: {"t": deque(maxlen=1), **{f: deque(maxlen=1) for f in FIELDS}}
    for z in ZONES
}

console = Console()

def make_table():
    table = Table(title="Live LSTM Particle Predictions", expand=True)
    table.add_column("Zone", justify="center")
    table.add_column("Timestamp", justify="center")
    for f in FIELDS:
        label = f.replace("_pred","").upper() if "_pred" in f else f
        table.add_column(label, justify="right")

    for z in ZONES:
        buf = buffers[z]
        if not buf["t"]:
            table.add_row(z, "-", *["—"]*len(FIELDS))
        else:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(buf["t"][-1]/1000))
            vals = [f"{buf[f][-1]:.1f}" for f in FIELDS]
            table.add_row(z, ts, *vals)

    return table

with Live(make_table(), refresh_per_second=4, console=console) as live:
    console.log("▶️ Listening for LSTM predictions…")
    try:
        while True:
            # 4) Poll for new messages
            records = consumer.poll(timeout_ms=POLL_MS)
            for tp, msgs in records.items():
                for msg in msgs:
                    rec = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
                    z, ts = rec["zone"], rec["timestamp"]
                    buffers[z]["t"].append(ts)
                    for f in FIELDS:
                        buffers[z][f].append(rec.get(f, 0.0))
            # 5) Refresh table at most once per DRAW_RATE
            live.update(make_table())
            time.sleep(DRAW_RATE)
    except KeyboardInterrupt:
        console.log("⏹️ Stopped.")
    finally:
        consumer.close()
