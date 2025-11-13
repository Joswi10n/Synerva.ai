#!/usr/bin/env python3
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque

from kafka import KafkaConsumer
import avro.schema, avro.io

# ──────────────────────────────────────────────────────────────────────────────
# Kafka + Avro setup
# ──────────────────────────────────────────────────────────────────────────────
SCHEMA = avro.schema.parse(open("schema/sensor_event.avsc").read())
reader = avro.io.DatumReader(writer_schema=SCHEMA)

consumer = KafkaConsumer(
    "sensor_events",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    value_deserializer=lambda v: v,
    group_id="daily_zone_pm"
)

# ──────────────────────────────────────────────────────────────────────────────
# in-memory buffers for a full 24 hours
# ──────────────────────────────────────────────────────────────────────────────
ZONES      = ["Z1","Z2","Z3","Z4"]
# 24 hours at roughly 1 message per second = 86400
WINDOW_LEN = 24 * 60 * 60
PARTICLES  = ["pm05", "pm1", "pm025", "pm04", "pm10"]

# Use deques with maxlen=WINDOW_LEN to keep up to 24 hours of data per zone/channel
buffers = {
    z: {
        "t":    deque(maxlen=WINDOW_LEN),
        **{p: deque(maxlen=WINDOW_LEN) for p in PARTICLES}
    }
    for z in ZONES
}

# ──────────────────────────────────────────────────────────────────────────────
# set up figure & axes (one subplot per zone)
# ──────────────────────────────────────────────────────────────────────────────
plt.ion()
fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = {z: axes_grid[i // 2][i % 2] for i, z in enumerate(ZONES)}

for ax in axes.values():
    ax.set_ylabel("Concentration (µg/m³)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=45)

YMIN, YMAX = 0, 4000

print("▶️  Live PM (24h window) per zone – Ctrl-C to stop")
try:
    for msg in consumer:
        raw = msg.value
        if not raw:
            continue
        try:
            ev = reader.read(avro.io.BinaryDecoder(io.BytesIO(raw)))
        except Exception:
            continue

        zone = ev.get("zone")
        if zone not in ZONES:
            continue

        buf = buffers[zone]
        ts = pd.to_datetime(ev["timestamp"], unit="ms")
        buf["t"].append(ts)

        for p in PARTICLES:
            buf[p].append(ev.get(p, 0.0))

        # Redraw entire 24h window each loop
        for z in ZONES:
            ax = axes[z]
            ax.clear()
            ax.set_title(f"Zone {z} (last 24h)")
            ax.set_ylabel("Concentration (µg/m³)")
            ax.set_ylim(YMIN, YMAX)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis="x", rotation=45)

            data = buffers[z]
            if not data["t"]:
                continue

            # Plot raw data faintly
            for p in PARTICLES:
                ax.plot(data["t"], data[p], alpha=0.2, linewidth=1)

            # Compute and plot rolling 10-point moving average
            df = pd.DataFrame(data, columns=["t"] + PARTICLES).set_index("t")
            ma = df.rolling(10, min_periods=1).mean()
            for p in PARTICLES:
                ax.plot(ma.index, ma[p], linewidth=2, label=p)

            ax.legend(loc="upper left", fontsize="small")

        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.1)

except KeyboardInterrupt:
    print("\n⏹️  Stopped.")
finally:
    consumer.close()
