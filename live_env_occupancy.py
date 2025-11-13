#!/usr/bin/env python3
import io
import time
from collections import deque

import avro.io
import avro.schema
import matplotlib.pyplot as plt
import pandas as pd
from kafka import KafkaConsumer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# now tracking all PM channels + occupancy
FEATURES = ["pm05", "pm1", "pm025", "pm04", "pm10", "occupancy"]
WINDOW_S = 30 * 60   # keep last 30 minutes (at ~1 Hz)
ZONES    = ["Z1", "Z2", "Z3", "Z4"]
DRAW_INT = 1.0       # redraw every 1 s
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # — Avro + Kafka setup —
    schema = avro.schema.parse(open("schema/sensor_event.avsc").read())
    reader = avro.io.DatumReader(writer_schema=schema)
    consumer = KafkaConsumer(
        "sensor_events",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        value_deserializer=lambda v: v
    )

    # — ring buffers per zone: rows = [timestamp, pm05, pm1, … occupancy]
    buffers = {z: deque(maxlen=WINDOW_S) for z in ZONES}

    # — interactive plotting setup —
    plt.ion()
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes = {z: axes_grid.flatten()[i] for i, z in enumerate(ZONES)}

    # keep a Line2D for each feature in each zone
    lines = {z: {} for z in ZONES}
    for z, ax in axes.items():
        for feat in FEATURES:
            ln, = ax.plot([], [], label=feat)
            lines[z][feat] = ln
        ax.set_title(f"Zone {z}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left", fontsize="small")
        ax.grid(True)

    fig.canvas.draw()
    last_draw = time.time()

    print("▶️  Live PM & occupancy time-series per zone… Ctrl-C to stop")
    try:
        for msg in consumer:
            ev = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
            z  = ev["zone"]
            ts = pd.to_datetime(ev["timestamp"], unit="ms")
            row = [ts] + [ev[f] for f in FEATURES]
            buffers[z].append(row)

            # throttle redraws
            now = time.time()
            if now - last_draw < DRAW_INT:
                continue
            last_draw = now

            # update each zone’s lines
            for z in ZONES:
                buf = buffers[z]
                if not buf:
                    continue
                df = pd.DataFrame(buf, columns=["timestamp"] + FEATURES)
                ax = axes[z]
                for feat in FEATURES:
                    lines[z][feat].set_data(df["timestamp"], df[feat])
                # rescale
                ax.set_xlim(df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
                ax.relim()
                ax.autoscale_view()

            # redraw
            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
