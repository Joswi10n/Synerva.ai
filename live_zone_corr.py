#!/usr/bin/env python3
import io
import time
from collections import deque

import avro.io
import avro.schema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kafka import KafkaConsumer

# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ["timestamp", "pm05", "pm1", "pm025", "pm04", "pm10", "occupancy"]
WINDOW_S = 30 * 60  # 30 minutes (at ~1 Hz)
ZONES    = ["Z1", "Z2", "Z3", "Z4"]
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # --- Avro + Kafka setup ---
    schema = avro.schema.parse(open("schema/sensor_event.avsc").read())
    reader = avro.io.DatumReader(writer_schema=schema)
    consumer = KafkaConsumer(
        "sensor_events",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        value_deserializer=lambda v: v
    )

    # --- ring buffers per zone ---
    buffers = {z: deque(maxlen=WINDOW_S) for z in ZONES}

    # --- interactive plotting setup ---
    plt.ion()
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes       = {z: ax for z, ax in zip(ZONES, axes_grid.flatten())}
    im_artists = {}

    # init heatmaps with identity matrices
    nvars = len(FEATURES) - 1  # drop timestamp column
    for z, ax in axes.items():
        corr0 = np.eye(nvars)
        im = ax.imshow(corr0, vmin=-1, vmax=1)
        ax.set_title(f"Zone {z}")
        ax.set_xticks(np.arange(nvars))
        ax.set_yticks(np.arange(nvars))
        ax.set_xticklabels(FEATURES[1:], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(FEATURES[1:], fontsize=8)
        im_artists[z] = im

    # shared colorbar
    cbar = fig.colorbar(im, ax=list(axes.values()), shrink=0.6, pad=0.02)
    cbar.set_label("Pearson ρ", rotation=270, labelpad=15)

    fig.suptitle(f"Live zone‐by‐zone correlations (last {WINDOW_S} samples)", fontsize=14)
    fig.canvas.draw()

    last_draw     = 0.0
    draw_interval = 1.0  # seconds

    print("📈  Live zone‐by‐zone correlation monitor running… Ctrl‐C to stop")
    try:
        for msg in consumer:
            # 1) Decode Avro + append to buffer
            ev = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
            z  = ev["zone"]
            ts = pd.to_datetime(ev["timestamp"], unit="ms")
            row = [ts] + [ev[f] for f in FEATURES[1:]]
            buffers[z].append(row)

            # 2) Check if it's time to redraw
            now = time.time()
            if now - last_draw < draw_interval:
                continue
            last_draw = now

            # 3) Recompute & update each zone's heatmap
            for zone in ZONES:
                buf = buffers[zone]
                df  = pd.DataFrame(buf, columns=FEATURES)

                if len(df) > 1:
                    corr = df.drop(columns="timestamp").corr().fillna(0)
                else:
                    corr = np.eye(nvars)

                im_artists[zone].set_data(corr)

                # update subplot title to show time window
                if len(df):
                    t0 = df["timestamp"].iloc[0].strftime("%H:%M:%S")
                    t1 = df["timestamp"].iloc[-1].strftime("%H:%M:%S")
                    axes[zone].set_title(f"Zone {zone} ({t0}–{t1})")

            # 4) Redraw
            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
