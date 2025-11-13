#!/usr/bin/env python3
import io
import time
from collections import deque

import avro.io
import avro.schema
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from kafka import KafkaConsumer

# ──────────────────────────────────────────────────────────────────────────────
FEATURES    = ["temperature", "humidity", "pressure", "co2", "occupancy"]
ZONES       = ["Z1", "Z2", "Z3", "Z4"]
DRAW_INT    = 60        # redraw once every hour
TIME_WINDOW = pd.Timedelta(hours=24)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Ensure we’re using a GUI backend that can actually open a window
    #    If you’re on WSL/Ubuntu inside VSCode, you’ll need an X server (e.g., VcXsrv) running.
    #    Here we explicitly pick TkAgg, which is the default “Tk” windowing backend.
    matplotlib.use("TkAgg")

    # 2) Avro + Kafka setup
    schema = avro.schema.parse(open("schema/sensor_event.avsc").read())
    reader = avro.io.DatumReader(writer_schema=schema)
    consumer = KafkaConsumer(
        "sensor_events",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        value_deserializer=lambda v: v
    )

    # 3) Buffers per zone: store tuples of (timestamp, [feat_values…])
    buffers = {z: deque() for z in ZONES}

    # 4) Interactive-plot setup
    plt.ion()
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    axes = {z: axes_grid.flatten()[i] for i, z in enumerate(ZONES)}

    # 5) Prepare one Line2D per (zone,feature), with “time on Y-axis”
    lines = {z: {} for z in ZONES}
    for z, ax in axes.items():
        for feat in FEATURES:
            ln, = ax.plot([], [], label=feat)
            lines[z][feat] = ln
        ax.set_title(f"Zone {z}")
        ax.set_ylabel("Time (last 24 h)")
        ax.set_xlabel("Value")
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True)
        ax.legend(loc="upper left", fontsize="small")

    # 6) Force the window to appear now
    fig.canvas.manager.set_window_title("Live Occupancy & Environment (last 24 h)")
    plt.show()      # this call blocks until the window is created, then returns immediately (since plt.ion() is on)

    last_draw = time.time()
    print("▶️ Live environment & occupancy (last 24 hours, time on Y-axis) – Ctrl-C to stop")

    try:
        for msg in consumer:
            ev = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
            z  = ev["zone"]
            if z not in ZONES:
                continue

            ts = pd.to_datetime(ev["timestamp"], unit="ms")
            values = [ev[f] for f in FEATURES]
            buffers[z].append((ts, values))

            # 7) Trim old data (> 24 h ago)
            cutoff = pd.Timestamp.now() - TIME_WINDOW
            buf = buffers[z]
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            # 8) Throttle redraws: only update once per hour
            now = time.time()
            if now - last_draw < DRAW_INT:
                continue
            last_draw = now

            # 9) Update each zone’s lines
            for z2 in ZONES:
                buf2 = buffers[z2]
                ax = axes[z2]

                if not buf2:
                    # no data yet for this zone → clear all lines
                    for feat in FEATURES:
                        lines[z2][feat].set_data([], [])
                    continue

                # Build a DataFrame from the buffer
                df = pd.DataFrame(
                    {
                        "timestamp": [row[0] for row in buf2],
                        **{
                            feat: [row[1][i] for row in buf2]
                            for i, feat in enumerate(FEATURES)
                        }
                    }
                )

                # Plot: X = feature value, Y = timestamp
                for feat in FEATURES:
                    lines[z2][feat].set_data(df[feat], df["timestamp"])

                # If all timestamps are identical, pad by ±1 minute to avoid singular xlim/ylim
                first_ts = df["timestamp"].iloc[0]
                last_ts  = df["timestamp"].iloc[-1]
                if first_ts == last_ts:
                    pad_before = first_ts - pd.Timedelta(minutes=1)
                    pad_after  = last_ts  + pd.Timedelta(minutes=1)
                    ax.set_ylim(pad_before, pad_after)
                else:
                    ax.set_ylim(cutoff, pd.Timestamp.now())

                # Autoscale X-axis (feature-value) to current data range
                ax.relim()
                ax.autoscale_view(scalex=True, scaley=False)

            # 10) Redraw the figure
            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n⏹️ Stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
