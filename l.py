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
DRAW_INT    = 60                     # redraw once per hour (adjust as needed)
TIME_WINDOW = pd.Timedelta(hours=24) # keep exactly the last 24 h
HISTORY_CSV = "raw_v2.csv"           # your 24 h CSV
# ──────────────────────────────────────────────────────────────────────────────


def load_history(csv_path):
    """
    Read raw_v2.csv and return a dict of deques, one per zone,
    where each deque holds tuples (timestamp, [feat_values…]).
    Only keep rows from the last 24 hours.
    """
    # Assume raw_v2.csv has at least these columns:
    #   timestamp (in ISO or ms)
    #   zone (one of Z1, Z2, Z3, Z4)
    #   temperature, humidity, pressure, co2, occupancy
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # If your CSV’s timestamp is in milliseconds rather than ISO, do:
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Filter to only “last 24 hours” relative to “now”
    cutoff = pd.Timestamp.now() - TIME_WINDOW
    df = df[df["timestamp"] >= cutoff].copy()
    
    # Sort by timestamp so that the deque is in chronological order
    df.sort_values("timestamp", inplace=True)
    
    # Create a deque for each zone
    buffers = {z: deque() for z in ZONES}
    
    # Iterate rows and append to the matching zone’s deque
    for _, row in df.iterrows():
        zone = row["zone"]
        if zone not in ZONES:
            continue
        ts = row["timestamp"]
        vals = [row[f] for f in FEATURES]
        buffers[zone].append((ts, vals))
    
    return buffers


def main():
    # 1) GUI backend
    matplotlib.use("TkAgg")

    # 2) Avro + Kafka setup (unchanged)
    schema = avro.schema.parse(open("schema/sensor_event.avsc").read())
    reader = avro.io.DatumReader(writer_schema=schema)
    consumer = KafkaConsumer(
        "sensor_events",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        value_deserializer=lambda v: v
    )

    # 3) Preload buffers from your 24 h CSV
    print(f"⏳  Loading 24 h of history from {HISTORY_CSV} …")
    buffers = load_history(HISTORY_CSV)
    print("✅  History loaded. Each zone buffer contains at most 24 h of samples.")

    # 4) Now build the plotting window
    plt.ion()
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes = {z: axes_grid.flatten()[i] for i, z in enumerate(ZONES)}

    # 5) Create one Line2D per (zone,feature)
    lines = {z: {} for z in ZONES}
    for z, ax in axes.items():
        for feat in FEATURES:
            ln, = ax.plot([], [], label=feat)
            lines[z][feat] = ln

        ax.set_title(f"Zone {z} (last 24 h)")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time (HH:MM)")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)
        ax.legend(loc="upper left", fontsize="small")

    fig.canvas.manager.set_window_title("Live Environment (24 h History + Live)")
    plt.show()
    last_draw = time.time()

    print("▶️  Now starting live‐stream… (press Ctrl‐C to stop)")

    try:
        # 6) First pass: draw the preloaded history so you see a complete 24 h curve
        for zone in ZONES:
            buf = buffers[zone]
            ax = axes[zone]
            if not buf:
                continue

            # Build a DataFrame from the buffer
            df = pd.DataFrame(
                {
                    "timestamp": [row[0] for row in buf],
                    **{feat: [row[1][i] for row in buf] for i, feat in enumerate(FEATURES)}
                }
            )

            # Plot each feature’s full 24 h line
            for feat in FEATURES:
                lines[zone][feat].set_data(df["timestamp"], df[feat])

            # Fix X limits to exactly last 24 h
            now_ts = pd.Timestamp.now()
            day_start = (now_ts - TIME_WINDOW).normalize() + pd.Timedelta(hours=now_ts.hour)
            # But simpler: just do now−24h → now
            ax.set_xlim(now_ts - TIME_WINDOW, now_ts)

            # Auto-scale Y to cover all historical values
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        # Draw that initial “24 h history” once
        fig.canvas.draw_idle()
        plt.pause(0.1)

        # 7) Enter the live‐stream loop: append new Kafka messages to these buffers
        for msg in consumer:
            ev = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
            z  = ev["zone"]
            if z not in ZONES:
                continue

            ts = pd.to_datetime(ev["timestamp"], unit="ms")
            vals = [ev[f] for f in FEATURES]
            buffers[z].append((ts, vals))

            # Trim anything older than 24 h
            cutoff = pd.Timestamp.now() - TIME_WINDOW
            buf = buffers[z]
            while buf and (buf[0][0] < cutoff):
                buf.popleft()

            # Throttle redraws to once per minute (or change DRAW_INT as needed)
            now_sec = time.time()
            if (now_sec - last_draw) < DRAW_INT:
                continue
            last_draw = now_sec

            # 8) Re-plot each zone’s data (last 24 h)
            for zone in ZONES:
                buf2 = buffers[zone]
                ax = axes[zone]

                if not buf2:
                    for feat in FEATURES:
                        lines[zone][feat].set_data([], [])
                    continue

                df2 = pd.DataFrame(
                    {
                        "timestamp": [row[0] for row in buf2],
                        **{feat: [row[1][i] for row in buf2] for i, feat in enumerate(FEATURES)}
                    }
                )

                # Update each feature’s line
                for feat in FEATURES:
                    lines[zone][feat].set_data(df2["timestamp"], df2[feat])

                # Fix X limits to [now–24 h, now]
                now_ts = pd.Timestamp.now()
                ax.set_xlim(now_ts - TIME_WINDOW, now_ts)

                # Autoscale only Y
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n⏹️  Live plotting stopped by user.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
