#!/usr/bin/env python3
"""
Continuously recompute a correlation heat-map from live Kafka sensor_events.
• keeps a rolling window of the last N rows               (ROLLING_ROWS)
• regenerates the heat-map every M new messages           (REFRESH_EVERY)
• overwrites live_corr.png in the project root
"""
import os, io, time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
import avro.schema, avro.io

# ── tuning knobs ────────────────────────────────────────────────────────────
ROLLING_ROWS   = int(os.getenv("ROLLING_ROWS",   5000))   # ≈ 20 s @ 250 Hz
REFRESH_EVERY  = int(os.getenv("REFRESH_EVERY",   500))   # redraw cadence
PNG_NAME       = os.getenv("PNG_NAME", "live_corr.png")
BOOTSTRAP      = os.getenv("BOOTSTRAP_SERVERS", "localhost:9092")
# ────────────────────────────────────────────────────────────────────────────

SCHEMA = avro.schema.parse(open("schema/sensor_event.avsc").read())
reader = avro.io.DatumReader(writer_schema=SCHEMA)

numeric_cols = [
    "timestamp", "pm01", "pm025", "pm04", "pm10",
    "temperature", "humidity", "pressure", "co2", "particle_size"
]

buf_rows = deque(maxlen=ROLLING_ROWS)
updates  = 0

consumer = KafkaConsumer(
    "sensor_events",
    bootstrap_servers=BOOTSTRAP,
    auto_offset_reset="latest",
    value_deserializer=lambda v: v,
    group_id="live_heatmap"
)

print(f"📡  Building live correlation heat-map → {PNG_NAME}  "
      f"(window={ROLLING_ROWS} rows, refresh every {REFRESH_EVERY} msgs)…")
try:
    for msg in consumer:
        # --- decode Avro payload ---
        record = reader.read(avro.io.BinaryDecoder(io.BytesIO(msg.value)))
        buf_rows.append(record)
        updates += 1

        if updates % REFRESH_EVERY == 0:
            df = pd.DataFrame(list(buf_rows))[numeric_cols]
            corr = df.corr(numeric_only=True)

            # --- draw & save heat-map ---
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            ax.set_title("Live correlation matrix  "
                         f"(last {len(df):,} rows)", pad=20)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(PNG_NAME, dpi=150)
            plt.close(fig)
            print(f"🖼️  [{time.strftime('%H:%M:%S')}] updated {PNG_NAME}")

except KeyboardInterrupt:
    print("\n⏹️  Stopped live heat-map.")
finally:
    consumer.close()
