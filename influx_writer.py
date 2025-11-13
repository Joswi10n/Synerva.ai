#!/usr/bin/env python3
"""
influx_writer.py
────────────────────────────
Simplified version for debugging / offline mode:
Instead of writing to InfluxDB, this script just prints predictions
to the terminal in a nicely formatted way.
"""

from datetime import datetime

# ─────────────────────────────────────────────
# PRINT-ONLY WRITER
# ─────────────────────────────────────────────

def write_ai_prediction(zone: str, field: str, value: float):
    """
    Mock replacement for Influx write.
    Prints out the predicted metric with timestamp.

    Args:
        zone  (str): Zone ID (e.g., "Z1", "Z2")
        field (str): Metric name (e.g., "pm05", "pm1")
        value (float): Predicted value
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [print-only] {zone}/{field} → {value:.2f}")


def write_batch_prediction(zone: str, metrics: dict):
    """
    Prints all predicted fields for a given zone together.
    This is just for convenience if you ever call it in batch form.

    Args:
        zone (str): Zone ID
        metrics (dict): Dict of {field_name: value}
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Predictions for {zone}:")
    for field, val in metrics.items():
        print(f"   ├── {field:<8} → {val:.2f}")
    print("   └── end of record\n")


# ─────────────────────────────────────────────
# Optional self-test (run standalone)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Testing influx_writer print-only mode...")
    write_ai_prediction("Z1", "pm05", 2115.7)
    write_batch_prediction("Z1", {
        "pm05": 2115.7,
        "pm1": 523.4,
        "pm025": 157.5,
        "pm04": 26.3,
        "pm10": 12.6
    })
