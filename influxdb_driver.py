#!/usr/bin/env python3
import time, os
import pandas as pd
from influxdb_client import InfluxDBClient
from Zone import Zone, BASE_FEATURES, TARGETS, WINDOW_BINS, HORIZON_SEC, TRAIN_BATCH, VAL_HOLD

# 1. CONFIG
INFLUX_URL   = "http://localhost:8086"
TOKEN        = "u2oQQGwTeRqG3q_M_wXP_A_MF-vuNV_rflbVWH1VhbjrLi9yyCnGHGfNLVS7Ja-YDKjb5xxXT9bFPMC3378jZw=="
ORG          = "EMS"
BUCKET       = "sensor"
MEASUREMENT  = "cleanroom_sensor"
POLL_INTERVAL= 10  # seconds

# 2. Init Influx client + zones
client   = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
query    = client.query_api()
zones    = {}

def write_influx_pred(zone, timestamp, values):
    from influxdb_client import Point, WritePrecision
    p = Point("sensor_forecast").tag("zone", zone).time(timestamp, WritePrecision.S)
    for k,v in values.items(): p = p.field(k, v)
    client.write_api().write(BUCKET, ORG, p)

for z in ["Z1","Z2","Z3","Z4"]:
    # load model, scalers, feats exactly like your old script
    model = load_model(f"best_lstm_{z}.keras", compile=False)
    sx, sy = MinMaxScaler(), MinMaxScaler()
    sx.scale_, sx.min_ = np.load(f"scaler_X_scale_{z}.npy"), np.load(f"scaler_X_min_{z}.npy")
    sy.scale_, sy.min_ = np.load(f"scaler_Y_scale_{z}.npy"), np.load(f"scaler_Y_min_{z}.npy")
    feats = [line.strip() for line in open(f"feature_columns_{z}.txt")]
    zones[z] = Zone(z, model, sx, sy, feats, write_influx_pred)

# 3. Poll loop
last_times = {z: None for z in zones}
while True:
    now = int(time.time() * 1e3)  # ms
    for z, zone_obj in zones.items():
        # Flux query: get all new points since last_time
        start = f"-{POLL_INTERVAL}s" if last_times[z] is None else f"{last_times[z]}ms"
        flux = f'''
          from(bucket:"{BUCKET}")
            |> range(start: {start})
            |> filter(fn: (r) => r._measurement=="{MEASUREMENT}" and r.zone=="{z}")
        '''
        tables = query.query(flux)
        recs = []
        for table in tables:
            for r in table.records:
                recs.append({**r.values, "timestamp": int(r.get_time().timestamp()*1000)})
        if not recs: 
            continue
        # feed into your zone
        for rec in sorted(recs, key=lambda x: x["timestamp"]):
            zone_obj.push_record(rec)
            last_times[z] = rec["timestamp"]
    # after pushing all zones, let each collect GT and train
    now_sec = int(time.time())
    for zone_obj in zones.values():
        zone_obj.try_collect_gt(now_sec)
    time.sleep(POLL_INTERVAL)
