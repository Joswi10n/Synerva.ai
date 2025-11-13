# export_sensor.py
from influxdb_client import InfluxDBClient

# ——— CONFIG ———
url    = "http://localhost:8086"
token  = "p6rG-Ia0utGTGr2cyJ-fn6oUQeFusyqFl4wSIXk8GQwitHZ-cxUDhHmKJOl_UKTFKU80B6s76NwIH5EsAu13kQ=="
org    = "EMS"
bucket = "sensor"

# ——— FLUX QUERY ———
flux = f'from(bucket:"{bucket}") |> range(start: 0)'

# ——— RUN & EXPORT ———
client   = InfluxDBClient(url=url, token=token, org=org)
response = client.query_api().query_raw(flux, org=org)

# write the raw bytes out
with open("sensor_all_data.csv", "wb") as f:
    f.write(response.read())
