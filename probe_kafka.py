from kafka import KafkaConsumer

BOOT = "host.docker.internal:19092"
TOPIC = "sensor_events"

print(f"Connecting to {BOOT} topic {TOPIC}")
try:
    c = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOT,
        group_id="probe",
        auto_offset_reset="earliest",
        consumer_timeout_ms=3000  # 3 sec timeout
    )
    print("Connected. Known topics:", c.topics())
    n = 0
    for m in c:
        n += 1
        if n <= 5:
            print("Msg", n, "offset", m.offset, "len", len(m.value))
    if n == 0:
        print("No messages received in 3s — topic empty or producer not running.")
    c.close()
except Exception as e:
    import traceback
    traceback.print_exc()
