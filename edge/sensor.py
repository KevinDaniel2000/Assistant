import json
import math
import random
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import paho.mqtt.client as mqtt

st.set_page_config(page_title="Edge Telemetry Simulator", layout="wide")

STREAM_INTERVAL_SEC = 5 # matches the real edge->cloud publish cadence

# ----------------------------------------------------
# 📌 GLOBAL INFRASTRUCTURE CONFIGURATIONS
# ----------------------------------------------------
# 🔥 NEW: Unique tracking hardware identifier for the data contract
STATIC_SENSOR_ID = "daniel-edge-001"

MQTT_BROKER_HOST = "Edge-mqtt-broker"
MQTT_BROKER_PORT = 1883
MQTT_PUBLISH_TOPIC = "edge/telemetry"

class TelemetrySimulator:
    def __init__(self):
        self.step = 0
        self.base_cpu = 15.0
        self.base_mem = 65.0
        self.base_disk = 12.0
        self.base_temp = 45.0
        self.total_drops = 0

    def generate_metrics(self, state="NORMAL"):
        self.step += 1
        time_flux = math.sin(self.step / 10.0) * 3.0
        noise = random.uniform(-1.5, 1.5)
        drops_this_tick = 0

        if state == "NORMAL":
            cpu = max(1.0, min(100.0, self.base_cpu + time_flux + noise))
            mem = max(1.0, min(100.0, self.base_mem - (time_flux * 0.2) + noise))
            disk = max(0.0, self.base_disk + (noise * 2))
            temp = max(30.0, self.base_temp + (time_flux * 0.5) + (noise * 0.2))
            if random.random() > 0.98:
                drops_this_tick = random.randint(1, 3)
        elif state == "DEGRADED":
            cpu = max(1.0, min(100.0, 75.0 + (time_flux * 2) + random.uniform(-5, 5)))
            mem = max(1.0, min(100.0, 25.0 - random.uniform(0, 5)))
            disk = max(0.0, 180.0 + random.uniform(-20, 20))
            temp = max(30.0, 72.0 + random.uniform(-2, 3))
            if random.random() > 0.85:
                drops_this_tick = random.randint(2, 8)
        elif state == "OUTAGE":
            cpu = random.uniform(98.0, 100.0)
            mem = random.uniform(1.0, 4.0)
            disk = random.uniform(450.0, 600.0)
            temp = random.uniform(88.0, 95.0)
            drops_this_tick = random.randint(15, 50)
        else:
            raise ValueError(f"Unknown state: {state}")

        self.total_drops += drops_this_tick
        now = datetime.now(timezone.utc)
        
        # 🔥 UPDATED BELOW: Injecting the unique sensor tracking keys
        return {
            "sensor_id": STATIC_SENSOR_ID,
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "state_label": state,
            "metrics": {
                "cpu_utilization_pct": round(cpu, 2),
                "memory_available_pct": round(mem, 2),
                "disk_io_wait_ms": round(disk, 2),
                "network_packet_drops": drops_this_tick,
                "network_packet_drops_total": self.total_drops,
                "hardware_temperature_c": round(temp, 2),
            },
        }

# ----------------------------------------------------
# 🔌 INITIALIZE AND PERSIST MQTT CONNECTION
# ----------------------------------------------------
if "mqtt_client" not in st.session_state:
    try:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2) # type: ignore
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        client.loop_start() 
        st.session_state.mqtt_client = client
        st.sidebar.success("⚡ Connected to local MQTT fabric!")
    except Exception as e:
        st.sidebar.error(f"❌ MQTT Connection Failure: {e}")
        st.session_state.mqtt_client = None

# Initialize base telemetry cache buffers
if "simulator" not in st.session_state:
    st.session_state.simulator = TelemetrySimulator()
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------
# LAYOUT VIEWPORTS
# ----------------------------------------------------
st.title("Edge Telemetry Simulator")
# Informing the engineer about which node ID profile is loaded
st.sidebar.info(f"📍 Machine Profile: {STATIC_SENSOR_ID}")
state_profile = st.sidebar.radio("Device state", ["NORMAL", "DEGRADED", "OUTAGE"], index=0)

col1, col2 = st.columns(2)
chart_placeholder = col1.empty()
json_placeholder = col2.empty()

# ----------------------------------------------------
# 🌀 MANAGED BACKGROUND CADENCE LOOP
# ----------------------------------------------------
@st.fragment(run_every=STREAM_INTERVAL_SEC)
def telemetry_loop(state_profile):
    payload = st.session_state.simulator.generate_metrics(state=state_profile)
    m = payload["metrics"]

    st.session_state.history.append({
        "step": st.session_state.simulator.step,
        "CPU %": m["cpu_utilization_pct"],
        "RAM Avail %": m["memory_available_pct"],
        "Temp C": m["hardware_temperature_c"],
    })
    st.session_state.history = st.session_state.history[-30:]

    if st.session_state.mqtt_client is not None:
        try:
            json_payload_str = json.dumps(payload)
            st.session_state.mqtt_client.publish(MQTT_PUBLISH_TOPIC, json_payload_str, qos=1)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to dispatch live package stream: {e}")

    df = pd.DataFrame(st.session_state.history).set_index("step")
    chart_placeholder.line_chart(df, use_container_width=True)
    json_placeholder.code(json.dumps(payload, indent=2), language="json")

telemetry_loop(state_profile)
