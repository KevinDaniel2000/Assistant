import os
import json
from collections import deque
import numpy as np
import paho.mqtt.client as mqtt
from kafka import KafkaProducer

try:
    from tflite_runtime.interpreter import Interpreter # type: ignore
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ----------------------------------------------------
# 📌 CONFIGURATION AND ROUTING CONSTANTS
# ----------------------------------------------------
MQTT_BROKER_HOST = "Edge-mqtt-broker"
MQTT_PORT = 1883
MQTT_TOPIC = "edge/telemetry"

KAFKA_BROKER_HOST = "Cloud-kafka:9092"
KAFKA_ALERT_TOPIC = "telemetry-anomalies"

MODEL_PATH = "/artifacts/model.tflite"
METADATA_PATH = "/artifacts/metadata.json"

# ----------------------------------------------------
# 🧠 REFACTORED ANOMALY DETECTOR ENGINE
# ----------------------------------------------------
class AnomalyDetector:
    def __init__(self, model_path: str, metadata_path: str):
        with open(metadata_path) as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]
        self.window_size = self.meta["window_size"]
        self.mean = np.array(self.meta["scaler_mean"], dtype="float32")
        self.scale = np.array(self.meta["scaler_scale"], dtype="float32")
        self.threshold = self.meta["anomaly_threshold_mse"]
        
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_detail = self.interpreter.get_input_details()[0]
        self._output_detail = self.interpreter.get_output_details()[0]
        self.buffer = deque(maxlen=self.window_size)

    def _to_vector(self, reading: dict) -> np.ndarray:
        # 🔥 FIXED: Safely bypasses 'sensor_id' and 'timestamp' root-level keys 
        # and extracts ONLY the core numerical metrics dict payload
        metrics_block = reading.get("metrics", reading)
        return np.array([metrics_block[f] for f in self.features], dtype="float32")

    def _run_tflite(self, scaled_window: np.ndarray) -> np.ndarray:
        batch = scaled_window[None, ...].astype(self._input_detail["dtype"])
        self.interpreter.set_tensor(self._input_detail["index"], batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._output_detail["index"])[0]

    def update(self, reading: dict):
        try:
            self.buffer.append(self._to_vector(reading))
        except KeyError as ke:
            print(f"⚠️ Missing expected feature key in payload: {ke}")
            return None
            
        if len(self.buffer) < self.window_size:
            return None
            
        raw_window = np.stack(self.buffer, axis=0)
        scaled_window = (raw_window - self.mean) / self.scale
        recon = self._run_tflite(scaled_window)
        mse = float(np.mean((scaled_window - recon) ** 2))
        
        return {
            "mse": mse,
            "threshold": self.threshold,
            "is_anomaly": mse > self.threshold,
            "severity": mse / self.threshold
        }

# ----------------------------------------------------
# 📡 MQTT & KAFKA NETWORK ORCHESTRATION PIPELINE
# ----------------------------------------------------
print("🧠 Initializing TFLite Runtime Anomaly Detection Client...")
detector = AnomalyDetector(MODEL_PATH, METADATA_PATH)

print(f"📭 Constructing Kafka Ingestion Producer pipeline pointing to: {KAFKA_BROKER_HOST}")
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER_HOST],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
except Exception as ke:
    print(f"❌ Kafka Connection Failure: {ke}. Running in local logging mode.")
    producer = None

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"✅ Edge Gateway linked to MQTT Broker. Monitoring topic: [{MQTT_TOPIC}]...")
    client.subscribe(MQTT_TOPIC, qos=1)

def on_message(client, userdata, msg):
    try:
        raw_payload = json.loads(msg.payload.decode("utf-8"))
        
        # Score the telemetry point through the lightweight TFLite engine
        result = detector.update(raw_payload)
        
        if result:
            if result["is_anomaly"]:
                print(f"🚨 [ANOMALY DETECTED] MSE: {result['mse']:.5f} exceeds threshold!")
                
                # 🔥 CRITICAL VALUE INCREASE: The alert payload now carries the 
                # sensor_id downstream, enabling the cloud brain to route the fix precisely!
                alert_payload = {
                    "anomaly_summary": result,
                    "telemetry_context": raw_payload
                }
                
                if producer:
                    producer.send(KAFKA_ALERT_TOPIC, alert_payload)
                    print(f"📤 Alert payload piped to Kafka topic: [{KAFKA_ALERT_TOPIC}]")
            else:
                print(f"🟢 Telemetry Normal. Window MSE: {result['mse']:.5f}")
                
    except Exception as e:
        print(f"❌ Error processing stream segment: {e}")

# Initialize and lock the continuous listener gateway
mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2) # type: ignore
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER_HOST, MQTT_PORT, keepalive=60)
mqtt_client.loop_forever()
