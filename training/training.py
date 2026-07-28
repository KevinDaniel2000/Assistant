"""
Train a Conv1D autoencoder for grid-sensor performance degradation detection.

Approach: unsupervised reconstruction-based anomaly detection.
- The mock dataset contains ONLY healthy/steady-state telemetry (confirmed via EDA:
  all four core features are stationary across the full 50k rows, no injected
  anomaly regime). That's exactly the right shape of data for an autoencoder:
  train it to reconstruct "normal", then flag high reconstruction error at
  inference time as degradation/outage.
- `network_packet_drops` is excluded from training. It correlates 0.9996 with row
  index in the mock data (a near-linear ramp from 0->1976), which looks like a
  generator artifact rather than real bursty packet-loss behavior. Training on it
  would teach the model "drops increase with time since dataset start", which is
  meaningless on a live stream. Flip INCLUDE_PACKET_DROPS to True once that column
  is regenerated with a realistic distribution (e.g. mostly-zero + Poisson bursts).
- Conv1D (not LSTM) by design: this model needs to convert cleanly to TFLite and
  run on a small edge gateway. Conv1D quantizes and runs faster on CPU than
  recurrent layers, at negligible cost in modeling power for a 12-step window.
"""

import json, os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.parquet")
INCLUDE_PACKET_DROPS = True  # see module docstring
WINDOW_SIZE = 12              # steps of context per window (~60s at 5s cadence)
STRIDE = 1
VAL_FRACTION = 0.15
BOTTLENECK_DIM = 8
BATCH_SIZE = 128
EPOCHS = 50
THRESHOLD_PERCENTILE = 99.5   # of TRAIN reconstruction error -> anomaly cutoff
SEED = 42

ALL_FEATURES = [
    "cpu_utilization_pct",
    "memory_available_pct",
    "disk_io_wait_ms",
    "network_packet_drops",
    "hardware_temperature_c",
]
FEATURES = ALL_FEATURES if INCLUDE_PACKET_DROPS else [
    f for f in ALL_FEATURES if f != "network_packet_drops"
]

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load + sort
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
data = df[FEATURES].values.astype("float32")
print(f"Loaded {len(df)} rows, {len(FEATURES)} features: {FEATURES}")

# ---------------------------------------------------------------------------
# 2. Scale (fit on ALL data here since it's confirmed all-healthy;
#    save the scaler params as plain JSON so the edge gateway, which may not
#    run Python/sklearn, can reimplement `(x - mean) / scale` trivially)
# ---------------------------------------------------------------------------
scaler = StandardScaler().fit(data)
data_scaled = scaler.transform(data)

# ---------------------------------------------------------------------------
# 3. Build overlapping sliding windows
# ---------------------------------------------------------------------------
def make_windows(arr, window, stride):
    n = (len(arr) - window) // stride + 1
    windows = np.empty((n, window, arr.shape[1]), dtype="float32")
    for i in range(n):
        start = i * stride
        windows[i] = arr[start:start + window]
    return windows

windows = make_windows(data_scaled, WINDOW_SIZE, STRIDE)
print(f"Built {windows.shape[0]} windows of shape {windows.shape[1:]}")

# Chronological split (last VAL_FRACTION of windows held out) -- realistic
# for time series even though this particular set is stationary.
n_val = int(len(windows) * VAL_FRACTION)
train_windows = windows[:-n_val]
val_windows = windows[-n_val:]
print(f"Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

# ---------------------------------------------------------------------------
# 4. Model: Conv1D autoencoder
# ---------------------------------------------------------------------------
n_features = len(FEATURES)
inputs = tf.keras.Input(shape=(WINDOW_SIZE, n_features), name="telemetry_window")

x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.Conv1D(8, 3, strides=2, padding="same", activation="relu")(x)  # 12 -> 6
x = tf.keras.layers.Flatten()(x)
bottleneck = tf.keras.layers.Dense(BOTTLENECK_DIM, activation="relu", name="bottleneck")(x)

x = tf.keras.layers.Dense(6 * 8, activation="relu")(bottleneck)
x = tf.keras.layers.Reshape((6, 8))(x)
x = tf.keras.layers.Conv1DTranspose(8, 3, strides=2, padding="same", activation="relu")(x)  # 6 -> 12
outputs = tf.keras.layers.Conv1D(n_features, 3, padding="same", activation="linear", name="reconstruction")(x)

model = tf.keras.Model(inputs, outputs, name="grid_telemetry_autoencoder")
model.compile(optimizer="adam", loss="mse")
model.summary()

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=6, restore_best_weights=True
)
history = model.fit(
    train_windows, train_windows,
    validation_data=(val_windows, val_windows),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)

# ---------------------------------------------------------------------------
# 6. Reconstruction error -> anomaly threshold (computed on TRAIN data only,
#    so the threshold reflects "normal" variance, not leaked val performance)
# ---------------------------------------------------------------------------
train_recon = model.predict(train_windows, verbose=0)
train_mse = np.mean(np.square(train_windows - train_recon), axis=(1, 2))
threshold = float(np.percentile(train_mse, THRESHOLD_PERCENTILE))

val_recon = model.predict(val_windows, verbose=0)
val_mse = np.mean(np.square(val_windows - val_recon), axis=(1, 2))

print(f"\nTrain reconstruction MSE: mean={train_mse.mean():.5f} std={train_mse.std():.5f}")
print(f"Val   reconstruction MSE: mean={val_mse.mean():.5f} std={val_mse.std():.5f}")
print(f"Anomaly threshold (train p{THRESHOLD_PERCENTILE}): {threshold:.5f}")
print(f"Val windows flagged as anomalous at this threshold: {(val_mse > threshold).sum()} / {len(val_mse)}")

# ---------------------------------------------------------------------------
# 7. Save Keras model, convert to TFLite, save metadata
# ---------------------------------------------------------------------------
model.save("autoencoder.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"\nTFLite model size: {len(tflite_model) / 1024:.1f} KB")

metadata = {
    "features": FEATURES,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "scaler_mean": scaler.mean_.tolist(), # type: ignore
    "scaler_scale": scaler.scale_.tolist(), # type: ignore
    "anomaly_threshold_mse": threshold,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "train_mse_mean": float(train_mse.mean()),
    "train_mse_std": float(train_mse.std()),
    "notes": "network_packet_drops excluded from features (generator artifact, see script docstring)",
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved: autoencoder.keras, model.tflite, metadata.json")