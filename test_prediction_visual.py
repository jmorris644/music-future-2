import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import SEQUENCE_LENGTH, FRAME_SIZE, SAMPLE_RATE, FRAME_DURATION_MS

# Load a test audio file
audio_path = os.path.join("raw_music", os.listdir("raw_music")[0])
samples, sample_rate = sf.read(audio_path)
if samples.ndim == 2:
    samples = np.mean(samples, axis=1)
if sample_rate != SAMPLE_RATE:
    samples = resample(samples, int(len(samples) * SAMPLE_RATE / sample_rate))
samples = samples.astype(np.float32)
samples /= np.max(np.abs(samples))

frame_len = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
num_frames = len(samples) // frame_len
frames = samples[:num_frames * frame_len].reshape((-1, frame_len))

model = tf.keras.models.load_model("saved_model/audio_predictor_lstm.keras")

preds = []
actuals = []
for i in range(len(frames) - SEQUENCE_LENGTH):
    window = frames[i:i + SEQUENCE_LENGTH]
    target = frames[i + SEQUENCE_LENGTH]
    pred = model.predict(np.expand_dims(window, axis=0), verbose=0)
    preds.append(pred[0])
    actuals.append(target)

actual_flat = np.array(actuals)
preds_flat = np.array(preds)

mse = mean_squared_error(actual_flat, preds_flat)
mae = mean_absolute_error(actual_flat, preds_flat)

print(f"\n📈 Prediction Quality:")
print(f"   🔹 MSE: {mse:.6f}")
print(f"   🔸 MAE: {mae:.6f}")

# --- Plot a few predictions ---
num_to_plot = 5
boost = 1000  # Artificial boost for visibility

plt.figure(figsize=(12, 8))
for i in range(num_to_plot):
    plt.subplot(num_to_plot, 1, i + 1)

    actual_scaled = actuals[i] * boost
    predicted_scaled = preds[i] * boost

    plt.plot(actual_scaled, label="Actual (x1000)")
    plt.plot(predicted_scaled, label="Predicted (x1000)", linestyle="dashed")

    min_val = min(np.min(actual_scaled), np.min(predicted_scaled))
    max_val = max(np.max(actual_scaled), np.max(predicted_scaled))
    plt.ylim(min_val - 1, max_val + 1)

    plt.title(f"Frame {i + SEQUENCE_LENGTH}")
    plt.legend()
plt.tight_layout()
plt.show(block=True)
