import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import SEQUENCE_LENGTH, FRAME_SIZE, SAMPLE_RATE, FRAME_DURATION_MS, PREDICT_FRAMES

DO_PLOT = True
MAX_PREDICTIONS = 5
SKIP_FRAMES = 50  # match training skip

# Load test audio
audio_path = os.path.join("raw_audio", os.listdir("raw_audio")[0])
samples, sample_rate = sf.read(audio_path)
if samples.ndim == 2:
    samples = np.mean(samples, axis=1)
if sample_rate != SAMPLE_RATE:
    samples = resample(samples, int(len(samples) * SAMPLE_RATE / sample_rate))
samples = samples.astype(np.float32)
peak = np.max(np.abs(samples))
if peak > 0:
    samples /= peak
samples *= 0.95

# Convert to frames
frame_len = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
num_frames = len(samples) // frame_len
frames = samples[:num_frames * frame_len].reshape((-1, frame_len))
frames = frames[SKIP_FRAMES:]

# Load model
model = tf.keras.models.load_model("saved_model/audio_predictor_lstm.keras")

preds = []
actuals = []

for i in range(min(len(frames) - SEQUENCE_LENGTH - PREDICT_FRAMES, MAX_PREDICTIONS)):
    window = frames[i:i + SEQUENCE_LENGTH]
    target = frames[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICT_FRAMES]

    pred_delta = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
    last_input = window[-1]
    pred_frames = pred_delta.reshape(PREDICT_FRAMES, FRAME_SIZE) + last_input

    preds.append(pred_frames)
    actuals.append(target)

# Flatten
actual_flat = np.array(actuals).reshape(-1, FRAME_SIZE)
preds_flat = np.array(preds).reshape(-1, FRAME_SIZE)

# Evaluate
mse = mean_squared_error(actual_flat, preds_flat)
mae = mean_absolute_error(actual_flat, preds_flat)

print(f"\n📈 Prediction Quality:")
print(f"   🔹 MSE: {mse:.6f}")
print(f"   🔸 MAE: {mae:.6f}")
print("\n🔬 Sample frame stats:")
print("   Actual[0] min/max:", np.min(actuals[0]), np.max(actuals[0]))
print("   Predicted[0] min/max:", np.min(preds[0]), np.max(preds[0]))

# Plot
if DO_PLOT:
    boost = 1000
    plt.figure(figsize=(10, 3 * PREDICT_FRAMES))
    for j in range(PREDICT_FRAMES):
        plt.subplot(PREDICT_FRAMES, 1, j + 1)
        plt.plot(actuals[0][j] * boost, label=f"Actual {j}")
        plt.plot(preds[0][j] * boost, linestyle="dashed", label=f"Predicted {j}")
        plt.legend()
    plt.tight_layout()
    plt.show()
