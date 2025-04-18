import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
import tensorflow as tf
from config import SAMPLE_RATE, FRAME_SIZE, SEQUENCE_LENGTH, PREDICT_FRAMES

output_path = "predicted_audio.wav"

# Load a test file
audio_path = os.path.join("raw_audio", os.listdir("raw_audio")[0])
samples, sample_rate = sf.read(audio_path)
if samples.ndim == 2:
    samples = np.mean(samples, axis=1)
if sample_rate != SAMPLE_RATE:
    samples = resample(samples, int(len(samples) * SAMPLE_RATE / sample_rate))
samples = samples.astype(np.float32)
samples /= np.max(np.abs(samples))
samples *= 0.95

frame_len = FRAME_SIZE
frames = samples[:len(samples) // frame_len * frame_len].reshape(-1, frame_len)
frames = frames[50:]  # skip intro like training

model = tf.keras.models.load_model("saved_model/audio_predictor_lstm.keras")

predicted_audio = []

MAX_EXPORT = 2000  # or 5000 for a short sample

for i in range(min(len(frames) - SEQUENCE_LENGTH - PREDICT_FRAMES, MAX_EXPORT)):
    window = frames[i:i + SEQUENCE_LENGTH]
    last_input = window[-1]
    pred_delta = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
    pred_frames = pred_delta.reshape(PREDICT_FRAMES, FRAME_SIZE) + last_input
    predicted_audio.append(pred_frames)

# Flatten predictions into 1D array
predicted_audio = np.concatenate(predicted_audio, axis=0)

# Normalize and write to wav
predicted_audio /= np.max(np.abs(predicted_audio))
sf.write(output_path, predicted_audio.flatten(), SAMPLE_RATE)

print(f"✅ Exported predicted audio to {output_path}")
