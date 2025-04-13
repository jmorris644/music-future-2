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

for i in range(len(frames) - SEQUENCE_LENGTH - PREDICT_FRAMES):
    window = frames[i:i + SEQUENCE_LENGTH]
    last_input = window[-1]
    pred_delta = model.predict(np.expand_dims(window,
