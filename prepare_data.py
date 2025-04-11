import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
from config import SAMPLE_RATE, FRAME_DURATION_MS

input_dir = "./raw_music"              # <<-- update to match your folder name
output_dir = "./training_frames"
os.makedirs(output_dir, exist_ok=True)

SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

def preprocess_audio(file_path):
    samples, sample_rate = sf.read(file_path)

    # Stereo → mono
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)

    # Resample to 11025 Hz
    if sample_rate != SAMPLE_RATE:
        samples = resample(samples, int(len(samples) * SAMPLE_RATE / sample_rate))

    samples = samples.astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    else:
        print(f"⚠️ Skipping silent or corrupt file: {file_path}")
        return None

    return samples

def split_into_frames(samples, frame_size):
    total_frames = len(samples) // frame_size
    return samples[:total_frames * frame_size].reshape((-1, frame_size))

file_counter = 0
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        print(f"🎧 Processing {filename}...")
        samples = preprocess_audio(filepath)
        if samples is not None:
            frames = split_into_frames(samples, SAMPLES_PER_FRAME)
            np.save(os.path.join(output_dir, f"frames_{file_counter}.npy"), frames)
            file_counter += 1

print(f"\n✅ Done! Processed and saved {file_counter} file(s) to {output_dir}")
