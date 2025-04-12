import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
from config import SAMPLE_RATE, FRAME_DURATION_MS, SEQUENCE_LENGTH, PREDICT_FRAMES

input_dir = "./raw_audio"
output_dir = "./training_frames"
os.makedirs(output_dir, exist_ok=True)

SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

def preprocess_audio(file_path):
    print(f"\n🧪 Preprocessing: {file_path}")
    samples, sample_rate = sf.read(file_path)

    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)

    if sample_rate != SAMPLE_RATE:
        samples = resample(samples, int(len(samples) * SAMPLE_RATE / sample_rate))

    samples = samples.astype(np.float32, copy=True)
    print("   🔹 Before boost: max =", np.max(np.abs(samples)))

    samples *= 10000
    peak = np.max(np.abs(samples))
    print("   🔸 After boost: max =", peak)

    if peak > 0:
        samples /= peak
        print("   ✅ After normalize: max =", np.max(np.abs(samples)))
    else:
        print("   ⚠️ Skipping silent file")
        return None

    samples *= 0.95
    samples = samples.copy()
    print("   📉 Final sample range: min =", np.min(samples), "max =", np.max(samples))
    return samples

def split_into_frames(samples, frame_size):
    total_frames = len(samples) // frame_size
    return samples[:total_frames * frame_size].reshape((-1, frame_size))

X_data = []
y_data = []

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(input_dir, filename)
        samples = preprocess_audio(filepath)
        if samples is not None:
            print(f"   🔍 SAMPLES for {filename}: min={np.min(samples)}, max={np.max(samples)}")

            frames = split_into_frames(samples, SAMPLES_PER_FRAME)

            if len(frames) > SEQUENCE_LENGTH + PREDICT_FRAMES:
                frames = frames[50:]  # Skip intro
                for i in range(len(frames) - SEQUENCE_LENGTH - PREDICT_FRAMES):
                    X_data.append(frames[i:i+SEQUENCE_LENGTH])
                    # Residual = target - last input frame
                    residual = frames[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+PREDICT_FRAMES] - frames[i+SEQUENCE_LENGTH - 1]
                    y_data.append(residual.flatten())


print(f"\n✅ Done! Total training samples: {len(X_data)}")
np.save(os.path.join(output_dir, "X.npy"), np.array(X_data))
np.save(os.path.join(output_dir, "y.npy"), np.array(y_data))

print("\n🔎 Sanity check (training tensors):")
print("   X[0] min/max:", np.min(X_data[0]), np.max(X_data[0]))
print("   y[0] min/max:", np.min(y_data[0]), np.max(y_data[0]))
print("   y[0][:10]:", y_data[0][:10])  # first 10 predicted values


print("\n🔎 Sanity check:")
print("   X[0] min/max:", np.min(X_data[0]), np.max(X_data[0]))
print("   y[0] min/max:", np.min(y_data[0]), np.max(y_data[0]))

