import numpy as np

# Simulate what we expect from preprocess_audio()
samples = np.linspace(-0.95, 0.93, 11025 * 2, dtype=np.float32)
print("✅ Simulated samples min/max:", np.min(samples), np.max(samples))

# Split into frames (20 ms each at 11025 Hz = 220 samples)
frame_len = 220
total_frames = len(samples) // frame_len
frames = samples[:total_frames * frame_len].reshape((-1, frame_len))

print("✅ Frame 0 min/max:", np.min(frames[0]), np.max(frames[0]))
