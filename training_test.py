import numpy as np

frames = np.load("training_frames/frames_0.npy")
print("✅ DEBUG FRAME SAMPLE:")
print("   Frame 0 min/max:", np.min(frames[0]), np.max(frames[0]))
print("   Frame 5 (target) min/max:", np.min(frames[5]), np.max(frames[5]))
