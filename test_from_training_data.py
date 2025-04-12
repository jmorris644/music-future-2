import numpy as np
import tensorflow as tf

# Load training frames from your dataset
frames = np.load("training_frames/frames_0.npy")
model = tf.keras.models.load_model("saved_model/audio_predictor_lstm.keras")

# Use first 5 frames to predict the 6th
X = np.expand_dims(frames[0:5], axis=0)
y_true = frames[5]
y_pred = model.predict(X)[0]

print("\n✅ TESTING TRAINING DATA DIRECTLY")
print("   y_true (actual) min/max:", np.min(y_true), np.max(y_true))
print("   y_pred (model output) min/max:", np.min(y_pred), np.max(y_pred))
