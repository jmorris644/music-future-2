import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import SEQUENCE_LENGTH, FRAME_SIZE
from build_model import model
from tensorflow.keras.callbacks import EarlyStopping

input_dir = "./training_frames"
X_data, y_data = [], []

# Load and window training data
for file in os.listdir(input_dir):
    if file.endswith(".npy"):
        frames = np.load(os.path.join(input_dir, file))
        for i in range(len(frames) - SEQUENCE_LENGTH):
            X_data.append(frames[i:i+SEQUENCE_LENGTH])
            y_data.append(frames[i+SEQUENCE_LENGTH])

X = np.array(X_data)
y = np.array(y_data)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/audio_predictor_lstm.keras")
print("✅ Model saved to saved_model/audio_predictor_lstm.keras")