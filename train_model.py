import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load training data
X = np.load("training_frames/X.npy")
y = np.load("training_frames/y.npy")

print(f"✅ Loaded training data: X.shape = {X.shape}, y.shape = {y.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Load the model
model = tf.keras.models.load_model("saved_model/audio_predictor_lstm.keras")

# Train the model
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop]
)

# Save updated model
model.save("saved_model/audio_predictor_lstm.keras")
print("✅ Model retrained and saved.")
