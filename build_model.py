from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from config import FRAME_SIZE, SEQUENCE_LENGTH, PREDICT_FRAMES

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
    LSTM(64),
    Dense(FRAME_SIZE * PREDICT_FRAMES)  # Predict multiple frames
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Save model
import os
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/audio_predictor_lstm.keras")
