from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import FRAME_SIZE, SEQUENCE_LENGTH, PREDICT_FRAMES

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(FRAME_SIZE * PREDICT_FRAMES, activation='tanh')  # Use tanh for waveform shape
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Save initial model
import os
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/audio_predictor_lstm.keras")
