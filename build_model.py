from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from config import FRAME_SIZE, SEQUENCE_LENGTH, PREDICT_FRAMES

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.2),

    Dense(FRAME_SIZE * PREDICT_FRAMES, activation='linear')  # allows full waveform energy
])

model.compile(optimizer='adam', loss='mae')
model.summary()

# Save model
import os
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/audio_predictor_lstm.keras")
