import tensorflow as tf
from config import SEQUENCE_LENGTH, FRAME_SIZE

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FRAME_SIZE)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(FRAME_SIZE)
])

model.compile(optimizer='adam', loss='mse')
model.summary()