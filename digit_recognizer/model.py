"""CNN architecture for MNIST digit classification.

Compared to the original notebook's model (two Conv2D/MaxPooling2D blocks
with no regularization), this adds BatchNormalization for more stable/faster
training and Dropout to reduce overfitting, and trains on integer labels
with sparse_categorical_crossentropy instead of one-hot vectors.
"""

from tensorflow import keras
from keras import layers

from digit_recognizer.config import Config


def build_model(config: Config = Config()) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=config.input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config.dropout_rate),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config.dropout_rate),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(config.num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
