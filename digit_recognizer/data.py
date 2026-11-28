"""Loading and preprocessing of the MNIST dataset."""

import numpy as np
from tensorflow import keras

from digit_recognizer.config import Config


def load_data(config: Config = Config()):
    """Load MNIST and split it into train / validation / test sets.

    The original Colab notebook validated against the test set during
    training, which leaks the test set into model selection (early
    stopping, checkpointing). Here validation data is carved out of the
    training set instead, so the test set stays truly held-out and the
    final accuracy reported on it is an unbiased estimate.
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = _preprocess_images(x_train_full)
    x_test = _preprocess_images(x_test)

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(x_train_full))
    split_at = int(len(x_train_full) * (1 - config.val_split))
    train_idx, val_idx = indices[:split_at], indices[split_at:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _preprocess_images(images: np.ndarray) -> np.ndarray:
    """Scale pixels to [0, 1] and add the single grayscale channel axis."""
    images = images.astype("float32") / 255.0
    return np.expand_dims(images, axis=-1)
