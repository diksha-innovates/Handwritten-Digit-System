"""Run inference on a single digit image with a trained model."""

import numpy as np
from PIL import Image
from tensorflow import keras


def preprocess_image(image_path: str) -> np.ndarray:
    """Load an arbitrary image file and convert it to the model's input format:
    28x28 grayscale, normalized to [0, 1], with a batch and channel axis."""
    image = Image.open(image_path).convert("L").resize((28, 28))
    array = np.array(image).astype("float32") / 255.0
    return array.reshape(1, 28, 28, 1)


def predict_digit(model: keras.Model, image_path: str):
    image = preprocess_image(image_path)
    probabilities = model.predict(image, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return digit, confidence
