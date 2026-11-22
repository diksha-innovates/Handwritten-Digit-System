"""CLI entry point to predict the digit in a single image file.

Usage:
    python -m scripts.predict --image path/to/digit.png
"""

import argparse

from tensorflow import keras

from digit_recognizer.config import Config
from digit_recognizer.predict import predict_digit


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Predict a digit from an image file.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                         default=f"{defaults.model_dir}/{defaults.model_name}")
    args = parser.parse_args()

    model = keras.models.load_model(args.model_path)
    digit, confidence = predict_digit(model, args.image)
    print(f"Predicted digit: {digit} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
