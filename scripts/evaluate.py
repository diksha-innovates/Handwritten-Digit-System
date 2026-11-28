"""CLI entry point to evaluate a saved model on the held-out test set.

Usage:
    python -m scripts.evaluate --model-path saved_models/digit_model.keras
"""

import argparse

from tensorflow import keras

from digit_recognizer.config import Config
from digit_recognizer.data import load_data
from digit_recognizer.evaluate import evaluate_model, plot_confusion_matrix


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Evaluate a trained digit classifier.")
    parser.add_argument("--model-path", type=str,
                         default=f"{defaults.model_dir}/{defaults.model_name}")
    parser.add_argument("--plots-dir", type=str, default=defaults.plots_dir)
    args = parser.parse_args()

    _, _, (x_test, y_test) = load_data(defaults)
    model = keras.models.load_model(args.model_path)

    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    plot_path, report = plot_confusion_matrix(model, x_test, y_test, args.plots_dir)
    print(f"Saved confusion matrix to {plot_path}")
    print(report)


if __name__ == "__main__":
    main()
