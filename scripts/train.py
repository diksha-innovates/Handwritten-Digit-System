"""CLI entry point to train the digit classifier.

Usage:
    python -m scripts.train --epochs 20 --batch-size 128
"""

import argparse
import dataclasses

from digit_recognizer.config import Config
from digit_recognizer.data import load_data
from digit_recognizer.evaluate import evaluate_model, plot_training_history
from digit_recognizer.model import build_model
from digit_recognizer.train import train_model


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train the MNIST digit classifier.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--val-split", type=float, default=defaults.val_split)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--model-dir", type=str, default=defaults.model_dir)
    parser.add_argument("--plots-dir", type=str, default=defaults.plots_dir)
    args = parser.parse_args()

    return dataclasses.replace(
        defaults,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        seed=args.seed,
        model_dir=args.model_dir,
        plots_dir=args.plots_dir,
    )


def main():
    config = parse_args()

    print("Loading and preprocessing MNIST...")
    train_data, val_data, test_data = load_data(config)
    print(f"Train: {train_data[0].shape}, Val: {val_data[0].shape}, Test: {test_data[0].shape}")

    model = build_model(config)
    model.summary()

    history = train_model(model, train_data, val_data, config)

    plot_path = plot_training_history(history, config.plots_dir)
    print(f"Saved training curves to {plot_path}")

    test_loss, test_accuracy = evaluate_model(model, *test_data)
    print(f"Held-out test loss: {test_loss:.4f}")
    print(f"Held-out test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
