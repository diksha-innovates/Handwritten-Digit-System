"""Evaluation helpers: held-out metrics, training curves, confusion matrix."""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


def plot_training_history(history, out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_history.png")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_confusion_matrix(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray,
                           out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "confusion_matrix.png")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, matrix[i, j], ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    report = classification_report(y_test, y_pred)
    return out_path, report
