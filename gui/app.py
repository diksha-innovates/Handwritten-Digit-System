"""Tkinter canvas app: draw a digit with the mouse and classify it live.

Addresses the GUI the original notebook proposed as future work ("a canvas
... to ensure interactive environment between the user and the developer")
but never built because Colab doesn't support local GUI windows.
"""

import argparse
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras

from digit_recognizer.config import Config

CANVAS_SIZE = 280          # on-screen canvas, in pixels
MODEL_INPUT_SIZE = 28      # MNIST resolution the model expects
BRUSH_RADIUS = 9


class DigitApp:
    def __init__(self, root: tk.Tk, model: keras.Model):
        self.model = model

        root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self._on_draw)

        # A same-sized PIL image mirrors the canvas strokes so we can feed
        # pixels straight to the model without a fragile screen-grab.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.result_var = tk.StringVar(value="Draw a digit, then click Predict")
        tk.Label(root, textvariable=self.result_var, font=("Helvetica", 14)).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))

        tk.Button(root, text="Predict", command=self.predict).grid(
            row=2, column=0, sticky="ew", padx=10)
        tk.Button(root, text="Clear", command=self.clear).grid(
            row=2, column=1, sticky="ew", padx=10)

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_var.set("Draw a digit, then click Predict")

    def predict(self):
        small = self.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)
        array = np.array(small).astype("float32") / 255.0
        array = array.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)

        probabilities = self.model.predict(array, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        self.result_var.set(f"Predicted: {digit}  ({confidence:.1%} confidence)")


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description="Draw a digit and classify it live.")
    parser.add_argument("--model-path", type=str,
                         default=f"{defaults.model_dir}/{defaults.model_name}")
    args = parser.parse_args()

    try:
        model = keras.models.load_model(args.model_path)
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"Could not load model at '{args.model_path}': {error}\n"
            "Train one first with: python -m scripts.train"
        )

    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
