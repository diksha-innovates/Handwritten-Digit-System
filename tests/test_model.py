import numpy as np

from digit_recognizer.config import Config
from digit_recognizer.model import build_model


def test_build_model_output_shape():
    model = build_model(Config())
    dummy_input = np.zeros((2, 28, 28, 1), dtype="float32")
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (2, 10)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)


def test_model_trains_one_step_without_error():
    model = build_model(Config())
    x = np.random.rand(16, 28, 28, 1).astype("float32")
    y = np.random.randint(0, 10, size=(16,))
    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    assert "loss" in history.history
