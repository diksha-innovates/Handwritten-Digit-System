import numpy as np

from digit_recognizer.config import Config
from digit_recognizer.data import load_data


def test_load_data_shapes_and_splits():
    config = Config(val_split=0.1)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(config)

    assert x_train.shape[1:] == (28, 28, 1)
    assert x_val.shape[1:] == (28, 28, 1)
    assert x_test.shape[1:] == (28, 28, 1)

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert x_test.shape[0] == 10000

    total_train = len(x_train) + len(x_val)
    assert total_train == 60000
    assert abs(len(x_val) / total_train - config.val_split) < 0.01


def test_pixel_values_are_normalized():
    (x_train, _), _, _ = load_data(Config())
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0
    assert x_train.dtype == np.float32
