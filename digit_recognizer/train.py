"""Training loop with callbacks for early stopping and LR scheduling."""

import os

from tensorflow import keras

from digit_recognizer.config import Config


def train_model(model: keras.Model, train_data, val_data, config: Config = Config()):
    x_train, y_train = train_data
    x_val, y_val = val_data

    os.makedirs(config.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.model_dir, config.model_name)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.lr_reduce_patience,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    return history
