"""Central place for hyperparameters and paths so scripts stay in sync."""

from dataclasses import dataclass


@dataclass
class Config:
    # Data
    val_split: float = 0.1
    seed: int = 42

    # Model
    input_shape: tuple = (28, 28, 1)
    num_classes: int = 10
    dropout_rate: float = 0.3

    # Training
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    early_stopping_patience: int = 4
    lr_reduce_patience: int = 2

    # Paths
    model_dir: str = "saved_models"
    model_name: str = "digit_model.keras"
    plots_dir: str = "artifacts"
