"""Training callbacks per section 6.4 of the prompt."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def make_callbacks(
    checkpoint_dir: str | Path,
    log_dir: str | Path,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-5,
) -> list[tf.keras.callbacks.Callback]:
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    log_p = Path(log_dir)
    log_p.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_insights_auc",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                cp_dir / "viva_ai_{epoch:03d}_{val_insights_auc:.4f}.keras"
            ),
            save_best_only=True,
            monitor="val_insights_auc",
            mode="max",
        ),
        tf.keras.callbacks.CSVLogger(str(log_p / "training_log.csv")),
    ]
