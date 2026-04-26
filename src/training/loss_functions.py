"""Custom losses for Viva AI training.

Includes a binary focal loss (Lin et al., 2017) toggle for the insights head
to better handle the rare insight categories noted in section 7 of the prompt.
"""

from __future__ import annotations

import tensorflow as tf


def binary_focal_loss(
    gamma: float = 2.0,
    alpha: float = 0.25,
    from_logits: bool = False,
) -> tf.keras.losses.Loss:
    """Multi-label-friendly focal loss applied element-wise."""

    class _BinaryFocalLoss(tf.keras.losses.Loss):
        def __init__(self) -> None:
            super().__init__(name="binary_focal_loss")

        def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            y_true = tf.cast(y_true, tf.float32)
            if from_logits:
                y_pred = tf.sigmoid(y_pred)
            eps = 1e-7
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
            ce = -(
                y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            )
            p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
            alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
            focal = alpha_t * tf.pow(1.0 - p_t, gamma) * ce
            return tf.reduce_mean(focal, axis=-1)

    return _BinaryFocalLoss()
