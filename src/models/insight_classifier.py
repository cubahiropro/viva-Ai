"""Viva Insight Net — the multi-output classification + regression model.

Architecture follows section 6.1 of the training prompt:
    Input(128) → Dense(256, relu, l2) → BN → Dropout(0.3)
                → Dense(128, relu, l2) → BN → Dropout(0.2)
                → Dense(64, relu)
                → 3 heads:
                    • insights:        Dense(num_classes, sigmoid)
                    • budget_risk:     Dense(1, sigmoid)
                    • mood_prediction: Dense(1, sigmoid)
"""

from __future__ import annotations

import tensorflow as tf

from insight_categories import NUM_INSIGHT_CLASSES


def build_viva_model(
    num_insight_classes: int = NUM_INSIGHT_CLASSES,
    input_dim: int = 128,
    hidden_dims: tuple[int, ...] = (128, 64, 32),
    dropout_rates: tuple[float, ...] = (0.2, 0.1),
    l2: float = 1e-4,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")

    x = tf.keras.layers.Dense(
        hidden_dims[0],
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name="trunk_dense_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="trunk_bn_1")(x)
    x = tf.keras.layers.Dropout(dropout_rates[0], name="trunk_drop_1")(x)

    x = tf.keras.layers.Dense(
        hidden_dims[1],
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name="trunk_dense_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="trunk_bn_2")(x)
    x = tf.keras.layers.Dropout(dropout_rates[1], name="trunk_drop_2")(x)

    x = tf.keras.layers.Dense(
        hidden_dims[2],
        activation="relu",
        name="trunk_dense_3",
    )(x)

    insight_output = tf.keras.layers.Dense(
        num_insight_classes, activation="sigmoid", name="insights"
    )(x)
    budget_output = tf.keras.layers.Dense(1, activation="sigmoid", name="budget_risk")(x)
    mood_output = tf.keras.layers.Dense(1, activation="sigmoid", name="mood_prediction")(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs={
            "insights": insight_output,
            "budget_risk": budget_output,
            "mood_prediction": mood_output,
        },
        name="viva_insight_net",
    )


def compile_viva_model(
    model: tf.keras.Model,
    learning_rate: float = 3e-4,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> tf.keras.Model:
    from training.loss_functions import binary_focal_loss

    insight_loss = (
        binary_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        if use_focal_loss
        else tf.keras.losses.BinaryCrossentropy()
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "insights": insight_loss,
            "budget_risk": tf.keras.losses.BinaryCrossentropy(),
            "mood_prediction": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "insights": 1.0,
            "budget_risk": 0.5,
            "mood_prediction": 0.3,
        },
        metrics={
            "insights": [
                tf.keras.metrics.AUC(name="auc", multi_label=True),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "budget_risk": [tf.keras.metrics.AUC(name="auc")],
            "mood_prediction": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model
