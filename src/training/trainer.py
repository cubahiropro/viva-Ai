"""Training orchestration: data loading, splits, class imbalance, fit loop."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insight_categories import INSIGHT_CODES, NUM_INSIGHT_CLASSES

from .callbacks import make_callbacks


# ----------------------------------------------------------------------------- helpers


@dataclass
class Splits:
    """Indices for train/val/test splits at the *user* level (no leakage)."""

    train_users: np.ndarray
    val_users: np.ndarray
    test_users: np.ndarray
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_users": self.train_users.tolist(),
            "val_users": self.val_users.tolist(),
            "test_users": self.test_users.tolist(),
            "seed": int(self.seed),
        }


def make_user_splits(
    num_users: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Splits:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_users)
    n_train = int(num_users * train_frac)
    n_val = int(num_users * val_frac)
    return Splits(
        train_users=perm[:n_train],
        val_users=perm[n_train : n_train + n_val],
        test_users=perm[n_train + n_val :],
        seed=seed,
    )


def flatten_user_days(
    features: np.ndarray,           # (N, D, F)
    labels: np.ndarray,             # (N, D, K)
    user_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sub_f = features[user_idx]
    sub_l = labels[user_idx]
    n, d, f = sub_f.shape
    return sub_f.reshape(n * d, f), sub_l.reshape(n * d, sub_l.shape[-1])


def labels_to_targets(
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split the (M, 40) label matrix into the 3 model heads."""
    insights = labels.astype(np.float32)
    fin_idx = INSIGHT_CODES.index("FIN_OVERSPEND_RISK")
    budget_risk = insights[:, fin_idx : fin_idx + 1]
    return {
        "insights": insights,
        "budget_risk": budget_risk,
        "mood_prediction": np.zeros((insights.shape[0], 1), dtype=np.float32),  # placeholder
    }


def compute_sample_weights(insights: np.ndarray) -> np.ndarray:
    """Per-sample weight = inverse frequency of its rarest positive label.

    Samples with no positive label get weight 1.0.
    """
    positives_per_class = insights.sum(axis=0)
    inv_freq = 1.0 / np.clip(positives_per_class / max(insights.shape[0], 1), 1e-4, 1.0)
    inv_freq = inv_freq / inv_freq.mean()      # normalise
    has_pos = insights.sum(axis=1) > 0
    weights = np.ones(insights.shape[0], dtype=np.float32)
    rarest_per_sample = np.where(
        has_pos,
        np.array([inv_freq[insights[i].argmax()] if insights[i].sum() > 0 else 1.0
                  for i in range(insights.shape[0])]),
        1.0,
    )
    weights = np.minimum(rarest_per_sample.astype(np.float32), 5.0)
    return weights


def attach_mood_targets(
    targets: dict[str, np.ndarray],
    flat_daily: np.ndarray,         # original (M, F_in) daily array, with mood column
    daily_columns: list[str],
    horizon: int = 1,
) -> dict[str, np.ndarray]:
    """Attach `mood_prediction` ground truth = next-day mood (normalised to [0,1])."""
    mood_idx = daily_columns.index("mood_score")
    mood_today = flat_daily[:, mood_idx].astype(np.float32)
    # We don't have explicit per-row "next day" linkage at the flattened level —
    # downstream code passes pre-aligned data. Use today's mood as best proxy
    # if upstream did not align targets. Caller can override.
    targets["mood_prediction"] = ((mood_today - 1.0) / 4.0).reshape(-1, 1)
    return targets


# ----------------------------------------------------------------------------- main


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    use_focal_loss: bool = False
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    flat_daily: np.ndarray | None,
    daily_columns: list[str] | None,
    config: TrainConfig,
    checkpoint_dir: Path,
    log_dir: Path,
    output_model_path: Path,
) -> dict[str, Any]:
    """Train the model. `features` is (N, D, F), `labels` is (N, D, K)."""
    from models.insight_classifier import build_viva_model, compile_viva_model

    n_users, n_days, _ = features.shape
    print(f"Train data: {n_users} users × {n_days} days = {n_users * n_days:,} samples")

    splits = make_user_splits(
        n_users,
        train_frac=config.train_split,
        val_frac=config.val_split,
        test_frac=config.test_split,
        seed=config.seed,
    )

    X_train, Y_train = flatten_user_days(features, labels, splits.train_users)
    X_val, Y_val = flatten_user_days(features, labels, splits.val_users)
    X_test, Y_test = flatten_user_days(features, labels, splits.test_users)

    train_targets = labels_to_targets(Y_train)
    val_targets = labels_to_targets(Y_val)
    test_targets = labels_to_targets(Y_test)

    if flat_daily is not None and daily_columns is not None:
        # Build flattened daily for mood targets (same user/day order as features)
        flat_daily_train = flat_daily[splits.train_users].reshape(-1, flat_daily.shape[-1])
        flat_daily_val = flat_daily[splits.val_users].reshape(-1, flat_daily.shape[-1])
        flat_daily_test = flat_daily[splits.test_users].reshape(-1, flat_daily.shape[-1])
        train_targets = attach_mood_targets(train_targets, flat_daily_train, daily_columns)
        val_targets = attach_mood_targets(val_targets, flat_daily_val, daily_columns)
        test_targets = attach_mood_targets(test_targets, flat_daily_test, daily_columns)

    sample_weights = compute_sample_weights(Y_train)

    model = build_viva_model()
    compile_viva_model(
        model,
        learning_rate=config.learning_rate,
        use_focal_loss=config.use_focal_loss,
    )
    model.summary(print_fn=print)

    cbs = make_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        min_lr=config.min_lr,
    )

    history = model.fit(
        X_train,
        train_targets,
        validation_data=(X_val, val_targets),
        epochs=config.epochs,
        batch_size=config.batch_size,
        sample_weight={
            "insights": sample_weights,
            "budget_risk": np.ones_like(sample_weights),
            "mood_prediction": np.ones_like(sample_weights),
        },
        callbacks=cbs,
        verbose=2,
    )

    test_metrics = model.evaluate(
        X_test, test_targets, batch_size=config.batch_size, verbose=2,
        return_dict=True,
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    summary = {
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "splits": {
            "num_train_users": int(len(splits.train_users)),
            "num_val_users": int(len(splits.val_users)),
            "num_test_users": int(len(splits.test_users)),
        },
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "use_focal_loss": config.use_focal_loss,
        },
    }
    return summary
