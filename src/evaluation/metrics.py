"""Metric computation for the Viva AI insight model."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def per_class_metrics(
    y_true: np.ndarray,                    # (N, K)
    y_prob: np.ndarray,                    # (N, K)
    class_names: list[str],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute precision/recall/F1/AUC/AP/Brier per class. Returns dict[name → metrics]."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(class_names):
        yt = y_true[:, k]
        yp = y_prob[:, k]
        ypred = y_pred[:, k]
        n_pos = int(yt.sum())
        m: dict[str, float] = {
            "support_positive": float(n_pos),
            "support_total": float(yt.shape[0]),
            "precision": float(precision_score(yt, ypred, zero_division=0)),
            "recall": float(recall_score(yt, ypred, zero_division=0)),
            "f1": float(f1_score(yt, ypred, zero_division=0)),
            "brier": float(brier_score_loss(yt, yp)) if n_pos > 0 else float("nan"),
        }
        if n_pos > 0 and n_pos < yt.shape[0]:
            m["roc_auc"] = float(roc_auc_score(yt, yp))
            m["ap"] = float(average_precision_score(yt, yp))
        else:
            m["roc_auc"] = float("nan")
            m["ap"] = float("nan")
        out[name] = m
    return out


def macro_metrics(
    per_class: dict[str, dict[str, float]],
    metric_keys: tuple[str, ...] = ("precision", "recall", "f1", "roc_auc", "ap"),
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys:
        vals = [v[key] for v in per_class.values() if not np.isnan(v[key])]
        out[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def micro_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_roc_auc": float(roc_auc_score(y_true.flatten(), y_prob.flatten()))
        if y_true.sum() > 0 else float("nan"),
    }


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    err = y_true.flatten() - y_pred.flatten()
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    grid: tuple[float, ...] = tuple(round(x, 2) for x in np.linspace(0.1, 0.9, 17)),
) -> dict[str, float]:
    """Find threshold that maximises F1 per class on the provided set."""
    best: dict[str, float] = {}
    for k, name in enumerate(class_names):
        yt = y_true[:, k]
        yp = y_prob[:, k]
        if yt.sum() == 0:
            best[name] = 0.5
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            f1 = f1_score(yt, (yp >= t).astype(np.int32), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best[name] = best_t
    return best
