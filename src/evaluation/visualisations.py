"""Plot helpers for the evaluation suite (training curves, confusion matrices, etc.)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_training_curves(history: dict[str, list[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    metric_pairs = [
        ("loss", "val_loss", "Total loss"),
        ("insights_loss", "val_insights_loss", "Insight head loss"),
        ("insights_auc", "val_insights_auc", "Insight head AUC"),
    ]
    n = sum(1 for k1, k2, _ in metric_pairs if k1 in keys)
    fig, axs = plt.subplots(1, max(n, 1), figsize=(5 * max(n, 1), 4))
    if n == 1:
        axs = [axs]
    j = 0
    for k1, k2, title in metric_pairs:
        if k1 not in keys:
            continue
        ax = axs[j]
        j += 1
        ax.plot(history[k1], label="train")
        if k2 in keys:
            ax.plot(history[k2], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_class_bar(
    per_class: dict[str, dict[str, float]],
    metric: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(per_class.items(), key=lambda kv: kv[1].get(metric, 0.0))
    names = [k for k, _ in items]
    vals = [v.get(metric, 0.0) for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(names))))
    ax.barh(names, vals)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Per-class {metric}")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_for_top_classes(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: Path,
    top_k: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = y_true.sum(axis=0)
    order = np.argsort(-pos)[:top_k]
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in order:
        if pos[k] == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, k], y_prob[:, k])
        ax.plot(fpr, tpr, label=class_names[k])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — top {top_k} categories")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_for_top_classes(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: Path,
    top_k: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = y_true.sum(axis=0)
    order = np.argsort(-pos)[:top_k]
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in order:
        if pos[k] == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, k], y_prob[:, k])
        ax.plot(rec, prec, label=class_names[k])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision/Recall — top {top_k} categories")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix_micro(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yt = y_true.flatten()
    yp = (y_prob >= threshold).astype(np.int32).flatten()
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Micro confusion @ {threshold}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    bins: int = 10,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yt = y_true.flatten()
    yp = y_prob.flatten()
    edges = np.linspace(0.0, 1.0, bins + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    obs = np.zeros(bins)
    for b in range(bins):
        m = (yp >= edges[b]) & (yp < edges[b + 1] + (1e-9 if b == bins - 1 else 0))
        if m.sum() > 0:
            obs[b] = yt[m].mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.plot(midpoints, obs, "o-")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
