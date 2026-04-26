#!/usr/bin/env python3
"""CLI: train the Viva AI model.

Examples
--------
    python scripts/train.py --config config/model_config.yaml
    python scripts/train.py --epochs 5 --max-users 500    # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

# Quiet TF a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np  # noqa: E402

from config_loader import load_config  # noqa: E402
from training.trainer import TrainConfig, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Viva AI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_config.yaml",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features.npz",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "labels.npz",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=PROJECT_ROOT / "logs"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "checkpoints" / "best.keras"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument(
        "--max-users", type=int, default=None,
        help="Limit users (for fast iteration)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = TrainConfig(
        epochs=args.epochs or cfg["training"]["epochs"],
        batch_size=args.batch_size or cfg["training"]["batch_size"],
        learning_rate=args.learning_rate or cfg["training"]["learning_rate"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        reduce_lr_patience=cfg["training"]["reduce_lr_patience"],
        reduce_lr_factor=cfg["training"]["reduce_lr_factor"],
        min_lr=cfg["training"]["min_lr"],
        use_focal_loss=args.use_focal_loss or cfg["training"].get("use_focal_loss", False),
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg["data"]["random_seed"],
    )

    print("Loading features and labels...")
    features = np.load(args.features)["features"]
    labels = np.load(args.labels)["labels"]

    flat_daily = None
    daily_columns = None
    daily_path = args.synthetic_dir / "daily.npz"
    cols_path = args.synthetic_dir / "feature_columns.json"
    if daily_path.exists() and cols_path.exists():
        daily = np.load(daily_path)["daily"]
        daily_columns = json.loads(cols_path.read_text())
        # align daily with features (drop the first WINDOW_DAYS-1 days)
        from feature_engineering.feature_pipeline import WINDOW_DAYS
        flat_daily = daily[:, WINDOW_DAYS - 1 :]

    if args.max_users:
        features = features[: args.max_users]
        labels = labels[: args.max_users]
        if flat_daily is not None:
            flat_daily = flat_daily[: args.max_users]

    summary = train_model(
        features=features,
        labels=labels,
        flat_daily=flat_daily,
        daily_columns=daily_columns,
        config=train_cfg,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        output_model_path=args.output,
    )

    summary_path = args.log_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nTraining summary saved to {summary_path}")

    auc = summary["test_metrics"].get("insights_auc")
    if auc is not None:
        print(f"Test insights AUC: {auc:.4f}")
        if auc < 0.80:
            print("WARNING: AUC below 0.80 target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
