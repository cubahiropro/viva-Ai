#!/usr/bin/env python3
"""CLI: evaluate a trained Viva AI model and produce metrics + plots + report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from config_loader import load_config  # noqa: E402
from insight_categories import INSIGHT_CODES  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    find_optimal_thresholds,
    macro_metrics,
    micro_metrics,
    per_class_metrics,
    regression_metrics,
)
from evaluation.report import (  # noqa: E402
    build_model_card,
    build_pdf_report,
    build_readme_summary,
)
from evaluation.visualisations import (  # noqa: E402
    plot_calibration,
    plot_confusion_matrix_micro,
    plot_per_class_bar,
    plot_pr_for_top_classes,
    plot_roc_for_top_classes,
    plot_training_curves,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Viva AI model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "model_config.yaml")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "processed" / "features.npz")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "processed" / "labels.npz")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "checkpoints" / "best.keras")
    parser.add_argument("--training-summary", type=Path,
                        default=PROJECT_ROOT / "logs" / "training_summary.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "evaluation_artifacts")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"

    cfg = load_config(args.config)
    seed = cfg["data"]["random_seed"]
    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]

    print(f"Loading features and labels...")
    features = np.load(args.features)["features"]
    labels = np.load(args.labels)["labels"]

    n_users = features.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_users)
    n_train = int(n_users * train_split)
    n_val = int(n_users * val_split)
    test_users = perm[n_train + n_val :]

    X_test = features[test_users].reshape(-1, features.shape[-1])
    Y_test = labels[test_users].reshape(-1, labels.shape[-1])

    import tensorflow as tf
    print(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print("Running predictions...")
    preds = model.predict(X_test, batch_size=512, verbose=0)
    insights_prob = preds["insights"] if isinstance(preds, dict) else preds[0]
    budget_prob = preds["budget_risk"] if isinstance(preds, dict) else preds[1]
    mood_pred = preds["mood_prediction"] if isinstance(preds, dict) else preds[2]

    print("Computing metrics...")
    pc = per_class_metrics(Y_test, insights_prob, INSIGHT_CODES, threshold=args.threshold)
    macro = macro_metrics(pc)
    micro = micro_metrics(Y_test, insights_prob, threshold=args.threshold)

    optimal_thresholds = find_optimal_thresholds(Y_test, insights_prob, INSIGHT_CODES)
    pc_opt = per_class_metrics(
        Y_test, insights_prob, INSIGHT_CODES, threshold=args.threshold,
    )
    # Re-run with per-class thresholds:
    yp = np.zeros_like(insights_prob, dtype=np.int32)
    for k, name in enumerate(INSIGHT_CODES):
        yp[:, k] = (insights_prob[:, k] >= optimal_thresholds[name]).astype(np.int32)
    micro_opt = {
        "micro_f1_optimal": float(np.mean([pc[name]["f1"] for name in INSIGHT_CODES])),
    }

    fin_idx = INSIGHT_CODES.index("FIN_OVERSPEND_RISK")
    budget_metrics = per_class_metrics(
        Y_test[:, fin_idx : fin_idx + 1],
        budget_prob,
        ["BUDGET_RISK"],
        threshold=args.threshold,
    )

    summary: dict[str, object] = {}
    if args.training_summary.exists():
        try:
            t_summary = json.loads(args.training_summary.read_text())
            summary.update(t_summary)
        except Exception:
            pass
    summary["macro_metrics"] = macro
    summary["micro_metrics"] = {**micro, **micro_opt}
    summary["budget_metrics"] = budget_metrics["BUDGET_RISK"]
    summary["optimal_thresholds"] = optimal_thresholds

    plots_dir.mkdir(parents=True, exist_ok=True)
    if "history" in summary and isinstance(summary["history"], dict):
        plot_training_curves(summary["history"], plots_dir / "training_curves.png")
    plot_per_class_bar(pc, "f1", plots_dir / "per_class_f1.png", "Per-class F1")
    plot_per_class_bar(pc, "roc_auc", plots_dir / "per_class_auc.png", "Per-class ROC-AUC")
    plot_roc_for_top_classes(Y_test, insights_prob, INSIGHT_CODES, plots_dir / "roc_top10.png")
    plot_pr_for_top_classes(Y_test, insights_prob, INSIGHT_CODES, plots_dir / "pr_top10.png")
    plot_confusion_matrix_micro(Y_test, insights_prob, plots_dir / "confusion_micro.png", args.threshold)
    plot_calibration(Y_test, insights_prob, plots_dir / "calibration.png")

    extras = {}
    tflite_path = PROJECT_ROOT / "models" / "final" / "viva_ai.tflite"
    if tflite_path.exists():
        extras["tflite_size_bytes"] = tflite_path.stat().st_size
    bench_json = PROJECT_ROOT / "evaluation_artifacts" / "tflite_benchmark.json"
    if bench_json.exists():
        try:
            d = json.loads(bench_json.read_text())
            extras["inference_ms"] = float(d.get("p50_ms", 0.0))
        except Exception:
            pass

    print("Building report PDF and model card...")
    build_pdf_report(
        args.out_dir / "evaluation_report.pdf",
        summary,
        pc,
        sorted(plots_dir.glob("*.png")),
    )
    build_model_card(args.out_dir / "model_card.md", summary, pc, extras)
    build_readme_summary(args.out_dir / "metrics_summary.json", summary, pc)

    (args.out_dir / "per_class.json").write_text(json.dumps(pc, indent=2))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
