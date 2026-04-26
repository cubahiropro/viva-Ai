"""Phase 5 tests: metric correctness, plot generation, report generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.metrics import (  # noqa: E402
    find_optimal_thresholds,
    macro_metrics,
    micro_metrics,
    per_class_metrics,
    regression_metrics,
)
from evaluation.report import build_pdf_report, build_model_card  # noqa: E402
from evaluation.visualisations import plot_per_class_bar  # noqa: E402


def test_per_class_metrics_perfect_predictions() -> None:
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.95, 0.95], [0.05, 0.05]], dtype=np.float32)
    pc = per_class_metrics(y_true, y_prob, ["A", "B"])
    for name in ["A", "B"]:
        assert pc[name]["precision"] == 1.0
        assert pc[name]["recall"] == 1.0
        assert pc[name]["f1"] == 1.0


def test_macro_micro_aggregates() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (200, 5)).astype(np.uint8)
    y_prob = rng.random((200, 5))
    pc = per_class_metrics(y_true, y_prob, ["a", "b", "c", "d", "e"])
    macro = macro_metrics(pc)
    assert "macro_f1" in macro
    micro = micro_metrics(y_true, y_prob)
    assert 0.0 <= micro["micro_f1"] <= 1.0


def test_regression_metrics() -> None:
    yt = np.array([0.2, 0.4, 0.6, 0.8])
    yp = np.array([0.25, 0.45, 0.55, 0.75])
    m = regression_metrics(yt, yp)
    assert m["mae"] < 0.1


def test_find_optimal_thresholds() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (200, 3)).astype(np.uint8)
    y_prob = rng.random((200, 3))
    th = find_optimal_thresholds(y_true, y_prob, ["a", "b", "c"])
    for name in ["a", "b", "c"]:
        assert 0.0 < th[name] < 1.0


def test_report_generation_smoke(tmp_path: Path) -> None:
    pc = {
        "A": {"support_positive": 50, "precision": 0.8, "recall": 0.7, "f1": 0.75,
              "roc_auc": 0.85, "ap": 0.7, "brier": 0.2},
        "B": {"support_positive": 10, "precision": 0.3, "recall": 0.4, "f1": 0.34,
              "roc_auc": 0.55, "ap": 0.2, "brier": 0.3},
    }
    plots: list[Path] = []
    bar_path = tmp_path / "bar.png"
    plot_per_class_bar(pc, "f1", bar_path, "test")
    plots.append(bar_path)
    summary = {"test_metrics": {"insights_auc": 0.85}, "macro_metrics": {"macro_f1": 0.55},
               "micro_metrics": {"micro_f1": 0.6}}
    pdf_path = tmp_path / "report.pdf"
    build_pdf_report(pdf_path, summary, pc, plots)
    assert pdf_path.exists() and pdf_path.stat().st_size > 100
    md_path = tmp_path / "card.md"
    build_model_card(md_path, summary, pc, {"tflite_size_bytes": 1024 * 1024,
                                            "inference_ms": 5.0})
    assert md_path.exists() and "Viva Insight Net" in md_path.read_text()
