"""PDF + Markdown report generation for the trained model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _table(data: list[list[str]]) -> Table:
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tbl


def build_pdf_report(
    out_path: Path,
    summary: dict[str, Any],
    per_class: dict[str, dict[str, float]],
    plots: list[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph("Viva AI — Model Evaluation Report", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated at {datetime.now(timezone.utc).isoformat()} (UTC)",
            styles["Italic"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Aggregate metrics", styles["Heading2"]))
    test = summary.get("test_metrics", {})
    macro = summary.get("macro_metrics", {})
    micro = summary.get("micro_metrics", {})
    rows = [["metric", "value"]]
    for d in (test, macro, micro):
        for k, v in d.items():
            try:
                rows.append([k, f"{float(v):.4f}"])
            except (TypeError, ValueError):
                rows.append([k, str(v)])
    story.append(_table(rows))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Top / bottom classes by F1", styles["Heading2"]))
    items = sorted(per_class.items(), key=lambda kv: kv[1].get("f1", 0.0))
    bottom = items[:5]
    top = items[-5:][::-1]

    def _class_table(rows: list[tuple[str, dict[str, float]]]) -> Table:
        head = ["class", "support", "precision", "recall", "f1", "auc", "ap"]
        out = [head]
        for name, m in rows:
            out.append([
                name,
                f"{int(m.get('support_positive', 0))}",
                f"{m.get('precision', 0.0):.3f}",
                f"{m.get('recall', 0.0):.3f}",
                f"{m.get('f1', 0.0):.3f}",
                f"{m.get('roc_auc', float('nan')):.3f}",
                f"{m.get('ap', float('nan')):.3f}",
            ])
        return _table(out)

    story.append(Paragraph("Top 5 classes (F1)", styles["Heading4"]))
    story.append(_class_table(top))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Bottom 5 classes (F1)", styles["Heading4"]))
    story.append(_class_table(bottom))
    story.append(PageBreak())

    story.append(Paragraph("Plots", styles["Heading2"]))
    for p in plots:
        if not p.exists():
            continue
        try:
            story.append(Paragraph(p.stem, styles["Heading4"]))
            story.append(Image(str(p), width=15 * cm, height=10 * cm, kind="proportional"))
            story.append(Spacer(1, 0.3 * cm))
        except Exception:
            continue

    doc.build(story)


def build_model_card(
    out_path: Path,
    summary: dict[str, Any],
    per_class: dict[str, dict[str, float]],
    extras: dict[str, Any] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extras = extras or {}
    test = summary.get("test_metrics", {})
    macro = summary.get("macro_metrics", {})
    micro = summary.get("micro_metrics", {})

    lines: list[str] = []
    lines.append("# Viva Insight Net — Model Card\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("\n## 1. Model details\n")
    lines.append("- Architecture: 3-layer MLP, multi-output head (insights / budget / mood)\n")
    lines.append("- Input: 128-d feature vector (see `feature_engineering/feature_pipeline.py`)\n")
    lines.append(f"- Outputs: {len(per_class)} insight classes, 1 budget logit, 1 mood scalar\n")
    if extras.get("tflite_size_bytes") is not None:
        lines.append(f"- TFLite size: {extras['tflite_size_bytes'] / 1024:.1f} KB\n")
    if extras.get("inference_ms") is not None:
        lines.append(f"- Mean inference time: {extras['inference_ms']:.2f} ms (CPU, single sample)\n")

    lines.append("\n## 2. Intended use\n")
    lines.append(
        "This model powers the Viva on-device assistant on Android. It produces "
        "user-personalised insights ranked by predicted probability. **It is not a "
        "medical or financial advice tool.**\n"
    )

    lines.append("\n## 3. Training data\n")
    lines.append(
        "Trained entirely on synthetic data generated from rule-based archetypes. "
        "No real user data was used. See `data/synthetic/` for generators.\n"
    )

    lines.append("\n## 4. Performance\n")
    lines.append("### Aggregate metrics\n")
    lines.append("| Metric | Value |\n|---|---|\n")
    for d in (test, macro, micro):
        for k, v in d.items():
            try:
                lines.append(f"| {k} | {float(v):.4f} |\n")
            except (TypeError, ValueError):
                lines.append(f"| {k} | {v} |\n")

    lines.append("\n### Per-class metrics\n")
    lines.append("| class | support | precision | recall | f1 | auc | ap |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for name, m in per_class.items():
        lines.append(
            f"| {name} | {int(m.get('support_positive', 0))} "
            f"| {m.get('precision', 0.0):.3f} "
            f"| {m.get('recall', 0.0):.3f} "
            f"| {m.get('f1', 0.0):.3f} "
            f"| {m.get('roc_auc', float('nan')):.3f} "
            f"| {m.get('ap', float('nan')):.3f} |\n"
        )

    lines.append("\n## 5. Limitations\n")
    lines.append(
        "- Trained only on synthetic data; real-world generalisation is unverified.\n"
        "- No explicit fairness audit across demographic groups beyond archetype balance.\n"
        "- The mood head is calibrated to a [0,1] proxy of the synthetic 1–5 scale.\n"
        "- Edge cases (new users, partial data) are handled by feature defaults; predictions in those regimes should be treated cautiously.\n"
    )

    lines.append("\n## 6. Privacy\n")
    lines.append(
        "All inference is on-device. No telemetry leaves the user's phone. "
        "Inputs are derived locally from the user's tracked activity inside Viva.\n"
    )

    out_path.write_text("".join(lines))


def build_readme_summary(
    out_path: Path,
    summary: dict[str, Any],
    per_class: dict[str, dict[str, float]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    macro = summary.get("macro_metrics", {})
    micro = summary.get("micro_metrics", {})
    test = summary.get("test_metrics", {})
    out_path.write_text(json.dumps({
        "test_metrics": test,
        "macro_metrics": macro,
        "micro_metrics": micro,
        "per_class": per_class,
    }, indent=2))
