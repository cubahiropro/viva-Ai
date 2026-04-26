"""Phase 6 tests: TFLite conversion, parity, validator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

tf = pytest.importorskip("tensorflow")

from conversion.tflite_converter import (  # noqa: E402
    benchmark_tflite,
    convert_to_tflite_dynamic_range,
    convert_to_tflite_int8,
    parity_check,
)
from conversion.validator import TFLiteAssertions, validate_tflite  # noqa: E402
from models.insight_classifier import build_viva_model, compile_viva_model  # noqa: E402


def _trained_tiny_model() -> tf.keras.Model:
    rng = np.random.default_rng(0)
    X = rng.random((128, 128), dtype=np.float32)
    Y = {
        "insights": rng.integers(0, 2, (128, 40)).astype(np.float32),
        "budget_risk": rng.integers(0, 2, (128, 1)).astype(np.float32),
        "mood_prediction": rng.random((128, 1)).astype(np.float32),
    }
    model = build_viva_model()
    compile_viva_model(model, learning_rate=1e-3)
    model.fit(X, Y, epochs=1, batch_size=32, verbose=0)
    return model


@pytest.mark.slow
def test_dynamic_range_conversion(tmp_path: Path) -> None:
    model = _trained_tiny_model()
    out = tmp_path / "viva_dyn.tflite"
    info = convert_to_tflite_dynamic_range(model, out)
    assert out.exists()
    assert info["size_bytes"] > 0


@pytest.mark.slow
def test_int8_conversion_and_parity(tmp_path: Path) -> None:
    model = _trained_tiny_model()
    rng = np.random.default_rng(1)
    samples = rng.random((200, 128), dtype=np.float32)
    out = tmp_path / "viva_int8.tflite"
    info = convert_to_tflite_int8(model, samples[:100], out)
    assert out.exists()
    parity = parity_check(model, out, samples[:25], n=25, tol=0.5)
    assert parity["max_abs_diff"] < 0.5
    bench = benchmark_tflite(out, samples[:50], runs=20, warmup=5)
    assert bench["mean_ms"] > 0
    val = validate_tflite(
        info,
        bench,
        parity,
        TFLiteAssertions(max_parity_mean_diff=0.5, max_parity_max_diff=0.5),
    )
    assert val["size_ok"]
    assert val["parity_ok"]
