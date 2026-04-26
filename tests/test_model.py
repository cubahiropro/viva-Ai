"""Phase 4 tests: model build, compile, single-batch fit, output shapes."""

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

from insight_categories import NUM_INSIGHT_CLASSES  # noqa: E402
from models.insight_classifier import build_viva_model, compile_viva_model  # noqa: E402
from training.loss_functions import binary_focal_loss  # noqa: E402
from training.trainer import (  # noqa: E402
    compute_sample_weights,
    flatten_user_days,
    labels_to_targets,
    make_user_splits,
)


def test_build_viva_model_shapes() -> None:
    model = build_viva_model()
    assert model.input_shape == (None, 128)
    # Keras 3 returns output_shape as a list; map by output_names instead.
    out_shapes = dict(zip(model.output_names, model.output_shape))
    assert out_shapes["insights"] == (None, NUM_INSIGHT_CLASSES)
    assert out_shapes["budget_risk"] == (None, 1)
    assert out_shapes["mood_prediction"] == (None, 1)


def test_compile_and_single_batch_fit() -> None:
    model = build_viva_model()
    compile_viva_model(model, learning_rate=1e-3)
    rng = np.random.default_rng(0)
    X = rng.random((32, 128), dtype=np.float32)
    Y = {
        "insights": rng.integers(0, 2, (32, NUM_INSIGHT_CLASSES)).astype(np.float32),
        "budget_risk": rng.integers(0, 2, (32, 1)).astype(np.float32),
        "mood_prediction": rng.random((32, 1)).astype(np.float32),
    }
    hist = model.fit(X, Y, epochs=1, batch_size=16, verbose=0)
    assert "loss" in hist.history
    out = model.predict(X[:4], verbose=0)
    assert out["insights"].shape == (4, NUM_INSIGHT_CLASSES)


def test_focal_loss_finite() -> None:
    loss = binary_focal_loss(gamma=2.0, alpha=0.25)
    yt = tf.constant(np.array([[1, 0, 1], [0, 0, 1]], dtype=np.float32))
    yp = tf.constant(np.array([[0.9, 0.1, 0.4], [0.05, 0.6, 0.95]], dtype=np.float32))
    val = loss(yt, yp).numpy()
    assert np.isfinite(val) and val >= 0


def test_user_splits_no_overlap() -> None:
    splits = make_user_splits(100, seed=7)
    a = set(splits.train_users.tolist())
    b = set(splits.val_users.tolist())
    c = set(splits.test_users.tolist())
    assert a.isdisjoint(b)
    assert a.isdisjoint(c)
    assert b.isdisjoint(c)
    assert len(a) + len(b) + len(c) == 100


def test_flatten_and_targets() -> None:
    rng = np.random.default_rng(0)
    F = rng.random((10, 30, 128), dtype=np.float32)
    L = rng.integers(0, 2, (10, 30, NUM_INSIGHT_CLASSES)).astype(np.uint8)
    Xf, Yf = flatten_user_days(F, L, np.arange(10))
    assert Xf.shape == (300, 128)
    assert Yf.shape == (300, NUM_INSIGHT_CLASSES)
    targets = labels_to_targets(Yf)
    assert targets["insights"].shape == (300, NUM_INSIGHT_CLASSES)
    assert targets["budget_risk"].shape == (300, 1)
    w = compute_sample_weights(Yf)
    assert w.shape == (300,)
    assert np.all(w >= 0)
    assert np.all(w <= 5.0)
