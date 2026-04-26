"""Phase 1 smoke tests: imports, config loading, insight category integrity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config_loader import DEFAULT_CONFIG_PATH, load_config
from insight_categories import (
    INSIGHT_CATEGORIES,
    INSIGHT_CODES,
    INSIGHT_INDEX,
    INSIGHT_SEVERITY,
    NUM_INSIGHT_CLASSES,
)


def test_core_dependencies_import() -> None:
    """Heavy deps (TF/sklearn) are not required for Phase 1 smoke."""
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import yaml  # noqa: F401


def test_config_loads(project_root: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg["features"]["input_dim"] == 128
    assert cfg["model"]["num_insight_classes"] == NUM_INSIGHT_CLASSES
    assert cfg["data"]["num_users"] == 5000
    assert cfg["data"]["days_per_user"] == 365
    assert pytest.approx(
        cfg["data"]["train_split"] + cfg["data"]["val_split"] + cfg["data"]["test_split"]
    ) == 1.0


def test_insight_categories_exactly_40() -> None:
    assert NUM_INSIGHT_CLASSES == 40, "Insight head must have exactly 40 classes."
    assert len(INSIGHT_CODES) == 40
    assert len(INSIGHT_CATEGORIES) == 40


def test_insight_codes_unique_and_indexed() -> None:
    assert len(set(INSIGHT_CODES)) == 40
    for i, code in enumerate(INSIGHT_CODES):
        assert INSIGHT_INDEX[code] == i


def test_insight_severity_complete() -> None:
    for code in INSIGHT_CODES:
        sev = INSIGHT_SEVERITY[code]
        assert sev in {"warning", "info", "positive"}


def test_rng_fixture_deterministic() -> None:
    """Two RNGs seeded the same must produce identical sequences."""
    a = np.random.default_rng(seed=42).standard_normal(5)
    b = np.random.default_rng(seed=42).standard_normal(5)
    np.testing.assert_allclose(a, b)
