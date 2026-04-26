"""Shared pytest fixtures for the Viva AI test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic numpy RNG used across tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def small_user_count() -> int:
    """Tiny user count used for fast tests."""
    return 12


@pytest.fixture(scope="session")
def small_days() -> int:
    """Days per user in fast tests — must be > 60 so 30-day windows fit."""
    return 90
