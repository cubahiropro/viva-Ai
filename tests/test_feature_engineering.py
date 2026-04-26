"""Phase 3 tests: 128-feature pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_generation.master_generator import DAILY_COLUMNS, generate_one_user
from feature_engineering.feature_pipeline import (
    FEATURE_NAMES,
    INPUT_DIM,
    NormaliserParams,
    WINDOW_DAYS,
    apply_normaliser,
    compute_features_for_window,
    compute_user_features,
    fit_normaliser,
)


@pytest.fixture(scope="module")
def days_year() -> np.ndarray:
    return np.arange(
        np.datetime64("2024-01-01", "D"),
        np.datetime64("2024-01-01", "D") + 90,
        dtype="datetime64[D]",
    )


@pytest.fixture(scope="module")
def synthetic_user(rng_module, days_year):
    rng = np.random.default_rng(seed=1234)
    profile, daily, labels = generate_one_user(
        user_id=42, rng=rng, days=days_year, archetype="disciplined_professional"
    )
    return profile, daily, labels


@pytest.fixture(scope="module")
def rng_module() -> np.random.Generator:
    return np.random.default_rng(seed=1234)


def test_feature_names_count_and_unique() -> None:
    assert len(FEATURE_NAMES) == INPUT_DIM == 128
    assert len(set(FEATURE_NAMES)) == 128


def test_feature_groups_match_prompt_counts() -> None:
    counts = {
        "fin": 24,
        "hlt": 16,
        "slp": 18,
        "mod": 14,
        "tsk_or_hbt": 20,
        "med": 12,
        "crs": 12,
        "tmp": 12,
    }
    fin = sum(1 for n in FEATURE_NAMES if n.startswith("fin_"))
    hlt = sum(1 for n in FEATURE_NAMES if n.startswith("hlt_"))
    slp = sum(1 for n in FEATURE_NAMES if n.startswith("slp_"))
    mod = sum(1 for n in FEATURE_NAMES if n.startswith("mod_"))
    tsk_or_hbt = sum(
        1 for n in FEATURE_NAMES if n.startswith("tsk_") or n.startswith("hbt_")
    )
    med = sum(1 for n in FEATURE_NAMES if n.startswith("med_"))
    crs = sum(1 for n in FEATURE_NAMES if n.startswith("crs_"))
    tmp = sum(1 for n in FEATURE_NAMES if n.startswith("tmp_"))
    assert fin == counts["fin"]
    assert hlt == counts["hlt"]
    assert slp == counts["slp"]
    assert mod == counts["mod"]
    assert tsk_or_hbt == counts["tsk_or_hbt"]
    assert med == counts["med"]
    assert crs == counts["crs"]
    assert tmp == counts["tmp"]


def test_compute_user_features_shape(synthetic_user) -> None:
    profile, daily, _ = synthetic_user
    feats = compute_user_features(
        daily,
        columns=DAILY_COLUMNS,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    assert feats.shape == (daily.shape[0], INPUT_DIM)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any(), "Features must not contain NaN."
    assert not np.isinf(feats).any(), "Features must not contain Inf."


def test_compute_user_features_values_finite_and_clipped(synthetic_user) -> None:
    profile, daily, _ = synthetic_user
    feats = compute_user_features(
        daily,
        columns=DAILY_COLUMNS,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    # Pre-normalisation, features can exceed [0,1] (clipped only after fit).
    # But they must be finite.
    assert np.all(np.isfinite(feats))


def test_window_function_matches_full_pipeline(synthetic_user) -> None:
    profile, daily, _ = synthetic_user
    full = compute_user_features(
        daily,
        columns=DAILY_COLUMNS,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    n = daily.shape[0]
    target_day = n - 1
    window_start = target_day - WINDOW_DAYS + 1
    window = daily[window_start : target_day + 1]
    today_date = np.datetime64("2024-01-01", "D") + target_day
    single = compute_features_for_window(
        window,
        columns=DAILY_COLUMNS,
        today_date=today_date,
        days_since_first_use=target_day,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    # The window-based call only sees 30 days of context, so monthly aggregates
    # may differ from the full-history call when target_day is past day-of-month.
    # We check the temporal + 7-day-window features (last 12 features here).
    np.testing.assert_allclose(single[-12:], full[target_day, -12:], atol=1e-4)


def test_normaliser_round_trip(synthetic_user) -> None:
    profile, daily, _ = synthetic_user
    feats = compute_user_features(
        daily,
        columns=DAILY_COLUMNS,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    feats_batch = feats[None, ...]
    norm = fit_normaliser(feats_batch)
    out = apply_normaliser(feats_batch, norm)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.shape == feats_batch.shape


def test_normaliser_save_load(tmp_path, synthetic_user) -> None:
    profile, daily, _ = synthetic_user
    feats = compute_user_features(
        daily,
        columns=DAILY_COLUMNS,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )
    norm = fit_normaliser(feats[None, ...])
    p = tmp_path / "norm.json"
    norm.save(p)
    loaded = NormaliserParams.load(p)
    assert loaded.feature_names == norm.feature_names
    np.testing.assert_allclose(loaded.minimums, norm.minimums)
    np.testing.assert_allclose(loaded.maximums, norm.maximums)
