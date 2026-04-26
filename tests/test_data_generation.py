"""Phase 2 tests: synthetic data + ground-truth labels."""

from __future__ import annotations

import numpy as np
import pytest

from data_generation.finance_generator import EXPENSE_CATEGORIES, generate_finance
from data_generation.health_generator import generate_health
from data_generation.labelling import LABEL_FUNCTIONS, label_all_categories
from data_generation.master_generator import (
    DAILY_COLUMNS,
    NUM_DAILY_COLUMNS,
    generate_dataset,
    generate_one_user,
    positive_counts_per_class,
)
from data_generation.medication_generator import generate_medications
from data_generation.mood_generator import generate_mood
from data_generation.sleep_generator import generate_sleep
from data_generation.tasks_generator import generate_tasks_and_habits
from data_generation.user_profiles import (
    USER_ARCHETYPES,
    archetype_weights,
    sample_user_profile,
)
from insight_categories import INSIGHT_CODES, NUM_INSIGHT_CLASSES


@pytest.fixture(scope="module")
def days_year() -> np.ndarray:
    return np.arange(
        np.datetime64("2024-01-01", "D"),
        np.datetime64("2024-01-01", "D") + 90,
        dtype="datetime64[D]",
    )


def test_archetype_weights_sum_to_one() -> None:
    w = archetype_weights()
    assert pytest.approx(sum(w.values())) == 1.0
    assert set(w.keys()) == set(USER_ARCHETYPES.keys())
    assert len(USER_ARCHETYPES) == 6


def test_sample_user_profile_fields(rng: np.random.Generator) -> None:
    p = sample_user_profile(0, rng)
    assert p.user_id == 0
    assert 0.0 < p.budget_adherence <= 1.0
    assert 0.0 < p.medication_compliance <= 1.0
    assert 1.0 <= p.mood_avg <= 5.0
    assert 4.0 <= p.sleep_avg_hours <= 9.0
    assert 2 <= p.num_habits <= 6


def test_finance_generator_shapes(rng: np.random.Generator, days_year: np.ndarray) -> None:
    p = sample_user_profile(1, rng, archetype="disciplined_professional")
    fin = generate_finance(p, days_year, rng)
    n = len(days_year)
    assert fin["daily_total_spend"].shape == (n,)
    assert fin["category_spend"].shape == (n, len(EXPENSE_CATEGORIES))
    assert (fin["daily_total_spend"] >= 0).all()
    assert (fin["num_transactions"] >= 0).all() and (fin["num_transactions"] <= 8).all()
    cs = fin["category_spend"]
    cs_sum = cs.sum(axis=1)
    np.testing.assert_allclose(cs_sum, fin["daily_total_spend"], rtol=1e-3, atol=1e-3)


def test_health_water_in_range(rng: np.random.Generator, days_year: np.ndarray) -> None:
    p = sample_user_profile(2, rng, archetype="health_focused")
    h = generate_health(p, days_year, rng)
    assert (h["water_cups"] >= 0).all() and (h["water_cups"] <= 12).all()


def test_sleep_quality_in_1_5(rng: np.random.Generator, days_year: np.ndarray) -> None:
    p = sample_user_profile(3, rng)
    s = generate_sleep(p, days_year, rng)
    q = s["sleep_quality"]
    assert q.min() >= 1 and q.max() <= 5
    assert (s["sleep_duration_hours"] >= 3.0).all()
    assert (s["sleep_duration_hours"] <= 11.0).all()


def test_mood_in_1_5(rng: np.random.Generator, days_year: np.ndarray) -> None:
    p = sample_user_profile(4, rng)
    sleep = generate_sleep(p, days_year, rng)
    th = generate_tasks_and_habits(p, days_year, rng)
    bot = np.ones(len(days_year), dtype=np.uint8)
    m = generate_mood(
        p,
        days_year,
        rng,
        sleep_quality=sleep["sleep_quality"],
        habit_completion_rate=th["habit_completion_rate"],
        budget_on_track=bot,
    )
    assert m["mood_score"].min() >= 1
    assert m["mood_score"].max() <= 5


def test_medications_zero_for_no_meds(rng: np.random.Generator, days_year: np.ndarray) -> None:
    p = sample_user_profile(5, rng, archetype="chaotic_creative")
    p.num_medications = 0
    p.has_critical_medications = False
    med = generate_medications(p, days_year, rng)
    assert (med["doses_scheduled"] == 0).all()
    assert (med["doses_taken"] == 0).all()


def test_full_user_pipeline_no_nan_labels(rng: np.random.Generator, days_year: np.ndarray) -> None:
    profile, daily, labels = generate_one_user(user_id=99, rng=rng, days=days_year)
    assert daily.shape == (len(days_year), NUM_DAILY_COLUMNS)
    assert labels.shape == (len(days_year), NUM_INSIGHT_CLASSES)
    # Daily array has NaN only in weight column (sentinel).
    weight_idx = DAILY_COLUMNS.index("weight_kg")
    other = np.delete(daily, weight_idx, axis=1)
    assert not np.isnan(other).any(), "Non-weight daily columns must not be NaN."
    # Labels are 0/1 only.
    assert set(np.unique(labels)).issubset({0, 1})


def test_label_functions_count_matches_codes() -> None:
    assert len(LABEL_FUNCTIONS) == NUM_INSIGHT_CLASSES
    assert set(LABEL_FUNCTIONS.keys()) == set(INSIGHT_CODES)


def test_label_all_categories_shape(rng: np.random.Generator, days_year: np.ndarray) -> None:
    _, _, labels = generate_one_user(user_id=10, rng=rng, days=days_year)
    assert labels.shape == (len(days_year), NUM_INSIGHT_CLASSES)


def test_small_dataset_label_coverage(tmp_path) -> None:
    """100-user smoke run: every category gets >= 10 positive examples."""
    paths = generate_dataset(
        num_users=100,
        days_per_user=365,
        seed=42,
        out_dir=tmp_path,
        show_progress=False,
    )
    labels = np.load(paths["labels"])["labels"]
    counts = positive_counts_per_class(labels)
    too_low = {k: v for k, v in counts.items() if v < 10}
    assert not too_low, f"Categories with < 10 positives in 100-user run: {too_low}"
