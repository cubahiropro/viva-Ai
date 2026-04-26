"""Master orchestrator: produce a full synthetic dataset of `num_users × num_days`.

Each user's per-day series is computed in dependency order, then run through the
40 ground-truth labellers. Results are saved to:

    data/synthetic/users.parquet      one row per user (metadata)
    data/synthetic/daily.npz          per-day arrays (num_users, num_days, F)
    data/synthetic/labels.npz         per-day labels (num_users, num_days, 40)
    data/synthetic/feature_columns.json  ordered names of the F daily columns
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insight_categories import NUM_INSIGHT_CLASSES

from .finance_generator import EXPENSE_CATEGORIES, generate_finance
from .health_generator import generate_health
from .labelling import label_all_categories
from .medication_generator import generate_medications
from .mood_generator import generate_mood
from .sleep_generator import generate_sleep
from .tasks_generator import generate_tasks_and_habits
from .user_profiles import sample_user_profile

# Order of columns in the daily.npz array — must match _build_daily_array().
DAILY_COLUMNS: list[str] = [
    "daily_total_spend",
    "num_transactions",
    "income_today",
    "monthly_budget",
    "is_unusual_expense",
    "is_weekend",
    *(f"category_spend_{c}" for c in EXPENSE_CATEGORIES),
    "water_cups",
    "water_logged",
    "weight_kg",                     # NaN sentinel allowed
    "weight_logged",
    "sleep_duration_hours",
    "sleep_quality",
    "bedtime_hour_after_20",
    "wake_hour",
    "sleep_logged",
    "mood_score",
    "mood_logged",
    "tasks_created",
    "tasks_completed",
    "tasks_overdue",
    "task_completion_rate_today",
    "habit_completion_rate",
    "morning_habit_rate",
    "evening_habit_rate",
    "habit_streak",
    "habit_longest_ever",
    "num_habits",
    "doses_scheduled",
    "doses_taken",
    "timing_offset_minutes",
    "has_critical_medications",
]

NUM_DAILY_COLUMNS = len(DAILY_COLUMNS)


def _budget_on_track(series: dict) -> np.ndarray:
    """Boolean array: at day d, is current monthly pace <= budget * 1.0?"""
    spend = series["daily_total_spend"]
    budget = series["monthly_budget"]
    month_idx = series["month_index"]
    n = len(spend)
    out = np.zeros(n, dtype=np.uint8)
    total = 0.0
    days = 0
    last_month = -1
    for i in range(n):
        if month_idx[i] != last_month:
            total = 0.0
            days = 0
            last_month = int(month_idx[i])
        total += float(spend[i])
        days += 1
        if days < 3:
            out[i] = 1
            continue
        projected = (total / days) * 30.0
        if projected <= float(budget[i]) * 1.05:
            out[i] = 1
    return out


def _build_daily_array(series: dict) -> np.ndarray:
    n = len(series["daily_total_spend"])
    out = np.zeros((n, NUM_DAILY_COLUMNS), dtype=np.float32)
    cs = series["category_spend"]                      # (n, 6)

    col = {name: i for i, name in enumerate(DAILY_COLUMNS)}
    out[:, col["daily_total_spend"]] = series["daily_total_spend"]
    out[:, col["num_transactions"]] = series["num_transactions"]
    out[:, col["income_today"]] = series["income_today"]
    out[:, col["monthly_budget"]] = series["monthly_budget"]
    out[:, col["is_unusual_expense"]] = series["is_unusual_expense"].astype(np.float32)
    out[:, col["is_weekend"]] = series["is_weekend"].astype(np.float32)
    for j, c in enumerate(EXPENSE_CATEGORIES):
        out[:, col[f"category_spend_{c}"]] = cs[:, j]
    out[:, col["water_cups"]] = series["water_cups"]
    out[:, col["water_logged"]] = series["water_logged"].astype(np.float32)
    out[:, col["weight_kg"]] = series["weight_kg"]
    out[:, col["weight_logged"]] = series["weight_logged"].astype(np.float32)
    out[:, col["sleep_duration_hours"]] = series["sleep_duration_hours"]
    out[:, col["sleep_quality"]] = series["sleep_quality"]
    out[:, col["bedtime_hour_after_20"]] = series["bedtime_hour_after_20"]
    out[:, col["wake_hour"]] = series["wake_hour"]
    out[:, col["sleep_logged"]] = series["sleep_logged"].astype(np.float32)
    out[:, col["mood_score"]] = series["mood_score"]
    out[:, col["mood_logged"]] = series["mood_logged"].astype(np.float32)
    out[:, col["tasks_created"]] = series["tasks_created"]
    out[:, col["tasks_completed"]] = series["tasks_completed"]
    out[:, col["tasks_overdue"]] = series["tasks_overdue"]
    out[:, col["task_completion_rate_today"]] = series["task_completion_rate_today"]
    out[:, col["habit_completion_rate"]] = series["habit_completion_rate"]
    out[:, col["morning_habit_rate"]] = series["morning_habit_rate"]
    out[:, col["evening_habit_rate"]] = series["evening_habit_rate"]
    out[:, col["habit_streak"]] = series["habit_streak"]
    out[:, col["habit_longest_ever"]] = series["habit_longest_ever"]
    out[:, col["num_habits"]] = series["num_habits"]
    out[:, col["doses_scheduled"]] = series["doses_scheduled"]
    out[:, col["doses_taken"]] = series["doses_taken"]
    out[:, col["timing_offset_minutes"]] = series["timing_offset_minutes"]
    out[:, col["has_critical_medications"]] = series["has_critical_medications"].astype(np.float32)
    return out


def generate_one_user(
    user_id: int,
    rng: np.random.Generator,
    days: np.ndarray,
    archetype: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return (profile_dict, daily_array (D, F), labels (D, 40))."""
    profile = sample_user_profile(user_id, rng, archetype=archetype)

    finance = generate_finance(profile, days, rng)
    tasks = generate_tasks_and_habits(profile, days, rng)
    health = generate_health(profile, days, rng, habit_completion_per_day=tasks["habit_completion_rate"])

    # Sleep + mood iteratively: previous-day mood feeds sleep quality (via prior).
    n = len(days)
    prev_mood = np.full(n, profile.mood_avg, dtype=np.float32)  # initial guess
    sleep = generate_sleep(profile, days, rng, prev_day_mood=prev_mood)

    # Compute budget-on-track using finance only
    budget_on_track = _budget_on_track({
        "daily_total_spend": finance["daily_total_spend"],
        "monthly_budget": finance["monthly_budget"],
        "month_index": finance["month_index"],
    })

    mood = generate_mood(
        profile,
        days,
        rng,
        sleep_quality=sleep["sleep_quality"],
        habit_completion_rate=tasks["habit_completion_rate"],
        budget_on_track=budget_on_track,
    )

    # One refinement pass: regenerate sleep using actual mood, then mood again.
    sleep = generate_sleep(profile, days, rng, prev_day_mood=mood["mood_score"])
    mood = generate_mood(
        profile,
        days,
        rng,
        sleep_quality=sleep["sleep_quality"],
        habit_completion_rate=tasks["habit_completion_rate"],
        budget_on_track=budget_on_track,
    )

    medications = generate_medications(profile, days, rng)

    series: dict[str, np.ndarray] = {
        **finance,
        **health,
        **sleep,
        **mood,
        **tasks,
        **medications,
        "is_weekend": finance["is_weekend"],   # canonical version
    }

    daily_array = _build_daily_array(series)
    labels = label_all_categories(series)

    profile_dict = {
        **asdict(profile),
        "user_id": user_id,
    }
    profile_dict.pop("extras", None)
    return profile_dict, daily_array, labels


def generate_dataset(
    num_users: int,
    days_per_user: int,
    seed: int,
    out_dir: str | Path,
    start_date: str = "2024-01-01",
    show_progress: bool = True,
) -> dict[str, Path]:
    """Generate the full synthetic dataset and write it to `out_dir`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    days = np.arange(
        np.datetime64(start_date, "D"),
        np.datetime64(start_date, "D") + days_per_user,
        dtype="datetime64[D]",
    )
    assert len(days) == days_per_user

    base_rng = np.random.default_rng(seed)
    user_seeds = base_rng.integers(0, 2**32 - 1, size=num_users, dtype=np.uint32)

    daily_all = np.zeros((num_users, days_per_user, NUM_DAILY_COLUMNS), dtype=np.float32)
    labels_all = np.zeros((num_users, days_per_user, NUM_INSIGHT_CLASSES), dtype=np.uint8)
    profiles: list[dict[str, Any]] = []

    iterator = range(num_users)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating users")

    for u in iterator:
        rng = np.random.default_rng(int(user_seeds[u]))
        profile, daily, labels = generate_one_user(user_id=u, rng=rng, days=days)
        daily_all[u] = daily
        labels_all[u] = labels
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    profiles_path = out_path / "users.parquet"
    profiles_df.to_parquet(profiles_path, index=False)

    daily_path = out_path / "daily.npz"
    np.savez_compressed(
        daily_path, daily=daily_all, days=days.astype("datetime64[D]").astype("int64")
    )

    labels_path = out_path / "labels.npz"
    np.savez_compressed(labels_path, labels=labels_all)

    cols_path = out_path / "feature_columns.json"
    with cols_path.open("w", encoding="utf-8") as fh:
        json.dump(DAILY_COLUMNS, fh, indent=2)

    return {
        "users": profiles_path,
        "daily": daily_path,
        "labels": labels_path,
        "columns": cols_path,
    }


def positive_counts_per_class(labels: np.ndarray) -> dict[str, int]:
    """Return positive count per insight code from a (N, D, 40) labels tensor."""
    from insight_categories import INSIGHT_CODES

    sums = labels.reshape(-1, labels.shape[-1]).sum(axis=0)
    return {code: int(sums[i]) for i, code in enumerate(INSIGHT_CODES)}
