"""Daily tasks + habits generator (the prompt groups habits under tasks)."""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile


def _weekday(days: np.ndarray) -> np.ndarray:
    return np.array(
        [int(d.astype("datetime64[D]").astype("O").weekday()) for d in days]
    )


def generate_tasks_and_habits(
    profile: UserProfile,
    days: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(days)
    weekday = _weekday(days)
    is_weekend = (weekday >= 5).astype(np.float32)

    # ---------------- Tasks ----------------
    tasks_created = rng.poisson(2.5 - 0.8 * is_weekend, size=n).clip(0, 8).astype(np.int32)
    completion_p = profile.task_completion_rate * (1.0 - 0.1 * is_weekend)
    completion_p = np.clip(completion_p, 0.05, 0.99)
    tasks_completed = rng.binomial(tasks_created, completion_p).astype(np.int32)

    overdue_running = 0
    overdue_arr = np.zeros(n, dtype=np.int32)
    for i in range(n):
        unfinished = max(0, int(tasks_created[i]) - int(tasks_completed[i]))
        overdue_running = max(0, overdue_running + unfinished - rng.binomial(overdue_running, 0.25))
        overdue_arr[i] = overdue_running

    # ---------------- Habits ----------------
    num_habits = profile.num_habits
    morning_habit_count = max(1, num_habits // 2)
    evening_habit_count = num_habits - morning_habit_count

    base_rate = profile.habit_completion_rate
    morning_rate = np.clip(base_rate + 0.1, 0.05, 0.99)
    evening_rate = np.clip(base_rate - 0.1, 0.02, 0.95)

    # Daily completion fraction with noise.
    morning_completed = rng.binomial(
        morning_habit_count, morning_rate, size=n
    ).astype(np.float32)
    evening_completed = rng.binomial(
        evening_habit_count, evening_rate, size=n
    ).astype(np.float32)

    morning_rate_per_day = morning_completed / max(morning_habit_count, 1)
    evening_rate_per_day = evening_completed / max(evening_habit_count, 1)

    total_completed_per_day = morning_completed + evening_completed
    overall_rate_per_day = total_completed_per_day / max(num_habits, 1)

    # Streak — increments when overall_rate_per_day == 1.0 (all habits done), resets otherwise.
    streak = np.zeros(n, dtype=np.int32)
    cur = 0
    for i in range(n):
        all_done = total_completed_per_day[i] == num_habits
        if all_done:
            cur += 1
        else:
            # Probabilistic streak break — softer for disciplined users.
            if rng.uniform() < 0.6:
                cur = 0
            else:
                # Treat as missed-but-recoverable
                cur = max(0, cur - 1)
        streak[i] = cur

    longest_ever = np.maximum.accumulate(streak)

    return {
        "tasks_created": tasks_created,
        "tasks_completed": tasks_completed,
        "tasks_overdue": overdue_arr,
        "task_completion_rate_today": np.where(
            tasks_created > 0, tasks_completed / np.maximum(tasks_created, 1), 0.5
        ).astype(np.float32),
        "habit_completion_rate": overall_rate_per_day.astype(np.float32),
        "morning_habit_rate": morning_rate_per_day.astype(np.float32),
        "evening_habit_rate": evening_rate_per_day.astype(np.float32),
        "habit_streak": streak,
        "habit_longest_ever": longest_ever.astype(np.int32),
        "num_habits": np.full(n, num_habits, dtype=np.int32),
    }
