"""Daily health series generator: water intake + weight + log presence."""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile


def generate_health(
    profile: UserProfile,
    days: np.ndarray,
    rng: np.random.Generator,
    habit_completion_per_day: np.ndarray | None = None,  # exercise habit, 0..1
) -> dict[str, np.ndarray]:
    n = len(days)
    month = np.array([int(np.datetime_as_string(d, unit="M")[-2:]) for d in days])

    # Summer (Jun-Aug) = +1.5 cups expected.
    summer = ((month >= 6) & (month <= 8)).astype(np.float32)
    base_water = {
        "disciplined_professional": 7.5,
        "stressed_young_adult": 4.5,
        "health_focused": 9.0,
        "budget_conscious": 6.0,
        "chaotic_creative": 4.0,
        "elderly_health_tracker": 6.5,
    }.get(profile.archetype, 6.0)

    water_mu = base_water + summer * 1.5
    water_cups = np.clip(rng.normal(water_mu, 1.8, size=n), 0, 12).round().astype(np.int32)

    # Logging frequency varies by archetype — stressed/chaotic skip more.
    log_prob = {
        "disciplined_professional": 0.92,
        "stressed_young_adult": 0.55,
        "health_focused": 0.98,
        "budget_conscious": 0.72,
        "chaotic_creative": 0.40,
        "elderly_health_tracker": 0.65,
    }.get(profile.archetype, 0.7)
    water_logged = rng.uniform(size=n) < log_prob

    # Weight: weighted only ~once per week. Slow drift modulated by exercise habit.
    weigh_prob = 1.0 / 7.0
    weighed = rng.uniform(size=n) < weigh_prob

    weight = np.full(n, np.nan, dtype=np.float32)
    if profile.archetype == "health_focused":
        slope_per_day = -0.01
    elif profile.archetype == "stressed_young_adult":
        slope_per_day = +0.005
    else:
        slope_per_day = float(rng.normal(0, 0.005))

    if habit_completion_per_day is not None:
        habit_factor = (habit_completion_per_day - 0.5) * -0.02
    else:
        habit_factor = np.zeros(n, dtype=np.float32)

    drift = np.cumsum(slope_per_day + habit_factor + rng.normal(0, 0.05, size=n))
    weight_series = profile.weight_baseline_kg + drift
    weight[weighed] = weight_series[weighed].astype(np.float32)

    return {
        "water_cups": water_cups,
        "water_logged": water_logged.astype(np.bool_),
        "weight_kg": weight,                  # NaN means not weighed
        "weight_logged": weighed.astype(np.bool_),
    }
