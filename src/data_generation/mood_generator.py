"""Daily mood generator implementing the weighted formula from section 4.2.

mood[d] = clip(round(
    0.35 · sleep_quality_norm[d]
  + 0.20 · habits_completed_yesterday
  + 0.15 · budget_on_track_yesterday
  + 0.15 · day_of_week_effect[d]
  + 0.15 · noise
), 1, 5)
"""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile


def generate_mood(
    profile: UserProfile,
    days: np.ndarray,
    rng: np.random.Generator,
    sleep_quality: np.ndarray,        # 1..5 ints, length n
    habit_completion_rate: np.ndarray,  # 0..1, length n
    budget_on_track: np.ndarray,      # 0/1 ints, length n (per day)
) -> dict[str, np.ndarray]:
    n = len(days)
    weekday = np.array(
        [int(d.astype("datetime64[D]").astype("O").weekday()) for d in days]
    )
    is_weekend = (weekday >= 5).astype(np.float32)

    # Normalise components to roughly 1..5 scale before weighting.
    sleep_term = sleep_quality.astype(np.float32)               # 1..5

    habits_yest = np.empty(n, dtype=np.float32)
    habits_yest[0] = 0.5
    habits_yest[1:] = habit_completion_rate[:-1]
    habits_term = 1.0 + habits_yest * 4.0                       # 1..5

    bot_yest = np.empty(n, dtype=np.float32)
    bot_yest[0] = 1.0
    bot_yest[1:] = budget_on_track[:-1].astype(np.float32)
    budget_term = 1.0 + bot_yest * 4.0                          # 1..5

    dow_term = np.where(is_weekend.astype(bool), 4.0, 3.0).astype(np.float32)

    # Centre noise around the user's archetype mood baseline.
    base = profile.mood_avg
    noise = rng.normal(loc=base, scale=profile.mood_std, size=n)
    noise = np.clip(noise, 1.0, 5.0).astype(np.float32)

    raw = (
        0.35 * sleep_term
        + 0.20 * habits_term
        + 0.15 * budget_term
        + 0.15 * dow_term
        + 0.15 * noise
    )
    mood = np.clip(np.round(raw), 1, 5).astype(np.int32)

    log_prob = {
        "disciplined_professional": 0.92,
        "stressed_young_adult": 0.50,
        "health_focused": 0.95,
        "budget_conscious": 0.75,
        "chaotic_creative": 0.42,
        "elderly_health_tracker": 0.65,
    }.get(profile.archetype, 0.7)
    logged = rng.uniform(size=n) < log_prob

    return {
        "mood_score": mood,                            # 1..5 int
        "mood_logged": logged.astype(np.bool_),
        "is_weekend": is_weekend.astype(np.bool_),
    }
