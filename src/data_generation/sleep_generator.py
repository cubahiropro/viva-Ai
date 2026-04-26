"""Daily sleep generator. Sleep[d] is the night going *into* day d."""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile


def generate_sleep(
    profile: UserProfile,
    days: np.ndarray,
    rng: np.random.Generator,
    prev_day_mood: np.ndarray | None = None,  # mood[d-1], 1..5; same length as days
) -> dict[str, np.ndarray]:
    n = len(days)
    weekday = np.array(
        [int(d.astype("datetime64[D]").astype("O").weekday()) for d in days]
    )
    is_weekend_night = (weekday >= 5).astype(np.float32)   # Fri->Sat & Sat->Sun

    # Duration: archetype mean + day-of-week effect + noise.
    duration_mu = profile.sleep_avg_hours + 0.4 * is_weekend_night
    duration = rng.normal(duration_mu, profile.sleep_std_hours, size=n)
    duration = np.clip(duration, 3.0, 11.0).astype(np.float32)

    # Bedtime hour (in hours after 8pm, so 0=20:00 and 8=04:00).
    base_bedtime = {
        "disciplined_professional": 2.5,    # ~22:30
        "stressed_young_adult": 4.5,        # ~00:30
        "health_focused": 2.0,              # ~22:00
        "budget_conscious": 3.0,            # ~23:00
        "chaotic_creative": 5.0,            # ~01:00, very variable
        "elderly_health_tracker": 1.5,      # ~21:30
    }.get(profile.archetype, 3.0)

    # Bedtime drift later as the week progresses, max delay on Friday/Saturday.
    week_progress = (weekday / 6.0).astype(np.float32)
    bedtime_offset = base_bedtime + 0.6 * week_progress + 0.7 * is_weekend_night
    bedtime_hour_after_20 = rng.normal(bedtime_offset, profile.sleep_std_hours * 0.8)
    bedtime_hour_after_20 = np.clip(bedtime_hour_after_20, 0.0, 8.0).astype(np.float32)

    wake_hour = (bedtime_hour_after_20 + duration) % 24.0

    # Quality 1..5 — degraded by short sleep, weekend social disruption,
    # and previous day's low mood (if provided).
    quality = np.full(n, 3.0, dtype=np.float32)
    duration_dev = duration - 7.5
    quality += np.clip(duration_dev * 0.4, -1.5, 1.0)
    quality -= 0.3 * is_weekend_night

    if prev_day_mood is not None:
        prev = np.asarray(prev_day_mood, dtype=np.float32)
        if prev.shape[0] == n:
            mood_dev = (prev - 3.0) / 2.0
            quality += 0.3 * mood_dev

    quality += rng.normal(0, 0.4, size=n)
    quality = np.clip(np.round(quality), 1, 5).astype(np.int32)

    log_prob = {
        "disciplined_professional": 0.95,
        "stressed_young_adult": 0.50,
        "health_focused": 0.97,
        "budget_conscious": 0.78,
        "chaotic_creative": 0.45,
        "elderly_health_tracker": 0.70,
    }.get(profile.archetype, 0.7)
    sleep_logged = rng.uniform(size=n) < log_prob

    return {
        "sleep_duration_hours": duration,
        "sleep_quality": quality,                       # 1..5 int
        "bedtime_hour_after_20": bedtime_hour_after_20, # 0..8
        "wake_hour": wake_hour.astype(np.float32),
        "sleep_logged": sleep_logged.astype(np.bool_),
        "is_weekend_night": is_weekend_night.astype(np.bool_),
    }
