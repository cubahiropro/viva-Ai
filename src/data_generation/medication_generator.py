"""Daily medication compliance + timing generator."""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile


def generate_medications(
    profile: UserProfile,
    days: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(days)
    n_meds = profile.num_medications

    if n_meds == 0:
        return {
            "doses_scheduled": np.zeros(n, dtype=np.int32),
            "doses_taken": np.zeros(n, dtype=np.int32),
            "timing_offset_minutes": np.zeros(n, dtype=np.float32),
            "has_critical_medications": np.zeros(n, dtype=np.bool_),
        }

    doses_scheduled = np.full(n, n_meds, dtype=np.int32)
    p = profile.medication_compliance
    doses_taken = rng.binomial(n_meds, p, size=n).astype(np.int32)

    # Timing irregularity: Gaussian noise around scheduled time.
    sigma = {
        "disciplined_professional": 8.0,
        "stressed_young_adult": 35.0,
        "health_focused": 6.0,
        "budget_conscious": 12.0,
        "chaotic_creative": 60.0,
        "elderly_health_tracker": 10.0,
    }.get(profile.archetype, 15.0)
    timing_offsets = rng.normal(0, sigma, size=n).astype(np.float32)
    # Convert to absolute minutes-late since "irregularity" is unsigned.
    timing_offsets = np.abs(timing_offsets)

    return {
        "doses_scheduled": doses_scheduled,
        "doses_taken": doses_taken,
        "timing_offset_minutes": timing_offsets,
        "has_critical_medications": np.full(
            n, profile.has_critical_medications, dtype=np.bool_
        ),
    }
