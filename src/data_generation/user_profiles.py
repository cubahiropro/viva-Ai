"""User archetypes used to seed synthetic Viva AI users.

Each archetype defines statistical distributions for finance / health / sleep /
mood / habits / tasks / medication that downstream generators sample from.

`USER_ARCHETYPES` mirrors section 4.1 of the training prompt verbatim.
`sample_user_profile()` returns a fully-instantiated profile drawing concrete
values from those distributions for a single synthetic user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

USER_ARCHETYPES: dict[str, dict[str, Any]] = {
    "disciplined_professional": {
        "description": "High income, consistent habits, good sleep, tracks everything",
        "weight": 0.15,
        "finance": {"monthly_income": (3000, 8000), "budget_adherence": (0.8, 1.0)},
        "sleep": {"avg_hours": (7.0, 8.5), "consistency": "high"},
        "habits": {"completion_rate": (0.75, 0.95), "streak_length": (15, 90)},
        "mood": {"avg_score": (3.5, 5.0), "volatility": "low"},
        "tasks": {"completion_rate": (0.7, 0.9)},
        "medication_compliance": (0.9, 1.0),
    },
    "stressed_young_adult": {
        "description": "Moderate income, irregular sleep, mood swings, overspends",
        "weight": 0.25,
        "finance": {"monthly_income": (800, 2500), "budget_adherence": (0.5, 0.85)},
        "sleep": {"avg_hours": (5.5, 7.5), "consistency": "low"},
        "habits": {"completion_rate": (0.3, 0.65), "streak_length": (1, 15)},
        "mood": {"avg_score": (2.0, 3.5), "volatility": "high"},
        "tasks": {"completion_rate": (0.4, 0.7)},
        "medication_compliance": (0.5, 0.8),
    },
    "health_focused": {
        "description": "Prioritises health metrics, tracks water/weight/sleep carefully",
        "weight": 0.15,
        "finance": {"monthly_income": (1500, 4000), "budget_adherence": (0.65, 0.9)},
        "sleep": {"avg_hours": (7.5, 9.0), "consistency": "high"},
        "habits": {"completion_rate": (0.7, 0.95), "streak_length": (20, 120)},
        "mood": {"avg_score": (3.5, 5.0), "volatility": "low"},
        "tasks": {"completion_rate": (0.5, 0.75)},
        "medication_compliance": (0.85, 1.0),
    },
    "budget_conscious": {
        "description": "Low income, very careful with money, tracks expenses obsessively",
        "weight": 0.20,
        "finance": {"monthly_income": (400, 1200), "budget_adherence": (0.85, 1.0)},
        "sleep": {"avg_hours": (6.0, 8.0), "consistency": "medium"},
        "habits": {"completion_rate": (0.5, 0.75), "streak_length": (5, 30)},
        "mood": {"avg_score": (2.5, 4.0), "volatility": "medium"},
        "tasks": {"completion_rate": (0.55, 0.8)},
        "medication_compliance": (0.7, 0.95),
    },
    "chaotic_creative": {
        "description": "Irregular everything — inconsistent but passionate bursts",
        "weight": 0.15,
        "finance": {"monthly_income": (600, 3000), "budget_adherence": (0.3, 0.75)},
        "sleep": {"avg_hours": (4.5, 9.0), "consistency": "very_low"},
        "habits": {"completion_rate": (0.2, 0.6), "streak_length": (1, 10)},
        "mood": {"avg_score": (1.5, 5.0), "volatility": "very_high"},
        "tasks": {"completion_rate": (0.3, 0.65)},
        "medication_compliance": (0.3, 0.7),
    },
    "elderly_health_tracker": {
        "description": "Older user, medications critical, health-focused, low tech usage",
        "weight": 0.10,
        "finance": {"monthly_income": (500, 2000), "budget_adherence": (0.7, 0.95)},
        "sleep": {"avg_hours": (6.0, 8.5), "consistency": "medium"},
        "habits": {"completion_rate": (0.6, 0.85), "streak_length": (10, 60)},
        "mood": {"avg_score": (2.5, 4.0), "volatility": "low"},
        "tasks": {"completion_rate": (0.5, 0.75)},
        "medication_compliance": (0.8, 1.0),
    },
}


_CONSISTENCY_TO_STD = {
    "high": 0.35,
    "medium": 0.6,
    "low": 1.0,
    "very_low": 1.5,
}

_VOLATILITY_TO_STD = {
    "low": 0.3,
    "medium": 0.55,
    "high": 0.9,
    "very_high": 1.3,
}


@dataclass
class UserProfile:
    """A concrete sampled user profile."""

    user_id: int
    archetype: str
    monthly_income: float
    budget_adherence: float           # 0..1, target ratio of spent/budget
    sleep_avg_hours: float
    sleep_std_hours: float            # day-to-day variability
    habit_completion_rate: float
    habit_streak_target: int
    mood_avg: float                   # 1..5
    mood_std: float
    task_completion_rate: float
    medication_compliance: float
    num_habits: int                   # 2..6
    num_medications: int              # 0..4
    has_critical_medications: bool
    weight_baseline_kg: float
    height_cm: float
    water_goal_cups: int = 8
    sex: str = "unspecified"
    age: int = 30
    extras: dict[str, Any] = field(default_factory=dict)


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _pick_archetype(rng: np.random.Generator) -> str:
    names = list(USER_ARCHETYPES.keys())
    weights = np.array(
        [USER_ARCHETYPES[n]["weight"] for n in names], dtype=np.float64
    )
    weights = weights / weights.sum()
    idx = int(rng.choice(len(names), p=weights))
    return names[idx]


def sample_user_profile(
    user_id: int,
    rng: np.random.Generator,
    archetype: str | None = None,
) -> UserProfile:
    """Draw a concrete profile for one user, optionally forcing an archetype."""
    name = archetype or _pick_archetype(rng)
    a = USER_ARCHETYPES[name]

    sleep_std = _CONSISTENCY_TO_STD[a["sleep"]["consistency"]]
    mood_std = _VOLATILITY_TO_STD[a["mood"]["volatility"]]

    age = int(rng.integers(18, 75))
    if name == "elderly_health_tracker":
        age = int(rng.integers(60, 85))
    if name == "stressed_young_adult":
        age = int(rng.integers(18, 32))

    sex = str(rng.choice(["female", "male", "other"], p=[0.49, 0.49, 0.02]))
    height_cm = float(rng.normal(170, 9))
    weight_kg = float(np.clip(rng.normal(72, 12), 45, 130))

    num_habits = int(rng.integers(2, 7))
    if name == "chaotic_creative":
        num_meds = int(rng.choice([0, 0, 0, 1], p=[0.6, 0.2, 0.1, 0.1]))
    elif name == "elderly_health_tracker":
        num_meds = int(rng.integers(2, 5))
    else:
        num_meds = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]))
    has_critical = num_meds >= 2 or name == "elderly_health_tracker"

    return UserProfile(
        user_id=user_id,
        archetype=name,
        monthly_income=_u(rng, a["finance"]["monthly_income"]),
        budget_adherence=_u(rng, a["finance"]["budget_adherence"]),
        sleep_avg_hours=_u(rng, a["sleep"]["avg_hours"]),
        sleep_std_hours=sleep_std,
        habit_completion_rate=_u(rng, a["habits"]["completion_rate"]),
        habit_streak_target=int(_u(rng, a["habits"]["streak_length"])),
        mood_avg=_u(rng, a["mood"]["avg_score"]),
        mood_std=mood_std,
        task_completion_rate=_u(rng, a["tasks"]["completion_rate"]),
        medication_compliance=_u(rng, a["medication_compliance"]),
        num_habits=num_habits,
        num_medications=num_meds,
        has_critical_medications=has_critical,
        weight_baseline_kg=weight_kg,
        height_cm=height_cm,
        sex=sex,
        age=age,
    )


def archetype_weights() -> dict[str, float]:
    return {n: float(USER_ARCHETYPES[n]["weight"]) for n in USER_ARCHETYPES}
