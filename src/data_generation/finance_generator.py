"""Daily finance series generator.

Produces, for one user over `num_days` days:
  - daily total spend
  - per-category spend (Food / Transport / Bills / Health / Shopping / Other)
  - number of transactions that day
  - income deposit on the 1st or 15th of each month
  - a per-month budget array (broadcast to each day in that month)
  - an "unusual expense" boolean flag for the synthetic 3–5 anomalies/year

Weekend spending is ~20% higher on average. December gets a holiday spike.
"""

from __future__ import annotations

import numpy as np

from .user_profiles import UserProfile

# Order matches the 6-element vector exposed downstream as a feature.
EXPENSE_CATEGORIES: tuple[str, ...] = (
    "food",
    "transport",
    "bills",
    "health",
    "shopping",
    "other",
)
_CATEGORY_PRIOR = np.array([0.35, 0.15, 0.10, 0.10, 0.20, 0.10])


def generate_finance(
    profile: UserProfile,
    days: np.ndarray,             # 1D array of pandas Timestamps or datetime64[D]
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(days)
    months = np.array([np.datetime64(d, "M") for d in days])
    unique_months, month_idx = np.unique(months, return_inverse=True)
    num_months = len(unique_months)

    # Monthly budget = monthly_income × (0.6..0.95) — represents discretionary cap.
    budget_ratio = rng.uniform(0.6, 0.95, size=num_months)
    monthly_budgets = profile.monthly_income * budget_ratio
    daily_budget = monthly_budgets[month_idx]

    # Income deposits — pay day 1 or 15 each month.
    income_today = np.zeros(n, dtype=np.float32)
    pay_day = int(rng.choice([1, 15]))
    days_of_month = np.array([int(np.datetime_as_string(d, unit="D")[-2:]) for d in days])
    pay_mask = days_of_month == pay_day
    income_today[pay_mask] = profile.monthly_income

    # Some chaotic_creative or stressed_young_adult users have irregular income —
    # drop ~25% of pay days and randomise some others.
    if profile.archetype in ("chaotic_creative", "stressed_young_adult"):
        drop_mask = rng.uniform(size=n) < 0.18
        income_today[pay_mask & drop_mask] = 0.0
        rand_pay_idx = rng.choice(n, size=max(1, n // 90), replace=False)
        income_today[rand_pay_idx] += profile.monthly_income * rng.uniform(
            0.3, 0.7, size=len(rand_pay_idx)
        )

    # Weekend / December multipliers.
    weekday = np.array([int(d.astype("datetime64[D]").astype("O").weekday()) for d in days])
    is_weekend = (weekday >= 5).astype(np.float32)
    month_of_year = np.array([int(np.datetime_as_string(d, unit="M")[-2:]) for d in days])
    is_december = (month_of_year == 12).astype(np.float32)
    multiplier = 1.0 + 0.20 * is_weekend + 0.25 * is_december

    # Number of transactions/day: Poisson around archetype-dependent mean.
    base_lambda = {
        "disciplined_professional": 1.6,
        "stressed_young_adult": 2.4,
        "health_focused": 1.8,
        "budget_conscious": 1.2,
        "chaotic_creative": 2.7,
        "elderly_health_tracker": 1.0,
    }.get(profile.archetype, 1.8)
    num_tx = rng.poisson(base_lambda, size=n).clip(0, 8).astype(np.int32)

    # Daily spend mean = monthly_budget / 30 × adherence × multiplier
    daily_mean = (daily_budget / 30.0) * profile.budget_adherence * multiplier

    # Heavy-tailed lognormal for a sense of realism.
    sigma = 0.55
    spend_per_day = rng.lognormal(mean=np.log(np.maximum(daily_mean, 1e-3)), sigma=sigma)
    spend_per_day = np.where(num_tx == 0, 0.0, spend_per_day).astype(np.float32)

    # Inject 3–5 unusual large expenses spread over the year.
    n_unusual = int(rng.integers(3, 6))
    unusual_mask = np.zeros(n, dtype=np.bool_)
    if n_unusual > 0:
        unusual_idx = rng.choice(n, size=min(n_unusual, n), replace=False)
        unusual_mask[unusual_idx] = True
        spend_per_day[unusual_idx] += rng.uniform(0.4, 0.9, size=len(unusual_idx)) * (
            profile.monthly_income * 0.3
        )

    # Split each day's spend across the 6 categories using a Dirichlet-perturbed
    # version of the prior — not deterministic but reasonably stable.
    category_spend = np.zeros((n, len(EXPENSE_CATEGORIES)), dtype=np.float32)
    for i in range(n):
        if spend_per_day[i] <= 0:
            continue
        alpha = _CATEGORY_PRIOR * 5.0
        if month_of_year[i] == 12:                 # December bumps shopping share
            alpha = alpha.copy()
            alpha[4] *= 2.0
        if is_weekend[i]:                          # Weekend bumps food share a bit
            alpha = alpha.copy()
            alpha[0] *= 1.4
        share = rng.dirichlet(alpha)
        category_spend[i] = (share * spend_per_day[i]).astype(np.float32)

    return {
        "daily_total_spend": spend_per_day.astype(np.float32),
        "num_transactions": num_tx,
        "category_spend": category_spend,                           # (n, 6)
        "income_today": income_today.astype(np.float32),
        "monthly_budget": daily_budget.astype(np.float32),          # broadcast to days
        "month_index": month_idx.astype(np.int32),
        "is_unusual_expense": unusual_mask,
        "is_weekend": is_weekend.astype(np.bool_),
    }
