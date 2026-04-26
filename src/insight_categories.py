"""Single source of truth for the 40 insight category codes the model emits.

Order matters — index in this list == output index in the multi-label head and
in the metadata JSON consumed by the Flutter inference engine.
"""

from __future__ import annotations

INSIGHT_CATEGORIES: dict[str, str] = {
    # Finance insights
    "FIN_OVERSPEND_RISK": "User likely to exceed budget before month end",
    "FIN_UNUSUAL_EXPENSE": "Unusually large expense detected in a category",
    "FIN_SAVINGS_POSITIVE": "User is saving more than usual this month",
    "FIN_CATEGORY_SPIKE": "Specific category spending spiked vs last month",
    "FIN_INCOME_IRREGULAR": "Income pattern is irregular this month",
    "FIN_BUDGET_ON_TRACK": "User is on track to stay within budget",
    # Health insights
    "HLT_WATER_LOW": "Water intake below target for 3+ days",
    "HLT_WEIGHT_TREND_UP": "Weight trending upward over 2 weeks",
    "HLT_WEIGHT_TREND_DOWN": "Weight trending downward — positive progress",
    "HLT_ACTIVITY_DROP": "Health logging frequency dropped",
    # Sleep insights
    "SLP_DEBT_ACCUMULATING": "Sleep debt building up over the week",
    "SLP_QUALITY_LOW": "Sleep quality below average for 5+ days",
    "SLP_CONSISTENCY_GOOD": "Consistent sleep schedule detected",
    "SLP_LATE_NIGHT_PATTERN": "Bedtime getting later each day",
    "SLP_MOOD_CORRELATION": "Poor sleep correlates with low mood for this user",
    # Mood insights
    "MOD_LOW_STREAK": "Mood has been low for 3+ consecutive days",
    "MOD_POSITIVE_STREAK": "Positive mood streak — reinforce habits causing it",
    "MOD_SLEEP_LINK": "Mood improves after good sleep nights",
    "MOD_FINANCE_STRESS": "Mood dips correlate with high spending days",
    "MOD_WEEKEND_PATTERN": "User mood consistently better on weekends",
    # Tasks & habits
    "TSK_OVERDUE_PILE": "Multiple overdue tasks accumulating",
    "TSK_COMPLETION_HIGH": "Task completion rate excellent this week",
    "HBT_STREAK_AT_RISK": "Habit streak about to break — no log today",
    "HBT_BEST_STREAK_EVER": "Personal best streak achieved",
    "HBT_MORNING_PATTERN": "User completes habits better in the morning",
    # Medications
    "MED_MISSED_DOSES": "Missed medication doses detected this week",
    "MED_COMPLIANCE_GOOD": "Excellent medication compliance this week",
    "MED_TIMING_IRREGULAR": "Medication taken at inconsistent times",
    # Goals
    "GOL_ON_TRACK": "Goal progress on track to hit deadline",
    "GOL_BEHIND_PACE": "Goal falling behind — needs attention",
    "GOL_MILESTONE_NEAR": "Next milestone within reach",
    "GOL_COMPLETED": "Goal completed",
    # Cross-module correlations
    "CRS_SLEEP_HABIT_LINK": "Better sleep => more habits completed next day",
    "CRS_EXERCISE_MOOD_LINK": "Exercise habit correlates with mood improvement",
    "CRS_SPENDING_MOOD_LINK": "Overspending days correlate with mood drops",
    "CRS_HYDRATION_ENERGY": "Good water intake correlates with task completion",
    "CRS_WEEKEND_PATTERN": "User performs significantly better on weekends",
    # Padding categories — kept reserved so the head stays at 40.
    "GEN_DATA_SUFFICIENT": "Enough data is logged for reliable insights",
    "GEN_DATA_THIN": "Recent logging is thin — keep tracking for richer insights",
    "GEN_NEW_USER": "New user — early observations only",
}

INSIGHT_CODES: list[str] = list(INSIGHT_CATEGORIES.keys())
NUM_INSIGHT_CLASSES: int = len(INSIGHT_CODES)
INSIGHT_INDEX: dict[str, int] = {code: i for i, code in enumerate(INSIGHT_CODES)}

# Severity classification for the Flutter formatter.
INSIGHT_SEVERITY: dict[str, str] = {
    # warnings
    "FIN_OVERSPEND_RISK": "warning",
    "FIN_UNUSUAL_EXPENSE": "warning",
    "FIN_INCOME_IRREGULAR": "warning",
    "FIN_CATEGORY_SPIKE": "warning",
    "HLT_WATER_LOW": "warning",
    "HLT_WEIGHT_TREND_UP": "warning",
    "HLT_ACTIVITY_DROP": "warning",
    "SLP_DEBT_ACCUMULATING": "warning",
    "SLP_QUALITY_LOW": "warning",
    "SLP_LATE_NIGHT_PATTERN": "warning",
    "MOD_LOW_STREAK": "warning",
    "MOD_FINANCE_STRESS": "warning",
    "TSK_OVERDUE_PILE": "warning",
    "HBT_STREAK_AT_RISK": "warning",
    "MED_MISSED_DOSES": "warning",
    "MED_TIMING_IRREGULAR": "warning",
    "GOL_BEHIND_PACE": "warning",
    "CRS_SPENDING_MOOD_LINK": "warning",
    "GEN_DATA_THIN": "info",
    # positives
    "FIN_SAVINGS_POSITIVE": "positive",
    "FIN_BUDGET_ON_TRACK": "positive",
    "HLT_WEIGHT_TREND_DOWN": "positive",
    "SLP_CONSISTENCY_GOOD": "positive",
    "MOD_POSITIVE_STREAK": "positive",
    "TSK_COMPLETION_HIGH": "positive",
    "HBT_BEST_STREAK_EVER": "positive",
    "MED_COMPLIANCE_GOOD": "positive",
    "GOL_ON_TRACK": "positive",
    "GOL_MILESTONE_NEAR": "positive",
    "GOL_COMPLETED": "positive",
    "GEN_DATA_SUFFICIENT": "positive",
}
# Anything else defaults to "info".
for _code in INSIGHT_CODES:
    INSIGHT_SEVERITY.setdefault(_code, "info")
