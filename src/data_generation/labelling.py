"""Ground-truth labelling functions for all 40 insight categories.

Each `label_*` function takes a `UserSeries` (the per-user dict-of-arrays
produced by the master generator) and returns a `np.ndarray` of shape
`(num_days,)` with 0/1 binary labels.

The full labeller `label_all_categories(series)` returns a `(num_days, 40)`
binary matrix indexed by `INSIGHT_CODES`.
"""

from __future__ import annotations

import numpy as np

from insight_categories import INSIGHT_CODES, NUM_INSIGHT_CLASSES

# ------------------------------------------------------------------ helpers


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum of length `window`."""
    out = np.zeros_like(arr, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(arr.astype(np.float64))])
    for i in range(len(arr)):
        a = max(0, i - window + 1)
        out[i] = cs[i + 1] - cs[a]
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        a = max(0, i - window + 1)
        out[i] = float(arr[a : i + 1].mean()) if i >= a else 0.0
    return out


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of arr over trailing `window` points (per index of arr)."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = max(0, i - window + 1)
        seg = arr[a : i + 1].astype(np.float64)
        if len(seg) < 3:
            continue
        x = np.arange(len(seg), dtype=np.float64)
        x_mean = x.mean()
        y_mean = seg.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 1e-12:
            continue
        out[i] = float(((x - x_mean) * (seg - y_mean)).sum()) / denom
    return out


def _rolling_pearson(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Trailing Pearson correlation between a and b. Returns 0 when undefined."""
    n = len(a)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            continue
        x = a[i - window + 1 : i + 1].astype(np.float64)
        y = b[i - window + 1 : i + 1].astype(np.float64)
        sx, sy = x.std(), y.std()
        if sx <= 1e-9 or sy <= 1e-9:
            continue
        out[i] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return out


def _safe_div(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, default, dtype=np.float64)
    mask = b > 1e-9
    out[mask] = a[mask] / b[mask]
    return out


# ------------------------------------------------------------------ FINANCE


def label_fin_overspend_risk(s: dict) -> np.ndarray:
    """At each day d, project pace * 30 against this month's budget."""
    spend = s["daily_total_spend"]
    budget = s["monthly_budget"]
    month_idx = s["month_index"]
    n = len(spend)
    out = np.zeros(n, dtype=np.uint8)
    cur_month_total = 0.0
    cur_month_days = 0
    last_month = -1
    for i in range(n):
        if month_idx[i] != last_month:
            cur_month_total = 0.0
            cur_month_days = 0
            last_month = int(month_idx[i])
        cur_month_total += float(spend[i])
        cur_month_days += 1
        if cur_month_days < 5:                  # too early to call
            continue
        pace = cur_month_total / cur_month_days
        projected = pace * 30.0
        if projected > float(budget[i]) * 1.05:
            out[i] = 1
    return out


def label_fin_unusual_expense(s: dict) -> np.ndarray:
    spend = s["daily_total_spend"]
    avg = _rolling_mean(spend, 30)
    std = np.zeros_like(avg)
    for i in range(len(spend)):
        a = max(0, i - 29)
        seg = spend[a : i + 1]
        std[i] = float(seg.std()) if len(seg) > 1 else 0.0
    z = np.where(std > 1e-3, (spend - avg) / np.maximum(std, 1e-3), 0)
    return (z > 2.5).astype(np.uint8)


def label_fin_savings_positive(s: dict) -> np.ndarray:
    """User saved more than 25 % of monthly income this month vs last month."""
    spend = s["daily_total_spend"]
    income = s["income_today"]
    month_idx = s["month_index"]
    n = len(spend)
    monthly_spend: dict[int, float] = {}
    monthly_income: dict[int, float] = {}
    for i in range(n):
        m = int(month_idx[i])
        monthly_spend[m] = monthly_spend.get(m, 0.0) + float(spend[i])
        monthly_income[m] = monthly_income.get(m, 0.0) + float(income[i])
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        m = int(month_idx[i])
        sp = monthly_spend[m]
        inc = monthly_income[m]
        if inc <= 1.0:
            continue
        savings_rate = (inc - sp) / inc
        if savings_rate >= 0.25:
            out[i] = 1
    return out


def label_fin_category_spike(s: dict) -> np.ndarray:
    """Any category at >= 1.6× its 60-day moving average."""
    cat = s["category_spend"]            # (n, 6)
    n = cat.shape[0]
    out = np.zeros(n, dtype=np.uint8)
    for c in range(cat.shape[1]):
        avg = _rolling_mean(cat[:, c], 60)
        spike = cat[:, c] > avg * 1.6
        early_data = (avg > 1.0) & (np.arange(n) >= 30)
        out |= (spike & early_data).astype(np.uint8)
    return out


def label_fin_income_irregular(s: dict) -> np.ndarray:
    """Income deposits did not arrive on the expected pay day(s) this month."""
    income = s["income_today"]
    month_idx = s["month_index"]
    n = len(income)
    out = np.zeros(n, dtype=np.uint8)
    months = np.unique(month_idx)
    income_by_month = {int(m): float(income[month_idx == m].sum()) for m in months}
    pay_days_by_month: dict[int, int] = {}
    for m in months:
        pay_days_by_month[int(m)] = int((income[month_idx == m] > 0).sum())
    for i in range(n):
        m = int(month_idx[i])
        if pay_days_by_month[m] == 0 or income_by_month[m] < 100.0:
            out[i] = 1
    return out


def label_fin_budget_on_track(s: dict) -> np.ndarray:
    """Inverse of overspend risk — projection within 95% of budget."""
    spend = s["daily_total_spend"]
    budget = s["monthly_budget"]
    month_idx = s["month_index"]
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
        if days < 5:
            continue
        projected = (total / days) * 30.0
        if projected <= float(budget[i]) * 0.95:
            out[i] = 1
    return out


# ------------------------------------------------------------------ HEALTH


def label_hlt_water_low(s: dict) -> np.ndarray:
    water = s["water_cups"].astype(np.float32)
    goal = 8.0
    n = len(water)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 2:
            continue
        if np.all(water[i - 2 : i + 1] < goal * 0.6):
            out[i] = 1
    return out


def label_hlt_weight_trend_up(s: dict) -> np.ndarray:
    weight = s["weight_kg"]
    n = len(weight)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 14:
            continue
        seg = weight[i - 13 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 3:
            continue
        x = np.arange(len(valid), dtype=np.float64)
        x_mean = x.mean()
        y_mean = valid.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 1e-9:
            continue
        slope = float(((x - x_mean) * (valid - y_mean)).sum()) / denom
        if slope > 0.05:                # ~ +50 g per measurement
            out[i] = 1
    return out


def label_hlt_weight_trend_down(s: dict) -> np.ndarray:
    weight = s["weight_kg"]
    n = len(weight)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 14:
            continue
        seg = weight[i - 13 : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 3:
            continue
        x = np.arange(len(valid), dtype=np.float64)
        x_mean = x.mean()
        y_mean = valid.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom <= 1e-9:
            continue
        slope = float(((x - x_mean) * (valid - y_mean)).sum()) / denom
        if slope < -0.05:
            out[i] = 1
    return out


def label_hlt_activity_drop(s: dict) -> np.ndarray:
    """Logging frequency dropped vs prior 30 days."""
    logged = (
        s["water_logged"].astype(np.uint8) | s["weight_logged"].astype(np.uint8)
    ).astype(np.float32)
    last_7 = _rolling_mean(logged, 7)
    prior_30 = _rolling_mean(logged, 30)
    n = len(logged)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 30:
            continue
        if prior_30[i] > 0.4 and last_7[i] < prior_30[i] * 0.5:
            out[i] = 1
    return out


# ------------------------------------------------------------------ SLEEP


def label_slp_debt_accumulating(s: dict) -> np.ndarray:
    duration = s["sleep_duration_hours"]
    debt = np.maximum(0.0, 7.0 - duration)
    cum = _rolling_sum(debt, 7)
    return (cum >= 5.0).astype(np.uint8)


def label_slp_quality_low(s: dict) -> np.ndarray:
    quality = s["sleep_quality"].astype(np.float32)
    n = len(quality)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 4:
            continue
        if np.all(quality[i - 4 : i + 1] <= 2):
            out[i] = 1
    return out


def label_slp_consistency_good(s: dict) -> np.ndarray:
    bedtime = s["bedtime_hour_after_20"]
    n = len(bedtime)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 6:
            continue
        seg = bedtime[i - 6 : i + 1]
        if seg.std() < 0.6 and seg.mean() < 4.5:
            out[i] = 1
    return out


def label_slp_late_night_pattern(s: dict) -> np.ndarray:
    bedtime = s["bedtime_hour_after_20"]
    slope = _rolling_slope(bedtime, 7)
    n = len(bedtime)
    out = np.zeros(n, dtype=np.uint8)
    out[(slope > 0.15) & (np.arange(n) >= 7)] = 1
    return out


def label_slp_mood_correlation(s: dict) -> np.ndarray:
    sleep_quality = s["sleep_quality"].astype(np.float32)
    mood = s["mood_score"].astype(np.float32)
    corr = _rolling_pearson(sleep_quality, mood, 14)
    return (corr > 0.5).astype(np.uint8)


# ------------------------------------------------------------------ MOOD


def label_mod_low_streak(s: dict) -> np.ndarray:
    mood = s["mood_score"].astype(np.float32)
    n = len(mood)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 2:
            continue
        if np.all(mood[i - 2 : i + 1] <= 2):
            out[i] = 1
    return out


def label_mod_positive_streak(s: dict) -> np.ndarray:
    mood = s["mood_score"].astype(np.float32)
    n = len(mood)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 4:
            continue
        if np.all(mood[i - 4 : i + 1] >= 4):
            out[i] = 1
    return out


def label_mod_sleep_link(s: dict) -> np.ndarray:
    """Average mood after good-sleep nights >= 0.5 above mood after poor-sleep nights."""
    sleep = s["sleep_quality"].astype(np.float32)
    mood = s["mood_score"].astype(np.float32)
    n = len(sleep)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 13:
            continue
        win_sleep = sleep[i - 13 : i + 1]
        win_mood = mood[i - 13 : i + 1]
        good = win_sleep >= 4
        bad = win_sleep <= 2
        if good.sum() < 3 or bad.sum() < 3:
            continue
        if win_mood[good].mean() - win_mood[bad].mean() >= 0.5:
            out[i] = 1
    return out


def label_mod_finance_stress(s: dict) -> np.ndarray:
    spend = s["daily_total_spend"].astype(np.float32)
    mood = s["mood_score"].astype(np.float32)
    corr = _rolling_pearson(spend, mood, 14)
    return (corr < -0.4).astype(np.uint8)


def label_mod_weekend_pattern(s: dict) -> np.ndarray:
    mood = s["mood_score"].astype(np.float32)
    is_weekend = s["is_weekend"].astype(bool)
    n = len(mood)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 27:
            continue
        win_mood = mood[i - 27 : i + 1]
        win_we = is_weekend[i - 27 : i + 1]
        if win_we.sum() < 4 or (~win_we).sum() < 4:
            continue
        if win_mood[win_we].mean() - win_mood[~win_we].mean() >= 0.5:
            out[i] = 1
    return out


# ------------------------------------------------------------------ TASKS / HABITS


def label_tsk_overdue_pile(s: dict) -> np.ndarray:
    return (s["tasks_overdue"] >= 5).astype(np.uint8)


def label_tsk_completion_high(s: dict) -> np.ndarray:
    rate = s["task_completion_rate_today"].astype(np.float32)
    avg7 = _rolling_mean(rate, 7)
    return (avg7 >= 0.8).astype(np.uint8)


def label_hbt_streak_at_risk(s: dict) -> np.ndarray:
    """User had a streak >=5 yesterday but hasn't completed all habits today."""
    streak = s["habit_streak"]
    rate = s["habit_completion_rate"]
    n = len(streak)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(1, n):
        if streak[i - 1] >= 5 and rate[i] < 1.0 and streak[i] == 0:
            out[i] = 1
    return out


def label_hbt_best_streak_ever(s: dict) -> np.ndarray:
    streak = s["habit_streak"].astype(np.int32)
    longest = s["habit_longest_ever"].astype(np.int32)
    n = len(streak)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if streak[i] >= 7 and streak[i] == longest[i]:
            out[i] = 1
    return out


def label_hbt_morning_pattern(s: dict) -> np.ndarray:
    morning = s["morning_habit_rate"].astype(np.float32)
    evening = s["evening_habit_rate"].astype(np.float32)
    avg7_m = _rolling_mean(morning, 14)
    avg7_e = _rolling_mean(evening, 14)
    return ((avg7_m - avg7_e) >= 0.2).astype(np.uint8)


# ------------------------------------------------------------------ MEDICATIONS


def label_med_missed_doses(s: dict) -> np.ndarray:
    sched = s["doses_scheduled"].astype(np.float32)
    taken = s["doses_taken"].astype(np.float32)
    missed = sched - taken
    cum = _rolling_sum(missed, 7)
    has = sched > 0
    return (has.astype(np.uint8) & (cum >= 3).astype(np.uint8))


def label_med_compliance_good(s: dict) -> np.ndarray:
    sched = s["doses_scheduled"].astype(np.float32)
    taken = s["doses_taken"].astype(np.float32)
    cum_sched = _rolling_sum(sched, 7)
    cum_taken = _rolling_sum(taken, 7)
    rate = _safe_div(cum_taken, cum_sched, default=0.0)
    return ((rate >= 0.95) & (cum_sched > 0)).astype(np.uint8)


def label_med_timing_irregular(s: dict) -> np.ndarray:
    offsets = s["timing_offset_minutes"].astype(np.float32)
    has = s["doses_scheduled"] > 0
    avg7 = _rolling_mean(offsets, 7)
    return (has.astype(np.uint8) & (avg7 >= 25.0).astype(np.uint8))


# ------------------------------------------------------------------ GOALS (synthetic-only)


def label_gol_on_track(s: dict) -> np.ndarray:
    """Synthetic: derived from habit completion + task completion."""
    h = _rolling_mean(s["habit_completion_rate"], 14)
    t = _rolling_mean(s["task_completion_rate_today"], 14)
    return ((h >= 0.7) & (t >= 0.7)).astype(np.uint8)


def label_gol_behind_pace(s: dict) -> np.ndarray:
    h = _rolling_mean(s["habit_completion_rate"], 14)
    t = _rolling_mean(s["task_completion_rate_today"], 14)
    return ((h < 0.45) & (t < 0.45)).astype(np.uint8)


def label_gol_milestone_near(s: dict) -> np.ndarray:
    """Triggered at the day a 7-day perfect-habit streak is reached."""
    streak = s["habit_streak"]
    return (streak == 7).astype(np.uint8)


def label_gol_completed(s: dict) -> np.ndarray:
    """Triggered while a 30-day-plus perfect habit streak is active.

    The "Goal completed — celebrate" insight stays true for as long as the
    user holds the milestone streak, ensuring sufficient positive density
    for training without changing the underlying streak generator.
    """
    streak = s["habit_streak"].astype(np.int32)
    return (streak >= 30).astype(np.uint8)


# ------------------------------------------------------------------ CROSS-MODULE


def label_crs_sleep_habit_link(s: dict) -> np.ndarray:
    """Sleep quality at d-1 correlates with habit completion at d (14-day window)."""
    sleep = s["sleep_quality"].astype(np.float32)
    habits = s["habit_completion_rate"].astype(np.float32)
    sleep_lag = np.empty_like(sleep)
    sleep_lag[0] = sleep[0]
    sleep_lag[1:] = sleep[:-1]
    corr = _rolling_pearson(sleep_lag, habits, 14)
    return (corr > 0.4).astype(np.uint8)


def label_crs_exercise_mood_link(s: dict) -> np.ndarray:
    """Habit completion correlates with mood (proxy for exercise habit)."""
    h = s["habit_completion_rate"].astype(np.float32)
    m = s["mood_score"].astype(np.float32)
    corr = _rolling_pearson(h, m, 14)
    return (corr > 0.4).astype(np.uint8)


def label_crs_spending_mood_link(s: dict) -> np.ndarray:
    return label_mod_finance_stress(s)


def label_crs_hydration_energy(s: dict) -> np.ndarray:
    water = s["water_cups"].astype(np.float32)
    tasks = s["task_completion_rate_today"].astype(np.float32)
    corr = _rolling_pearson(water, tasks, 14)
    return (corr > 0.4).astype(np.uint8)


def label_crs_weekend_pattern(s: dict) -> np.ndarray:
    """Combined weekend uplift across mood + habits + tasks."""
    is_weekend = s["is_weekend"].astype(bool)
    composite = (
        s["mood_score"].astype(np.float32)
        + s["habit_completion_rate"].astype(np.float32) * 5
        + s["task_completion_rate_today"].astype(np.float32) * 5
    ) / 3.0
    n = len(composite)
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if i < 27:
            continue
        win = composite[i - 27 : i + 1]
        we = is_weekend[i - 27 : i + 1]
        if we.sum() < 4 or (~we).sum() < 4:
            continue
        if win[we].mean() - win[~we].mean() >= 0.4:
            out[i] = 1
    return out


# ------------------------------------------------------------------ Generic / utility


def label_gen_data_sufficient(s: dict) -> np.ndarray:
    """Most modules logged in the last 7 days."""
    flags = (
        s["water_logged"].astype(np.uint8)
        + s["weight_logged"].astype(np.uint8)
        + s["sleep_logged"].astype(np.uint8)
        + s["mood_logged"].astype(np.uint8)
    )
    avg = _rolling_mean(flags.astype(np.float32), 7)
    return (avg >= 2.0).astype(np.uint8)


def label_gen_data_thin(s: dict) -> np.ndarray:
    flags = (
        s["water_logged"].astype(np.uint8)
        + s["weight_logged"].astype(np.uint8)
        + s["sleep_logged"].astype(np.uint8)
        + s["mood_logged"].astype(np.uint8)
    )
    avg = _rolling_mean(flags.astype(np.float32), 7)
    return ((avg < 1.0) & (np.arange(len(flags)) >= 7)).astype(np.uint8)


def label_gen_new_user(s: dict) -> np.ndarray:
    n = len(s["mood_score"])
    out = np.zeros(n, dtype=np.uint8)
    out[: min(7, n)] = 1
    return out


# ------------------------------------------------------------------ Registry

LABEL_FUNCTIONS: dict[str, callable] = {
    "FIN_OVERSPEND_RISK": label_fin_overspend_risk,
    "FIN_UNUSUAL_EXPENSE": label_fin_unusual_expense,
    "FIN_SAVINGS_POSITIVE": label_fin_savings_positive,
    "FIN_CATEGORY_SPIKE": label_fin_category_spike,
    "FIN_INCOME_IRREGULAR": label_fin_income_irregular,
    "FIN_BUDGET_ON_TRACK": label_fin_budget_on_track,
    "HLT_WATER_LOW": label_hlt_water_low,
    "HLT_WEIGHT_TREND_UP": label_hlt_weight_trend_up,
    "HLT_WEIGHT_TREND_DOWN": label_hlt_weight_trend_down,
    "HLT_ACTIVITY_DROP": label_hlt_activity_drop,
    "SLP_DEBT_ACCUMULATING": label_slp_debt_accumulating,
    "SLP_QUALITY_LOW": label_slp_quality_low,
    "SLP_CONSISTENCY_GOOD": label_slp_consistency_good,
    "SLP_LATE_NIGHT_PATTERN": label_slp_late_night_pattern,
    "SLP_MOOD_CORRELATION": label_slp_mood_correlation,
    "MOD_LOW_STREAK": label_mod_low_streak,
    "MOD_POSITIVE_STREAK": label_mod_positive_streak,
    "MOD_SLEEP_LINK": label_mod_sleep_link,
    "MOD_FINANCE_STRESS": label_mod_finance_stress,
    "MOD_WEEKEND_PATTERN": label_mod_weekend_pattern,
    "TSK_OVERDUE_PILE": label_tsk_overdue_pile,
    "TSK_COMPLETION_HIGH": label_tsk_completion_high,
    "HBT_STREAK_AT_RISK": label_hbt_streak_at_risk,
    "HBT_BEST_STREAK_EVER": label_hbt_best_streak_ever,
    "HBT_MORNING_PATTERN": label_hbt_morning_pattern,
    "MED_MISSED_DOSES": label_med_missed_doses,
    "MED_COMPLIANCE_GOOD": label_med_compliance_good,
    "MED_TIMING_IRREGULAR": label_med_timing_irregular,
    "GOL_ON_TRACK": label_gol_on_track,
    "GOL_BEHIND_PACE": label_gol_behind_pace,
    "GOL_MILESTONE_NEAR": label_gol_milestone_near,
    "GOL_COMPLETED": label_gol_completed,
    "CRS_SLEEP_HABIT_LINK": label_crs_sleep_habit_link,
    "CRS_EXERCISE_MOOD_LINK": label_crs_exercise_mood_link,
    "CRS_SPENDING_MOOD_LINK": label_crs_spending_mood_link,
    "CRS_HYDRATION_ENERGY": label_crs_hydration_energy,
    "CRS_WEEKEND_PATTERN": label_crs_weekend_pattern,
    "GEN_DATA_SUFFICIENT": label_gen_data_sufficient,
    "GEN_DATA_THIN": label_gen_data_thin,
    "GEN_NEW_USER": label_gen_new_user,
}

assert set(LABEL_FUNCTIONS.keys()) == set(INSIGHT_CODES), (
    "LABEL_FUNCTIONS keys must match INSIGHT_CODES exactly."
)


def label_all_categories(series: dict) -> np.ndarray:
    """Apply all 40 labellers and return a (num_days, 40) uint8 matrix."""
    n = len(series["mood_score"])
    out = np.zeros((n, NUM_INSIGHT_CLASSES), dtype=np.uint8)
    for code, fn in LABEL_FUNCTIONS.items():
        idx = INSIGHT_CODES.index(code)
        out[:, idx] = fn(series).astype(np.uint8)
    return out
