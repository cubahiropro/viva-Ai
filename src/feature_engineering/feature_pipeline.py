"""128-feature pipeline implementing section 5.1 of the training prompt.

This module is the **single source of truth** for both Python training and Dart
on-device inference. The Dart port `flutter_integration/feature_extractor.dart`
must mirror this code exactly so that the same 30-day window produces an
identical 128-vector to within 1e-4.

Public API:
    FEATURE_NAMES                 ordered list of 128 names (length 128)
    compute_user_features(...)    vectorised over a full per-user daily array
    compute_features_for_window(...)
                                   single-day feature vector from a 30-day window
                                   (mirrors the Dart implementation)
    NormaliserParams              fitted min/max per feature
    fit_normaliser(features)
    apply_normaliser(features, params)

All ops are basic arithmetic, rolling means, simple slopes, Pearson correlation
on equal-length 1-D arrays, and a couple of fixed sin/cos cycles. No randomness.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW_DAYS = 30
INPUT_DIM = 128

# Indexes into the daily column array that come from `data_generation`.
_DAILY_COLUMNS_DEFAULT: list[str] = [
    "daily_total_spend",
    "num_transactions",
    "income_today",
    "monthly_budget",
    "is_unusual_expense",
    "is_weekend",
    "category_spend_food",
    "category_spend_transport",
    "category_spend_bills",
    "category_spend_health",
    "category_spend_shopping",
    "category_spend_other",
    "water_cups",
    "water_logged",
    "weight_kg",
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

CATEGORY_COLS = (
    "category_spend_food",
    "category_spend_transport",
    "category_spend_bills",
    "category_spend_health",
    "category_spend_shopping",
    "category_spend_other",
)


# ----------------------------------------------------------------------------- helpers

def _col(daily: np.ndarray, columns: list[str], name: str) -> np.ndarray:
    return daily[:, columns.index(name)]


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean. Index i is mean of arr[max(0, i-w+1) .. i+1]."""
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        a = max(0, i - window + 1)
        out[i] = float(arr[a : i + 1].mean())
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        a = max(0, i - window + 1)
        seg = arr[a : i + 1]
        out[i] = float(seg.std()) if len(seg) > 1 else 0.0
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        a = max(0, i - window + 1)
        out[i] = float(arr[a : i + 1].sum())
    return out


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
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
    n = len(a)
    out = np.zeros(n, dtype=np.float32)
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


def _safe(x: float, default: float = 0.0) -> float:
    if not math.isfinite(x):
        return default
    return x


# ----------------------------------------------------------------------------- FEATURE NAMES

FEATURE_NAMES: list[str] = [
    # ---------------- FINANCE (24)
    "fin_days_until_month_end",
    "fin_budget_utilisation_rate",
    "fin_daily_avg_spend_this_month",
    "fin_daily_avg_spend_last_month",
    "fin_spend_velocity",
    "fin_largest_expense_7d_norm",
    *(f"fin_category_share_{c.split('_')[-1]}" for c in CATEGORY_COLS),  # 6
    "fin_days_since_last_income",
    "fin_income_regularity",
    "fin_savings_rate_this_month",
    "fin_savings_rate_last_month",
    "fin_unusual_expense_zscore",
    "fin_budget_adherence_30d",
    "fin_budget_adherence_7d",
    "fin_weekend_spend_ratio_30d",
    "fin_num_tx_avg_7d",
    "fin_num_tx_avg_30d",
    "fin_max_category_share_30d",
    "fin_income_to_spend_ratio_30d",
    # ---------------- HEALTH (16)
    "hlt_avg_water_7d",
    "hlt_avg_water_30d",
    "hlt_water_trend_7d",
    "hlt_days_logged_water_7d",
    "hlt_latest_weight_norm",
    "hlt_weight_trend_14d",
    "hlt_weight_volatility_14d",
    "hlt_days_logged_weight_30d",
    "hlt_bmi_category",
    "hlt_max_water_7d",
    "hlt_min_water_7d",
    "hlt_water_consistency_7d",
    "hlt_weight_change_30d",
    "hlt_has_recent_weigh_in_7d",
    "hlt_avg_weight_norm_30d",
    "hlt_weight_baseline_offset",
    # ---------------- SLEEP (18)
    "slp_avg_hours_7d",
    "slp_avg_hours_30d",
    "slp_hours_trend_7d",
    "slp_debt_7d",
    "slp_avg_quality_7d",
    "slp_avg_quality_30d",
    "slp_quality_trend_7d",
    "slp_bedtime_consistency_7d",
    "slp_avg_bedtime_7d",
    "slp_weekend_change",
    "slp_days_good_sleep_7d",
    "slp_duration_volatility_7d",
    "slp_min_hours_7d",
    "slp_max_hours_7d",
    "slp_avg_wake_hour_7d",
    "slp_short_nights_7d",
    "slp_long_nights_7d",
    "slp_log_frequency_7d",
    # ---------------- MOOD (14)
    "mod_avg_7d",
    "mod_avg_30d",
    "mod_trend_7d",
    "mod_volatility_7d",
    "mod_days_low_7d",
    "mod_days_high_7d",
    "mod_yesterday",
    "mod_streak_direction",
    "mod_today",
    "mod_min_7d",
    "mod_max_7d",
    "mod_log_freq_7d",
    "mod_weekend_avg_30d",
    "mod_weekday_avg_30d",
    # ---------------- TASKS & HABITS (20)
    "tsk_completion_rate_7d",
    "tsk_completion_rate_30d",
    "tsk_overdue_count_norm",
    "tsk_created_7d_norm",
    "hbt_completion_rate_7d",
    "hbt_completion_rate_30d",
    "hbt_longest_current_streak_norm",
    "hbt_at_risk_today",
    "hbt_completion_trend_7d",
    "hbt_morning_vs_evening",
    "hbt_avg_morning_rate_7d",
    "hbt_avg_evening_rate_7d",
    "hbt_morning_evening_diff_30d",
    "tsk_completed_7d_norm",
    "tsk_completed_30d_norm",
    "tsk_overdue_trend_7d",
    "hbt_streak_normalised",
    "hbt_streak_growth_7d",
    "hbt_num_habits_norm",
    "hbt_perfect_days_7d",
    # ---------------- MEDICATIONS (12)
    "med_compliance_rate_7d",
    "med_compliance_rate_30d",
    "med_missed_doses_7d_norm",
    "med_timing_irregularity_7d",
    "med_has_critical",
    "med_compliance_trend_7d",
    "med_doses_taken_7d_norm",
    "med_missed_doses_30d_norm",
    "med_timing_irregularity_30d",
    "med_compliance_volatility_7d",
    "med_num_medications_norm",
    "med_has_any",
    # ---------------- CROSS-MODULE (12)
    "crs_sleep_mood_corr_14d",
    "crs_sleep_habit_corr_14d",
    "crs_spend_mood_corr_14d",
    "crs_hydration_task_corr_14d",
    "crs_weekend_perf_delta",
    "crs_stress_composite",
    "crs_wellness_composite",
    "crs_consistency_score",
    "crs_mood_finance_corr_14d",
    "crs_exercise_mood_corr_14d",
    "crs_weekend_mood_delta",
    "crs_weekday_consistency",
    # ---------------- TEMPORAL (12)
    "tmp_dow_sin",
    "tmp_dow_cos",
    "tmp_dom_norm",
    "tmp_month_sin",
    "tmp_month_cos",
    "tmp_is_weekend",
    "tmp_is_month_start",
    "tmp_is_month_end",
    "tmp_days_since_first_use",
    "tmp_data_completeness",
    "tmp_season_norm",
    "tmp_is_holiday_month",
]
assert len(FEATURE_NAMES) == INPUT_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)}, expected {INPUT_DIM}"
)
assert len(set(FEATURE_NAMES)) == INPUT_DIM, "FEATURE_NAMES must be unique."


# ----------------------------------------------------------------------------- main vectorised path


def compute_user_features(
    daily: np.ndarray,
    columns: list[str] | None = None,
    days_since_first_use: np.ndarray | None = None,
    monthly_income: float | None = None,
    weight_baseline_kg: float | None = None,
    height_cm: float | None = None,
    start_date: np.datetime64 | str | None = None,
) -> np.ndarray:
    """Compute the (num_days, 128) feature tensor for one user.

    `start_date` is the calendar date corresponding to row 0 of `daily`. If
    None, defaults to 2024-01-01 (the dataset generator's default).
    """
    cols = columns or _DAILY_COLUMNS_DEFAULT
    n = daily.shape[0]
    F = INPUT_DIM
    out = np.zeros((n, F), dtype=np.float32)

    # ---- Pull columns
    spend = _col(daily, cols, "daily_total_spend").astype(np.float32)
    num_tx = _col(daily, cols, "num_transactions").astype(np.float32)
    income = _col(daily, cols, "income_today").astype(np.float32)
    budget = _col(daily, cols, "monthly_budget").astype(np.float32)
    unusual = _col(daily, cols, "is_unusual_expense").astype(np.float32)
    is_weekend = _col(daily, cols, "is_weekend").astype(np.float32)
    cat_spend = np.stack(
        [_col(daily, cols, c).astype(np.float32) for c in CATEGORY_COLS], axis=1
    )
    water = _col(daily, cols, "water_cups").astype(np.float32)
    water_log = _col(daily, cols, "water_logged").astype(np.float32)
    weight = _col(daily, cols, "weight_kg").astype(np.float32)         # NaN allowed
    weight_log = _col(daily, cols, "weight_logged").astype(np.float32)
    sleep_h = _col(daily, cols, "sleep_duration_hours").astype(np.float32)
    sleep_q = _col(daily, cols, "sleep_quality").astype(np.float32)
    bedtime = _col(daily, cols, "bedtime_hour_after_20").astype(np.float32)
    wake = _col(daily, cols, "wake_hour").astype(np.float32)
    sleep_log = _col(daily, cols, "sleep_logged").astype(np.float32)
    mood = _col(daily, cols, "mood_score").astype(np.float32)
    mood_log = _col(daily, cols, "mood_logged").astype(np.float32)
    t_created = _col(daily, cols, "tasks_created").astype(np.float32)
    t_completed = _col(daily, cols, "tasks_completed").astype(np.float32)
    t_overdue = _col(daily, cols, "tasks_overdue").astype(np.float32)
    t_rate = _col(daily, cols, "task_completion_rate_today").astype(np.float32)
    h_rate = _col(daily, cols, "habit_completion_rate").astype(np.float32)
    h_morning = _col(daily, cols, "morning_habit_rate").astype(np.float32)
    h_evening = _col(daily, cols, "evening_habit_rate").astype(np.float32)
    h_streak = _col(daily, cols, "habit_streak").astype(np.float32)
    h_longest = _col(daily, cols, "habit_longest_ever").astype(np.float32)
    num_habits = _col(daily, cols, "num_habits").astype(np.float32)
    doses_sched = _col(daily, cols, "doses_scheduled").astype(np.float32)
    doses_taken = _col(daily, cols, "doses_taken").astype(np.float32)
    tim_off = _col(daily, cols, "timing_offset_minutes").astype(np.float32)
    has_crit = _col(daily, cols, "has_critical_medications").astype(np.float32)

    # ---- Pre-computed rolling stats (vectorised once per user) ---------------
    spend_avg_7 = _rolling_mean(spend, 7)
    spend_avg_30 = _rolling_mean(spend, 30)
    spend_std_30 = _rolling_std(spend, 30)
    spend_max_7 = np.array([float(spend[max(0, i - 6) : i + 1].max()) for i in range(n)])
    cat_avg_30 = np.zeros_like(cat_spend)
    for c in range(cat_spend.shape[1]):
        cat_avg_30[:, c] = _rolling_mean(cat_spend[:, c], 30)
    income_avg_30 = _rolling_mean(income, 30)

    water_avg_7 = _rolling_mean(water, 7)
    water_avg_30 = _rolling_mean(water, 30)
    water_slope_7 = _rolling_slope(water, 7)
    water_log_avg_7 = _rolling_mean(water_log, 7)
    water_log_avg_30 = _rolling_mean(weight_log, 30)
    water_max_7 = np.array(
        [float(water[max(0, i - 6) : i + 1].max()) for i in range(n)], dtype=np.float32
    )
    water_min_7 = np.array(
        [float(water[max(0, i - 6) : i + 1].min()) for i in range(n)], dtype=np.float32
    )
    water_std_7 = _rolling_std(water, 7)
    weight_log_max_7 = np.array(
        [float(weight_log[max(0, i - 6) : i + 1].max()) for i in range(n)], dtype=np.float32
    )

    sleep_h_avg_7 = _rolling_mean(sleep_h, 7)
    sleep_h_avg_30 = _rolling_mean(sleep_h, 30)
    sleep_h_slope_7 = _rolling_slope(sleep_h, 7)
    sleep_q_avg_7 = _rolling_mean(sleep_q, 7)
    sleep_q_avg_30 = _rolling_mean(sleep_q, 30)
    sleep_q_slope_7 = _rolling_slope(sleep_q, 7)
    bedtime_std_7 = _rolling_std(bedtime, 7)
    bedtime_avg_7 = _rolling_mean(bedtime, 7)
    sleep_h_std_7 = _rolling_std(sleep_h, 7)
    sleep_log_avg_7 = _rolling_mean(sleep_log, 7)
    wake_avg_7 = _rolling_mean(wake, 7)

    mood_avg_7 = _rolling_mean(mood, 7)
    mood_avg_30 = _rolling_mean(mood, 30)
    mood_slope_7 = _rolling_slope(mood, 7)
    mood_std_7 = _rolling_std(mood, 7)
    mood_log_avg_7 = _rolling_mean(mood_log, 7)

    t_rate_avg_7 = _rolling_mean(t_rate, 7)
    t_rate_avg_30 = _rolling_mean(t_rate, 30)
    t_created_avg_7 = _rolling_mean(t_created, 7)
    t_completed_sum_7 = _rolling_sum(t_completed, 7)
    t_completed_sum_30 = _rolling_sum(t_completed, 30)
    t_overdue_slope_7 = _rolling_slope(t_overdue, 7)
    h_rate_avg_7 = _rolling_mean(h_rate, 7)
    h_rate_avg_30 = _rolling_mean(h_rate, 30)
    h_rate_slope_7 = _rolling_slope(h_rate, 7)
    h_morning_avg_7 = _rolling_mean(h_morning, 7)
    h_evening_avg_7 = _rolling_mean(h_evening, 7)
    h_morning_avg_30 = _rolling_mean(h_morning, 30)
    h_evening_avg_30 = _rolling_mean(h_evening, 30)
    h_streak_slope_7 = _rolling_slope(h_streak, 7)

    has_med = (doses_sched > 0).astype(np.float32)
    compliance_per_day = np.where(doses_sched > 0, doses_taken / np.maximum(doses_sched, 1), 1.0)
    comp_avg_7 = _rolling_mean(compliance_per_day, 7)
    comp_avg_30 = _rolling_mean(compliance_per_day, 30)
    comp_slope_7 = _rolling_slope(compliance_per_day, 7)
    comp_std_7 = _rolling_std(compliance_per_day, 7)
    missed = doses_sched - doses_taken
    missed_sum_7 = _rolling_sum(missed, 7)
    missed_sum_30 = _rolling_sum(missed, 30)
    taken_sum_7 = _rolling_sum(doses_taken, 7)
    tim_avg_7 = _rolling_mean(tim_off, 7)
    tim_avg_30 = _rolling_mean(tim_off, 30)
    num_tx_avg_7 = _rolling_mean(num_tx, 7)
    num_tx_avg_30 = _rolling_mean(num_tx, 30)
    weight_log_avg_30 = _rolling_mean(weight_log, 30)

    # ---- Cross-module correlations (14-day window)
    sleep_mood_corr = _rolling_pearson(sleep_q, mood, 14)
    sleep_habit_corr = _rolling_pearson(sleep_q, h_rate, 14)
    spend_mood_corr = _rolling_pearson(spend, mood, 14)
    water_task_corr = _rolling_pearson(water, t_rate, 14)
    mood_finance_corr = _rolling_pearson(mood, spend, 14)
    exercise_mood_corr = _rolling_pearson(h_rate, mood, 14)

    # Weekend deltas — at each day d, average of weekend-vs-weekday over last 28 days.
    we_perf_delta = np.zeros(n, dtype=np.float32)
    we_mood_delta = np.zeros(n, dtype=np.float32)
    weekday_consistency = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - 27)
        we = is_weekend[a : i + 1] > 0.5
        if we.sum() < 2 or (~we).sum() < 2:
            continue
        composite = (mood[a : i + 1] + h_rate[a : i + 1] * 5 + t_rate[a : i + 1] * 5) / 3.0
        we_perf_delta[i] = float(composite[we].mean() - composite[~we].mean())
        we_mood_delta[i] = float(mood[a : i + 1][we].mean() - mood[a : i + 1][~we].mean())
        weekday_consistency[i] = 1.0 / (1.0 + float(mood[a : i + 1][~we].std()))

    # Stress / wellness composites
    stress = np.clip(
        (5.0 - mood_avg_7) / 5.0 * 0.4
        + np.clip((spend_avg_7 - spend_avg_30) / np.maximum(spend_avg_30, 1e-3), 0, 2) * 0.3
        + np.clip((7.0 - sleep_h_avg_7) / 4.0, 0, 1) * 0.3,
        0,
        1,
    )
    wellness = np.clip(
        (mood_avg_7 / 5.0) * 0.4
        + np.clip(sleep_h_avg_7 / 9.0, 0, 1) * 0.3
        + h_rate_avg_7 * 0.3,
        0,
        1,
    )

    # Consistency: 1 - normalised volatility across the main signals
    consistency = np.clip(
        1.0
        - 0.5 * (mood_std_7 / 2.0)
        - 0.3 * (sleep_h_std_7 / 3.0)
        - 0.2 * (water_std_7 / 4.0),
        0,
        1,
    )

    # ---- Per-day scalars / month windows
    days_dt = np.arange(n)                              # day-since-start (used for temporal sin/cos)
    if days_since_first_use is None:
        days_since_first_use = days_dt.astype(np.float32)

    if start_date is None:
        start = np.datetime64("2024-01-01", "D")
    else:
        start = np.datetime64(start_date, "D")
    dates = start + days_dt
    dom = np.array([int(np.datetime_as_string(d, unit="D")[-2:]) for d in dates])
    moy = np.array([int(np.datetime_as_string(d, unit="M")[-2:]) for d in dates])
    dow = np.array([int(d.astype("datetime64[D]").astype("O").weekday()) for d in dates])
    days_in_month_per_day = np.zeros(n, dtype=np.int32)
    for i in range(n):
        first_next = (np.datetime64(dates[i], "M") + 1).astype("datetime64[D]")
        first_this = np.datetime64(dates[i], "M").astype("datetime64[D]")
        days_in_month_per_day[i] = int((first_next - first_this).astype(int))

    # Per-month aggregates
    spend_month_to_date = np.zeros(n, dtype=np.float32)
    income_month_to_date = np.zeros(n, dtype=np.float32)
    spend_last_month = np.zeros(n, dtype=np.float32)
    income_last_month = np.zeros(n, dtype=np.float32)
    days_in_current_month = np.zeros(n, dtype=np.float32)
    last_month_days = np.zeros(n, dtype=np.float32)
    last_month_id = -1
    cur_spend = 0.0
    cur_income = 0.0
    cur_days = 0
    prev_total_spend = 0.0
    prev_total_income = 0.0
    prev_days = 0
    for i in range(n):
        m = int(np.datetime64(dates[i], "M").astype(int))
        if m != last_month_id:
            prev_total_spend = cur_spend
            prev_total_income = cur_income
            prev_days = cur_days
            cur_spend = 0.0
            cur_income = 0.0
            cur_days = 0
            last_month_id = m
        cur_spend += float(spend[i])
        cur_income += float(income[i])
        cur_days += 1
        spend_month_to_date[i] = cur_spend
        income_month_to_date[i] = cur_income
        days_in_current_month[i] = cur_days
        spend_last_month[i] = prev_total_spend
        income_last_month[i] = prev_total_income
        last_month_days[i] = prev_days

    days_since_income = np.zeros(n, dtype=np.float32)
    last_income = -999
    for i in range(n):
        if income[i] > 0:
            last_income = i
        days_since_income[i] = float(min(31, i - last_income)) if last_income >= 0 else 31.0

    income_in_30d = _rolling_sum(income, 30)
    spend_in_30d = _rolling_sum(spend, 30)

    # ---- Iterate days and assemble features --------------------------------
    income_target = monthly_income or 1500.0
    height_m = (height_cm or 170.0) / 100.0
    weight_baseline = weight_baseline_kg or 72.0
    bmi_baseline = weight_baseline / max(height_m * height_m, 1e-3)
    if bmi_baseline < 18.5:
        bmi_cat = 0.0
    elif bmi_baseline < 25.0:
        bmi_cat = 0.5
    else:
        bmi_cat = 1.0

    weekend_spend_ratio_30 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - 29)
        seg = spend[a : i + 1]
        we = is_weekend[a : i + 1] > 0.5
        if we.sum() == 0 or (~we).sum() == 0:
            weekend_spend_ratio_30[i] = 1.0
            continue
        wd_avg = seg[~we].mean()
        we_avg = seg[we].mean()
        weekend_spend_ratio_30[i] = float(we_avg / max(wd_avg, 1e-3))

    we_mood_avg_30 = np.zeros(n, dtype=np.float32)
    wd_mood_avg_30 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - 29)
        seg = mood[a : i + 1]
        we = is_weekend[a : i + 1] > 0.5
        we_mood_avg_30[i] = float(seg[we].mean()) if we.sum() else 0.0
        wd_mood_avg_30[i] = float(seg[~we].mean()) if (~we).sum() else 0.0

    cat_share_max_30 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - 29)
        seg = cat_spend[a : i + 1]
        total = seg.sum()
        if total <= 1e-6:
            cat_share_max_30[i] = 0.0
        else:
            cat_share_max_30[i] = float(seg.sum(axis=0).max() / total)

    safe_spend_30 = np.maximum(spend_in_30d, 1.0)
    income_to_spend_30 = np.where(spend_in_30d > 1.0, income_in_30d / safe_spend_30, 1.0)
    budget_per_day = budget / 30.0

    for i in range(n):
        f: list[float] = []

        # ---------------- FINANCE (24)
        days_left = (days_in_month_per_day[i] - dom[i] + 1) / 31.0
        f.append(days_left)
        budget_util = spend_month_to_date[i] / max(budget[i], 1e-3)
        f.append(min(budget_util, 3.0))
        f.append(spend_month_to_date[i] / max(days_in_current_month[i], 1.0))
        f.append(spend_last_month[i] / max(last_month_days[i], 1.0) if last_month_days[i] > 0 else 0.0)
        last_w = spend[max(0, i - 13) : max(0, i - 6)]
        last_w_avg = float(last_w.mean()) if len(last_w) > 0 else 0.0
        spend_velocity = spend_avg_7[i] / max(last_w_avg, 1e-3) if last_w_avg > 0 else 1.0
        f.append(min(spend_velocity, 5.0))
        f.append(spend_max_7[i] / max(income_target / 30.0, 1e-3))
        cat_total_30 = cat_avg_30[i].sum()
        if cat_total_30 > 1e-6:
            for c in range(cat_avg_30.shape[1]):
                f.append(float(cat_avg_30[i, c] / cat_total_30))
        else:
            for _ in range(6):
                f.append(0.0)
        f.append(days_since_income[i] / 31.0)
        income_reg = 1.0 / (1.0 + float(np.std(income_in_30d[max(0, i - 60): i + 1]) / max(income_target, 1.0)))
        f.append(income_reg)
        sav_this = (income_month_to_date[i] - spend_month_to_date[i]) / max(income_month_to_date[i], 1.0) if income_month_to_date[i] > 0 else 0.0
        f.append(float(np.clip(sav_this, -1, 1)))
        sav_last = (income_last_month[i] - spend_last_month[i]) / max(income_last_month[i], 1.0) if income_last_month[i] > 0 else 0.0
        f.append(float(np.clip(sav_last, -1, 1)))
        z = (spend[i] - spend_avg_30[i]) / max(spend_std_30[i], 1.0)
        f.append(float(np.clip(z / 3.0, -1, 1)))
        f.append(float(np.clip(spend_avg_30[i] / max(budget_per_day[i], 1e-3), 0, 3)))
        f.append(float(np.clip(spend_avg_7[i] / max(budget_per_day[i], 1e-3), 0, 3)))
        f.append(float(np.clip(weekend_spend_ratio_30[i], 0, 3)))
        f.append(float(np.clip(num_tx_avg_7[i] / 8.0, 0, 1)))
        f.append(float(np.clip(num_tx_avg_30[i] / 8.0, 0, 1)))
        f.append(cat_share_max_30[i])
        f.append(float(np.clip(income_to_spend_30[i], 0, 5)))

        # ---------------- HEALTH (16)
        f.append(float(np.clip(water_avg_7[i] / 8.0, 0, 2)))
        f.append(float(np.clip(water_avg_30[i] / 8.0, 0, 2)))
        f.append(float(np.clip(water_slope_7[i] / 2.0, -1, 1)))
        f.append(water_log_avg_7[i])
        if math.isnan(weight[i]):
            valid_idx = np.where(~np.isnan(weight[: i + 1]))[0]
            latest = float(weight[valid_idx[-1]]) if len(valid_idx) else weight_baseline
        else:
            latest = float(weight[i])
        f.append(float(np.clip(latest / max(weight_baseline, 1e-3), 0.5, 1.5)))

        a = max(0, i - 13)
        seg = weight[a : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 3:
            x = np.arange(len(valid), dtype=np.float64)
            y = valid.astype(np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            denom = float(((x - x_mean) ** 2).sum())
            slope = float(((x - x_mean) * (y - y_mean)).sum()) / max(denom, 1e-12)
            vol = float(valid.std())
        else:
            slope = 0.0
            vol = 0.0
        f.append(float(np.clip(slope / 0.5, -1, 1)))
        f.append(float(np.clip(vol / 2.0, 0, 1)))
        f.append(float(weight_log_avg_30[i]))
        f.append(bmi_cat)
        f.append(float(np.clip(water_max_7[i] / 12.0, 0, 1)))
        f.append(float(np.clip(water_min_7[i] / 12.0, 0, 1)))
        cons = 1.0 / (1.0 + float(water_std_7[i]) / 3.0)
        f.append(cons)
        a30 = max(0, i - 29)
        seg30 = weight[a30 : i + 1]
        valid30 = seg30[~np.isnan(seg30)]
        wch = float(valid30[-1] - valid30[0]) if len(valid30) >= 2 else 0.0
        f.append(float(np.clip(wch / 5.0, -1, 1)))
        f.append(float(weight_log_max_7[i]))
        avg_weight_30 = float(np.nanmean(seg30)) if len(valid30) else weight_baseline
        f.append(float(np.clip(avg_weight_30 / max(weight_baseline, 1e-3), 0.5, 1.5)))
        f.append(float(np.clip((latest - weight_baseline) / 10.0, -1, 1)))

        # ---------------- SLEEP (18)
        f.append(float(np.clip(sleep_h_avg_7[i] / 8.0, 0, 1.5)))
        f.append(float(np.clip(sleep_h_avg_30[i] / 8.0, 0, 1.5)))
        f.append(float(np.clip(sleep_h_slope_7[i] / 1.0, -1, 1)))
        debt_seg = np.maximum(0.0, 7.0 - sleep_h[max(0, i - 6) : i + 1]).sum()
        f.append(float(np.clip(debt_seg / 14.0, 0, 1)))
        f.append(float(np.clip((sleep_q_avg_7[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip((sleep_q_avg_30[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip(sleep_q_slope_7[i] / 1.0, -1, 1)))
        f.append(float(np.clip(1.0 - bedtime_std_7[i] / 4.0, 0, 1)))
        f.append(float(np.clip(bedtime_avg_7[i] / 8.0, 0, 1)))
        weekend_change = 0.0
        a7 = max(0, i - 6)
        we_seg = is_weekend[a7 : i + 1] > 0.5
        if we_seg.sum() > 0 and (~we_seg).sum() > 0:
            weekend_change = float(sleep_h[a7 : i + 1][we_seg].mean()
                                   - sleep_h[a7 : i + 1][~we_seg].mean())
        f.append(float(np.clip(weekend_change / 2.0, -1, 1)))
        good_sleep = float((sleep_q[a7 : i + 1] >= 4).sum()) / max(min(7, i + 1), 1)
        f.append(good_sleep)
        f.append(float(np.clip(sleep_h_std_7[i] / 3.0, 0, 1)))
        f.append(float(np.clip(sleep_h[a7 : i + 1].min() / 11.0, 0, 1)))
        f.append(float(np.clip(sleep_h[a7 : i + 1].max() / 11.0, 0, 1)))
        f.append(float(np.clip(wake_avg_7[i] / 24.0, 0, 1)))
        short_n = float((sleep_h[a7 : i + 1] < 6.0).sum()) / max(min(7, i + 1), 1)
        long_n = float((sleep_h[a7 : i + 1] > 9.0).sum()) / max(min(7, i + 1), 1)
        f.append(short_n)
        f.append(long_n)
        f.append(sleep_log_avg_7[i])

        # ---------------- MOOD (14)
        f.append(float(np.clip((mood_avg_7[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip((mood_avg_30[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip(mood_slope_7[i] / 1.0, -1, 1)))
        f.append(float(np.clip(mood_std_7[i] / 2.0, 0, 1)))
        f.append(float((mood[a7 : i + 1] <= 2).sum()) / max(min(7, i + 1), 1))
        f.append(float((mood[a7 : i + 1] >= 4).sum()) / max(min(7, i + 1), 1))
        yesterday = mood[i - 1] if i > 0 else mood[i]
        f.append(float(np.clip((yesterday - 1) / 4.0, 0, 1)))
        # Streak direction: +1 / 0 / -1 → 0 / 0.5 / 1
        if i >= 2:
            d3 = mood[i] - mood[max(0, i - 2)]
            sd = 1.0 if d3 > 0 else (0.0 if d3 < 0 else 0.5)
        else:
            sd = 0.5
        f.append(sd)
        f.append(float(np.clip((mood[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip((mood[a7 : i + 1].min() - 1) / 4.0, 0, 1)))
        f.append(float(np.clip((mood[a7 : i + 1].max() - 1) / 4.0, 0, 1)))
        f.append(mood_log_avg_7[i])
        f.append(float(np.clip((we_mood_avg_30[i] - 1) / 4.0, 0, 1)))
        f.append(float(np.clip((wd_mood_avg_30[i] - 1) / 4.0, 0, 1)))

        # ---------------- TASKS & HABITS (20)
        f.append(t_rate_avg_7[i])
        f.append(t_rate_avg_30[i])
        f.append(float(np.clip(t_overdue[i] / 20.0, 0, 1)))
        f.append(float(np.clip(t_created_avg_7[i] / 8.0, 0, 1)))
        f.append(h_rate_avg_7[i])
        f.append(h_rate_avg_30[i])
        f.append(float(np.clip(h_streak[i] / 90.0, 0, 1)))
        # at-risk today: streak >= 5 yesterday and not all habits done today
        if i > 0 and h_streak[i - 1] >= 5 and h_rate[i] < 1.0:
            risk = 1.0
        else:
            risk = 0.0
        f.append(risk)
        f.append(float(np.clip(h_rate_slope_7[i] / 1.0, -1, 1)))
        f.append(float(np.clip(h_morning_avg_7[i] - h_evening_avg_7[i], -1, 1) * 0.5 + 0.5))
        f.append(h_morning_avg_7[i])
        f.append(h_evening_avg_7[i])
        f.append(float(np.clip(h_morning_avg_30[i] - h_evening_avg_30[i], -1, 1) * 0.5 + 0.5))
        f.append(float(np.clip(t_completed_sum_7[i] / 56.0, 0, 1)))
        f.append(float(np.clip(t_completed_sum_30[i] / 240.0, 0, 1)))
        f.append(float(np.clip(t_overdue_slope_7[i] / 5.0, -1, 1)))
        f.append(float(np.clip(h_streak[i] / max(h_longest[i], 1.0), 0, 1)))
        f.append(float(np.clip(h_streak_slope_7[i] / 1.0, -1, 1)))
        f.append(float(np.clip(num_habits[i] / 6.0, 0, 1)))
        perfect = float((h_rate[a7 : i + 1] >= 1.0).sum()) / max(min(7, i + 1), 1)
        f.append(perfect)

        # ---------------- MEDICATIONS (12)
        f.append(comp_avg_7[i])
        f.append(comp_avg_30[i])
        f.append(float(np.clip(missed_sum_7[i] / 28.0, 0, 1)))
        f.append(float(np.clip(tim_avg_7[i] / 60.0, 0, 1)))
        f.append(float(has_crit[i]))
        f.append(float(np.clip(comp_slope_7[i], -1, 1) * 0.5 + 0.5))
        f.append(float(np.clip(taken_sum_7[i] / 28.0, 0, 1)))
        f.append(float(np.clip(missed_sum_30[i] / 120.0, 0, 1)))
        f.append(float(np.clip(tim_avg_30[i] / 60.0, 0, 1)))
        f.append(float(np.clip(comp_std_7[i] / 0.5, 0, 1)))
        f.append(float(np.clip(doses_sched[i] / 4.0, 0, 1)))
        f.append(float(has_med[i]))

        # ---------------- CROSS-MODULE (12)
        f.append(float(sleep_mood_corr[i]) * 0.5 + 0.5)
        f.append(float(sleep_habit_corr[i]) * 0.5 + 0.5)
        f.append(float(spend_mood_corr[i]) * 0.5 + 0.5)
        f.append(float(water_task_corr[i]) * 0.5 + 0.5)
        f.append(float(np.clip(we_perf_delta[i] / 2.0, -1, 1)) * 0.5 + 0.5)
        f.append(float(stress[i]))
        f.append(float(wellness[i]))
        f.append(float(consistency[i]))
        f.append(float(mood_finance_corr[i]) * 0.5 + 0.5)
        f.append(float(exercise_mood_corr[i]) * 0.5 + 0.5)
        f.append(float(np.clip(we_mood_delta[i] / 2.0, -1, 1)) * 0.5 + 0.5)
        f.append(float(np.clip(weekday_consistency[i], 0, 1)))

        # ---------------- TEMPORAL (12)
        f.append((math.sin(2 * math.pi * dow[i] / 7.0) + 1.0) * 0.5)
        f.append((math.cos(2 * math.pi * dow[i] / 7.0) + 1.0) * 0.5)
        f.append(float(dom[i]) / 31.0)
        f.append((math.sin(2 * math.pi * moy[i] / 12.0) + 1.0) * 0.5)
        f.append((math.cos(2 * math.pi * moy[i] / 12.0) + 1.0) * 0.5)
        f.append(float(is_weekend[i]))
        f.append(1.0 if dom[i] <= 5 else 0.0)
        f.append(1.0 if dom[i] > days_in_month_per_day[i] - 5 else 0.0)
        f.append(float(np.clip(days_since_first_use[i] / 365.0, 0, 1)))
        completeness = (
            water_log[i] + weight_log[i] + sleep_log[i] + mood_log[i]
        ) / 4.0
        f.append(float(completeness))
        # Season: 0=winter, 1=spring, 2=summer, 3=autumn → /3
        season_raw = ((moy[i] % 12) // 3)
        f.append(float(season_raw) / 3.0)
        f.append(1.0 if moy[i] == 12 else 0.0)

        out[i] = np.array(f, dtype=np.float32)

    return out


# ----------------------------------------------------------------------------- single-day window API


def compute_features_for_window(
    window_daily: np.ndarray,
    columns: list[str],
    today_date: np.datetime64 | str,
    days_since_first_use: int,
    monthly_income: float,
    weight_baseline_kg: float,
    height_cm: float,
) -> np.ndarray:
    """Compute the 128-feature vector for the *last* day of `window_daily`.

    `window_daily` should have shape (W, num_columns) where W >= 30.
    The function reuses `compute_user_features` and returns the row at
    `len(window_daily) - 1`. Used for parity testing with the Dart port.
    """
    n_window = window_daily.shape[0]
    today = np.datetime64(today_date, "D")
    start_date = today - (n_window - 1)
    days_since_first_use_arr = np.arange(
        days_since_first_use - n_window + 1,
        days_since_first_use + 1,
        dtype=np.float32,
    )
    feats = compute_user_features(
        window_daily,
        columns=columns,
        days_since_first_use=days_since_first_use_arr,
        monthly_income=monthly_income,
        weight_baseline_kg=weight_baseline_kg,
        height_cm=height_cm,
        start_date=start_date,
    )
    return feats[-1]


# ----------------------------------------------------------------------------- normalisation


@dataclass
class NormaliserParams:
    feature_names: list[str]
    minimums: np.ndarray
    maximums: np.ndarray

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "minimums": [float(x) for x in self.minimums.tolist()],
            "maximums": [float(x) for x in self.maximums.tolist()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormaliserParams":
        return cls(
            feature_names=list(d["feature_names"]),
            minimums=np.array(d["minimums"], dtype=np.float32),
            maximums=np.array(d["maximums"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NormaliserParams":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_normaliser(features: np.ndarray) -> NormaliserParams:
    """Fit a min-max normaliser. Inputs of shape (N, F) — last axis is features."""
    flat = features.reshape(-1, features.shape[-1])
    mins = np.percentile(flat, 1, axis=0).astype(np.float32)
    maxs = np.percentile(flat, 99, axis=0).astype(np.float32)
    spread = np.maximum(maxs - mins, 1e-6)
    maxs = mins + spread
    return NormaliserParams(
        feature_names=list(FEATURE_NAMES),
        minimums=mins,
        maximums=maxs,
    )


def apply_normaliser(features: np.ndarray, params: NormaliserParams) -> np.ndarray:
    span = params.maximums - params.minimums
    span = np.maximum(span, 1e-6)
    out = (features - params.minimums) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)
