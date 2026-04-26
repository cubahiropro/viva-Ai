# Viva Insight Net — Model Card
Generated: 2026-04-26T00:22:39.719561+00:00

## 1. Model details
- Architecture: 3-layer MLP, multi-output head (insights / budget / mood)
- Input: 128-d feature vector (see `feature_engineering/feature_pipeline.py`)
- Outputs: 40 insight classes, 1 budget logit, 1 mood scalar

## 2. Intended use
This model powers the Viva on-device assistant on Android. It produces user-personalised insights ranked by predicted probability. **It is not a medical or financial advice tool.**

## 3. Training data
Trained entirely on synthetic data generated from rule-based archetypes. No real user data was used. See `data/synthetic/` for generators.

## 4. Performance
### Aggregate metrics
| Metric | Value |
|---|---|
| budget_risk_auc | 0.9991 |
| budget_risk_loss | 0.0504 |
| insights_auc | 0.7504 |
| insights_loss | 0.2415 |
| insights_precision | 0.7499 |
| insights_recall | 0.5334 |
| loss | 0.2715 |
| mood_prediction_loss | 0.0005 |
| mood_prediction_mae | 0.0157 |
| macro_precision | 0.3374 |
| macro_recall | 0.2374 |
| macro_f1 | 0.2456 |
| macro_roc_auc | 0.7793 |
| macro_ap | 0.3578 |
| micro_precision | 0.7499 |
| micro_recall | 0.5334 |
| micro_f1 | 0.6234 |
| micro_roc_auc | 0.9211 |
| micro_f1_optimal | 0.2456 |

### Per-class metrics
| class | support | precision | recall | f1 | auc | ap |
|---|---|---|---|---|---|---|
| FIN_OVERSPEND_RISK | 6789 | 0.997 | 0.888 | 0.939 | 0.999 | 0.997 |
| FIN_UNUSUAL_EXPENSE | 1143 | 0.300 | 0.638 | 0.408 | 0.932 | 0.388 |
| FIN_SAVINGS_POSITIVE | 26325 | 0.761 | 0.701 | 0.730 | 0.706 | 0.800 |
| FIN_CATEGORY_SPIKE | 24149 | 0.804 | 0.650 | 0.719 | 0.782 | 0.848 |
| FIN_INCOME_IRREGULAR | 2127 | 0.216 | 0.313 | 0.255 | 0.833 | 0.191 |
| FIN_BUDGET_ON_TRACK | 24789 | 0.938 | 0.987 | 0.962 | 0.995 | 0.997 |
| HLT_WATER_LOW | 2144 | 0.000 | 0.000 | 0.000 | 0.834 | 0.170 |
| HLT_WEIGHT_TREND_UP | 2195 | 0.000 | 0.000 | 0.000 | 0.685 | 0.099 |
| HLT_WEIGHT_TREND_DOWN | 3378 | 0.000 | 0.000 | 0.000 | 0.666 | 0.150 |
| HLT_ACTIVITY_DROP | 809 | 0.000 | 0.000 | 0.000 | 0.762 | 0.048 |
| SLP_DEBT_ACCUMULATING | 8012 | 0.700 | 0.659 | 0.679 | 0.929 | 0.766 |
| SLP_QUALITY_LOW | 1020 | 0.333 | 0.001 | 0.002 | 0.902 | 0.180 |
| SLP_CONSISTENCY_GOOD | 14952 | 0.678 | 0.769 | 0.721 | 0.856 | 0.695 |
| SLP_LATE_NIGHT_PATTERN | 7767 | 0.000 | 0.000 | 0.000 | 0.590 | 0.246 |
| SLP_MOOD_CORRELATION | 16737 | 0.595 | 0.426 | 0.496 | 0.682 | 0.586 |
| MOD_LOW_STREAK | 304 | 0.000 | 0.000 | 0.000 | 0.984 | 0.201 |
| MOD_POSITIVE_STREAK | 8311 | 0.799 | 0.893 | 0.844 | 0.979 | 0.919 |
| MOD_SLEEP_LINK | 1839 | 0.000 | 0.000 | 0.000 | 0.712 | 0.120 |
| MOD_FINANCE_STRESS | 1977 | 0.000 | 0.000 | 0.000 | 0.660 | 0.089 |
| MOD_WEEKEND_PATTERN | 987 | 0.000 | 0.000 | 0.000 | 0.622 | 0.036 |
| TSK_OVERDUE_PILE | 12390 | 0.630 | 0.098 | 0.170 | 0.692 | 0.486 |
| TSK_COMPLETION_HIGH | 4428 | 1.000 | 0.000 | 0.000 | 0.780 | 0.294 |
| HBT_STREAK_AT_RISK | 291 | 0.000 | 0.000 | 0.000 | 0.778 | 0.018 |
| HBT_BEST_STREAK_EVER | 233 | 0.000 | 0.000 | 0.000 | 0.857 | 0.026 |
| HBT_MORNING_PATTERN | 19548 | 0.539 | 0.570 | 0.554 | 0.582 | 0.564 |
| MED_MISSED_DOSES | 7747 | 0.883 | 0.188 | 0.310 | 0.839 | 0.635 |
| MED_COMPLIANCE_GOOD | 8815 | 0.655 | 0.325 | 0.434 | 0.859 | 0.589 |
| MED_TIMING_IRREGULAR | 3793 | 0.697 | 0.216 | 0.329 | 0.927 | 0.557 |
| GOL_ON_TRACK | 6927 | 0.561 | 0.267 | 0.362 | 0.864 | 0.497 |
| GOL_BEHIND_PACE | 3218 | 0.525 | 0.010 | 0.019 | 0.889 | 0.350 |
| GOL_MILESTONE_NEAR | 166 | 0.000 | 0.000 | 0.000 | 0.815 | 0.015 |
| GOL_COMPLETED | 24 | 0.000 | 0.000 | 0.000 | 0.824 | 0.002 |
| CRS_SLEEP_HABIT_LINK | 3086 | 0.000 | 0.000 | 0.000 | 0.502 | 0.078 |
| CRS_EXERCISE_MOOD_LINK | 2087 | 0.000 | 0.000 | 0.000 | 0.526 | 0.058 |
| CRS_SPENDING_MOOD_LINK | 1977 | 0.000 | 0.000 | 0.000 | 0.655 | 0.086 |
| CRS_HYDRATION_ENERGY | 3123 | 0.000 | 0.000 | 0.000 | 0.561 | 0.096 |
| CRS_WEEKEND_PATTERN | 2290 | 0.000 | 0.000 | 0.000 | 0.578 | 0.071 |
| GEN_DATA_SUFFICIENT | 25729 | 0.883 | 0.897 | 0.890 | 0.922 | 0.957 |
| GEN_DATA_THIN | 574 | 0.000 | 0.000 | 0.000 | 0.831 | 0.051 |
| GEN_NEW_USER | 0 | 0.000 | 0.000 | 0.000 | nan | nan |

## 5. Limitations
- Trained only on synthetic data; real-world generalisation is unverified.
- No explicit fairness audit across demographic groups beyond archetype balance.
- The mood head is calibrated to a [0,1] proxy of the synthetic 1–5 scale.
- Edge cases (new users, partial data) are handled by feature defaults; predictions in those regimes should be treated cautiously.

## 6. Privacy
All inference is on-device. No telemetry leaves the user's phone. Inputs are derived locally from the user's tracked activity inside Viva.
