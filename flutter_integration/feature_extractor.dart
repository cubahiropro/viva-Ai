// lib/core/ai/feature_extractor.dart
//
// Dart port of `src/feature_engineering/feature_pipeline.py`. Reads a 30-day
// "daily array" produced by the in-app aggregator (one row per day, 37
// numeric columns matching `DAILY_COLUMNS` in the Python pipeline) and
// produces the 128-feature vector consumed by the TFLite model.
//
// Parity is verified by `parity_fixture.json` produced by
// `scripts/export_parity_fixture.py`. The Dart and Python outputs must agree
// to within 1e-4 across all 128 features.
//
// This file deliberately depends only on `dart:math` and `dart:convert` so it
// can be dropped into any Flutter project without extra packages.

import 'dart:convert';
import 'dart:math' as math;

const int kWindowDays = 30;
const int kFeatureDim = 128;

/// Order must match Python's `DAILY_COLUMNS` exactly.
const List<String> kDailyColumns = <String>[
  'daily_total_spend',
  'num_transactions',
  'income_today',
  'monthly_budget',
  'is_unusual_expense',
  'is_weekend',
  'category_spend_food',
  'category_spend_transport',
  'category_spend_bills',
  'category_spend_health',
  'category_spend_shopping',
  'category_spend_other',
  'water_cups',
  'water_logged',
  'weight_kg',
  'weight_logged',
  'sleep_duration_hours',
  'sleep_quality',
  'bedtime_hour_after_20',
  'wake_hour',
  'sleep_logged',
  'mood_score',
  'mood_logged',
  'tasks_created',
  'tasks_completed',
  'tasks_overdue',
  'task_completion_rate_today',
  'habit_completion_rate',
  'morning_habit_rate',
  'evening_habit_rate',
  'habit_streak',
  'habit_longest_ever',
  'num_habits',
  'doses_scheduled',
  'doses_taken',
  'timing_offset_minutes',
  'has_critical_medications',
];

const List<String> _categoryCols = <String>[
  'category_spend_food',
  'category_spend_transport',
  'category_spend_bills',
  'category_spend_health',
  'category_spend_shopping',
  'category_spend_other',
];

/// Result of feature extraction.
class VivaFeatureVector {
  VivaFeatureVector(this.features) : assert(features.length == kFeatureDim);
  final List<double> features;

  Map<String, dynamic> toJson() => <String, dynamic>{'features': features};
}

class FeatureExtractor {
  FeatureExtractor({
    required this.window,
    required this.todayDate,
    required this.daysSinceFirstUse,
    required this.monthlyIncome,
    required this.weightBaselineKg,
    required this.heightCm,
  })  : assert(window.length == kWindowDays),
        assert(window.first.length == kDailyColumns.length);

  /// The 30-day daily matrix, oldest day first, newest day last.
  /// Each row has [kDailyColumns.length] entries in the order of [kDailyColumns].
  final List<List<double>> window;

  /// The calendar date of the *last* row of [window] (today).
  final DateTime todayDate;

  final int daysSinceFirstUse;
  final double monthlyIncome;
  final double weightBaselineKg;
  final double heightCm;

  // ---------------------------------------------------------------------- helpers

  int _i(String name) => kDailyColumns.indexOf(name);

  List<double> _col(String name) {
    final int idx = _i(name);
    return <double>[for (final row in window) row[idx]];
  }

  static double _safeDiv(double a, double b, {double fallback = 0.0}) =>
      b.abs() < 1e-9 ? fallback : a / b;

  static double _clip(double v, double lo, double hi) {
    if (v.isNaN || !v.isFinite) return lo;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
  }

  static double _mean(List<double> xs) {
    if (xs.isEmpty) return 0.0;
    double s = 0;
    for (final v in xs) {
      s += v;
    }
    return s / xs.length;
  }

  static double _std(List<double> xs) {
    if (xs.length < 2) return 0.0;
    final double m = _mean(xs);
    double s = 0;
    for (final v in xs) {
      s += (v - m) * (v - m);
    }
    return math.sqrt(s / xs.length);
  }

  static double _slope(List<double> xs) {
    if (xs.length < 3) return 0.0;
    final double n = xs.length.toDouble();
    final double xMean = (n - 1) / 2.0;
    final double yMean = _mean(xs);
    double num = 0;
    double den = 0;
    for (int i = 0; i < xs.length; i++) {
      num += (i - xMean) * (xs[i] - yMean);
      den += (i - xMean) * (i - xMean);
    }
    return den < 1e-12 ? 0.0 : num / den;
  }

  static double _pearson(List<double> a, List<double> b) {
    if (a.length != b.length || a.length < 3) return 0.0;
    final double ma = _mean(a);
    final double mb = _mean(b);
    final double sa = _std(a);
    final double sb = _std(b);
    if (sa < 1e-9 || sb < 1e-9) return 0.0;
    double s = 0;
    for (int i = 0; i < a.length; i++) {
      s += (a[i] - ma) * (b[i] - mb);
    }
    return (s / a.length) / (sa * sb);
  }

  static List<double> _tail(List<double> xs, int k) =>
      xs.sublist(math.max(0, xs.length - k));

  // ---------------------------------------------------------------------- main

  VivaFeatureVector compute() {
    final List<double> spend = _col('daily_total_spend');
    final List<double> numTx = _col('num_transactions');
    final List<double> income = _col('income_today');
    final List<double> budget = _col('monthly_budget');
    final List<double> isWeekend = _col('is_weekend');
    final List<List<double>> cat = <List<double>>[
      for (final c in _categoryCols) _col(c)
    ];
    final List<double> water = _col('water_cups');
    final List<double> waterLog = _col('water_logged');
    final List<double> weight = _col('weight_kg');
    final List<double> weightLog = _col('weight_logged');
    final List<double> sleepH = _col('sleep_duration_hours');
    final List<double> sleepQ = _col('sleep_quality');
    final List<double> bedtime = _col('bedtime_hour_after_20');
    final List<double> wake = _col('wake_hour');
    final List<double> sleepLog = _col('sleep_logged');
    final List<double> mood = _col('mood_score');
    final List<double> moodLog = _col('mood_logged');
    final List<double> tCreated = _col('tasks_created');
    final List<double> tCompleted = _col('tasks_completed');
    final List<double> tOverdue = _col('tasks_overdue');
    final List<double> tRate = _col('task_completion_rate_today');
    final List<double> hRate = _col('habit_completion_rate');
    final List<double> hMorn = _col('morning_habit_rate');
    final List<double> hEve = _col('evening_habit_rate');
    final List<double> hStreak = _col('habit_streak');
    final List<double> hLongest = _col('habit_longest_ever');
    final List<double> numHabits = _col('num_habits');
    final List<double> dosesSched = _col('doses_scheduled');
    final List<double> dosesTaken = _col('doses_taken');
    final List<double> timOff = _col('timing_offset_minutes');
    final List<double> hasCrit = _col('has_critical_medications');

    final List<double> spend7 = _tail(spend, 7);
    final List<double> spend30 = spend; // window length is 30
    final List<double> water7 = _tail(water, 7);
    final List<double> water30 = water;
    final List<double> sleepH7 = _tail(sleepH, 7);
    final List<double> sleepH30 = sleepH;
    final List<double> sleepQ7 = _tail(sleepQ, 7);
    final List<double> sleepQ30 = sleepQ;
    final List<double> mood7 = _tail(mood, 7);
    final List<double> mood30 = mood;

    // monthly aggregates approximated from the 30-day window:
    final double dom = todayDate.day.toDouble();
    final int monthDays = DateTime(todayDate.year, todayDate.month + 1, 0).day;
    double spendMonthToDate = 0;
    double incomeMonthToDate = 0;
    double daysInCurMonth = 0;
    double spendLastMonth = 0;
    double incomeLastMonth = 0;
    double lastMonthDays = 0;
    for (int i = 0; i < window.length; i++) {
      final DateTime d = todayDate.subtract(Duration(days: window.length - 1 - i));
      if (d.year == todayDate.year && d.month == todayDate.month) {
        spendMonthToDate += spend[i];
        incomeMonthToDate += income[i];
        daysInCurMonth += 1;
      } else {
        spendLastMonth += spend[i];
        incomeLastMonth += income[i];
        lastMonthDays += 1;
      }
    }

    int daysSinceIncome = 31;
    for (int i = window.length - 1; i >= 0; i--) {
      if (income[i] > 0) {
        daysSinceIncome = window.length - 1 - i;
        break;
      }
    }

    final List<double> features = <double>[];

    // ---------------------------- FINANCE (24)
    features.add((monthDays - dom + 1) / 31.0);
    features.add(math.min(_safeDiv(spendMonthToDate, math.max(budget.last, 1e-3)), 3.0));
    features.add(_safeDiv(spendMonthToDate, math.max(daysInCurMonth, 1.0)));
    features.add(lastMonthDays > 0 ? _safeDiv(spendLastMonth, lastMonthDays) : 0.0);
    final double lastWAvg = _mean(spend.sublist(math.max(0, spend.length - 14), math.max(0, spend.length - 7)));
    final double spendVelocity = lastWAvg > 0 ? _mean(spend7) / lastWAvg : 1.0;
    features.add(math.min(spendVelocity, 5.0));
    final double largest7 = spend7.fold<double>(0, (m, x) => x > m ? x : m);
    features.add(_safeDiv(largest7, math.max(monthlyIncome / 30.0, 1e-3)));

    // category fractions over the window's mean
    final List<double> catMean30 = <double>[for (final c in cat) _mean(c)];
    final double totalCat30 = catMean30.fold<double>(0, (a, b) => a + b);
    for (int c = 0; c < 6; c++) {
      features.add(totalCat30 > 1e-6 ? catMean30[c] / totalCat30 : 0.0);
    }
    features.add(daysSinceIncome / 31.0);
    final double incomeReg =
        1.0 / (1.0 + _std(income) / math.max(monthlyIncome, 1.0));
    features.add(incomeReg);
    final double savThis = incomeMonthToDate > 0
        ? _clip((incomeMonthToDate - spendMonthToDate) / incomeMonthToDate, -1, 1)
        : 0.0;
    features.add(savThis);
    final double savLast = incomeLastMonth > 0
        ? _clip((incomeLastMonth - spendLastMonth) / incomeLastMonth, -1, 1)
        : 0.0;
    features.add(savLast);
    final double spendStd30 = _std(spend30);
    final double spendMean30 = _mean(spend30);
    final double z = spendStd30 > 1.0 ? (spend.last - spendMean30) / spendStd30 : 0.0;
    features.add(_clip(z / 3.0, -1, 1));
    final double budgetPerDay = budget.last / 30.0;
    features.add(_clip(_safeDiv(spendMean30, math.max(budgetPerDay, 1e-3)), 0, 3));
    features.add(_clip(_safeDiv(_mean(spend7), math.max(budgetPerDay, 1e-3)), 0, 3));
    // weekend spend ratio (over the 30-day window)
    double weTot = 0, wdTot = 0;
    int weCnt = 0, wdCnt = 0;
    for (int i = 0; i < window.length; i++) {
      if (isWeekend[i] > 0.5) {
        weTot += spend[i];
        weCnt++;
      } else {
        wdTot += spend[i];
        wdCnt++;
      }
    }
    final double weAvg = weCnt > 0 ? weTot / weCnt : 0.0;
    final double wdAvg = wdCnt > 0 ? wdTot / wdCnt : 0.0;
    features.add(_clip(_safeDiv(weAvg, math.max(wdAvg, 1e-3), fallback: 1.0), 0, 3));
    features.add(_clip(_mean(_tail(numTx, 7)) / 8.0, 0, 1));
    features.add(_clip(_mean(numTx) / 8.0, 0, 1));
    final double catTot30 = totalCat30;
    double catMaxShare30 = 0;
    if (catTot30 > 1e-6) {
      double maxCat = 0;
      for (final v in catMean30) {
        if (v > maxCat) maxCat = v;
      }
      catMaxShare30 = maxCat / catTot30;
    }
    features.add(catMaxShare30);
    final double income30Sum = income.fold<double>(0, (a, b) => a + b);
    final double spend30Sum = spend.fold<double>(0, (a, b) => a + b);
    features.add(_clip(spend30Sum > 1.0 ? income30Sum / spend30Sum : 1.0, 0, 5));

    // ---------------------------- HEALTH (16)
    features.add(_clip(_mean(water7) / 8.0, 0, 2));
    features.add(_clip(_mean(water30) / 8.0, 0, 2));
    features.add(_clip(_slope(water7) / 2.0, -1, 1));
    features.add(_mean(_tail(waterLog, 7)));
    double latestWeight = weightBaselineKg;
    for (int i = window.length - 1; i >= 0; i--) {
      if (!weight[i].isNaN) {
        latestWeight = weight[i];
        break;
      }
    }
    features.add(_clip(latestWeight / math.max(weightBaselineKg, 1e-3), 0.5, 1.5));
    final List<double> w14 = <double>[
      for (int i = math.max(0, window.length - 14); i < window.length; i++)
        if (!weight[i].isNaN) weight[i]
    ];
    final double wSlope = _slope(w14);
    final double wVol = _std(w14);
    features.add(_clip(wSlope / 0.5, -1, 1));
    features.add(_clip(wVol / 2.0, 0, 1));
    features.add(_mean(weightLog));
    final double bmiBaseline = weightBaselineKg /
        math.max((heightCm / 100.0) * (heightCm / 100.0), 1e-3);
    final double bmiCat = bmiBaseline < 18.5
        ? 0.0
        : bmiBaseline < 25.0
            ? 0.5
            : 1.0;
    features.add(bmiCat);
    final double waterMax7 =
        water7.fold<double>(0, (m, x) => x > m ? x : m);
    final double waterMin7 =
        water7.fold<double>(double.infinity, (m, x) => x < m ? x : m);
    features.add(_clip(waterMax7 / 12.0, 0, 1));
    features.add(_clip(waterMin7 / 12.0, 0, 1));
    final double cons = 1.0 / (1.0 + _std(water7) / 3.0);
    features.add(cons);
    final double weightChange30 =
        w14.length >= 2 ? _clip((w14.last - w14.first) / 5.0, -1, 1) : 0.0;
    features.add(weightChange30);
    final double weightLogMax7 =
        _tail(weightLog, 7).fold<double>(0, (m, x) => x > m ? x : m);
    features.add(weightLogMax7);
    final double avgWeight30 = w14.isNotEmpty ? _mean(w14) : weightBaselineKg;
    features.add(_clip(avgWeight30 / math.max(weightBaselineKg, 1e-3), 0.5, 1.5));
    features.add(_clip((latestWeight - weightBaselineKg) / 10.0, -1, 1));

    // ---------------------------- SLEEP (18)
    features.add(_clip(_mean(sleepH7) / 8.0, 0, 1.5));
    features.add(_clip(_mean(sleepH30) / 8.0, 0, 1.5));
    features.add(_clip(_slope(sleepH7) / 1.0, -1, 1));
    double debt7 = 0;
    for (final h in sleepH7) {
      if (h < 7) debt7 += 7 - h;
    }
    features.add(_clip(debt7 / 14.0, 0, 1));
    features.add(_clip((_mean(sleepQ7) - 1) / 4.0, 0, 1));
    features.add(_clip((_mean(sleepQ30) - 1) / 4.0, 0, 1));
    features.add(_clip(_slope(sleepQ7) / 1.0, -1, 1));
    features.add(_clip(1.0 - _std(_tail(bedtime, 7)) / 4.0, 0, 1));
    features.add(_clip(_mean(_tail(bedtime, 7)) / 8.0, 0, 1));
    final List<double> sleepWe = <double>[];
    final List<double> sleepWd = <double>[];
    for (int i = math.max(0, window.length - 7); i < window.length; i++) {
      if (isWeekend[i] > 0.5) {
        sleepWe.add(sleepH[i]);
      } else {
        sleepWd.add(sleepH[i]);
      }
    }
    final double weChange = (sleepWe.isNotEmpty && sleepWd.isNotEmpty)
        ? _mean(sleepWe) - _mean(sleepWd)
        : 0.0;
    features.add(_clip(weChange / 2.0, -1, 1));
    features.add(sleepQ7.where((q) => q >= 4).length / sleepQ7.length);
    features.add(_clip(_std(sleepH7) / 3.0, 0, 1));
    features.add(_clip(sleepH7.reduce(math.min) / 11.0, 0, 1));
    features.add(_clip(sleepH7.reduce(math.max) / 11.0, 0, 1));
    features.add(_clip(_mean(_tail(wake, 7)) / 24.0, 0, 1));
    features.add(sleepH7.where((h) => h < 6.0).length / sleepH7.length);
    features.add(sleepH7.where((h) => h > 9.0).length / sleepH7.length);
    features.add(_mean(_tail(sleepLog, 7)));

    // ---------------------------- MOOD (14)
    features.add(_clip((_mean(mood7) - 1) / 4.0, 0, 1));
    features.add(_clip((_mean(mood30) - 1) / 4.0, 0, 1));
    features.add(_clip(_slope(mood7) / 1.0, -1, 1));
    features.add(_clip(_std(mood7) / 2.0, 0, 1));
    features.add(mood7.where((m) => m <= 2).length / mood7.length);
    features.add(mood7.where((m) => m >= 4).length / mood7.length);
    final double moodYesterday = window.length >= 2 ? mood[window.length - 2] : mood.last;
    features.add(_clip((moodYesterday - 1) / 4.0, 0, 1));
    final double sd = window.length >= 3
        ? () {
            final double d3 = mood.last - mood[window.length - 3];
            if (d3 > 0) return 1.0;
            if (d3 < 0) return 0.0;
            return 0.5;
          }()
        : 0.5;
    features.add(sd);
    features.add(_clip((mood.last - 1) / 4.0, 0, 1));
    features.add(_clip((mood7.reduce(math.min) - 1) / 4.0, 0, 1));
    features.add(_clip((mood7.reduce(math.max) - 1) / 4.0, 0, 1));
    features.add(_mean(_tail(moodLog, 7)));
    final List<double> moodWe = <double>[];
    final List<double> moodWd = <double>[];
    for (int i = 0; i < window.length; i++) {
      if (isWeekend[i] > 0.5) {
        moodWe.add(mood[i]);
      } else {
        moodWd.add(mood[i]);
      }
    }
    final double weMood = moodWe.isNotEmpty ? _mean(moodWe) : 0.0;
    final double wdMood = moodWd.isNotEmpty ? _mean(moodWd) : 0.0;
    features.add(_clip((weMood - 1) / 4.0, 0, 1));
    features.add(_clip((wdMood - 1) / 4.0, 0, 1));

    // ---------------------------- TASKS & HABITS (20)
    features.add(_mean(_tail(tRate, 7)));
    features.add(_mean(tRate));
    features.add(_clip(tOverdue.last / 20.0, 0, 1));
    features.add(_clip(_mean(_tail(tCreated, 7)) / 8.0, 0, 1));
    features.add(_mean(_tail(hRate, 7)));
    features.add(_mean(hRate));
    features.add(_clip(hStreak.last / 90.0, 0, 1));
    final bool atRisk = window.length >= 2 &&
        hStreak[window.length - 2] >= 5 &&
        hRate.last < 1.0;
    features.add(atRisk ? 1.0 : 0.0);
    features.add(_clip(_slope(_tail(hRate, 7)) / 1.0, -1, 1));
    features.add(_clip(_mean(_tail(hMorn, 7)) - _mean(_tail(hEve, 7)), -1, 1) * 0.5 + 0.5);
    features.add(_mean(_tail(hMorn, 7)));
    features.add(_mean(_tail(hEve, 7)));
    features.add(_clip(_mean(hMorn) - _mean(hEve), -1, 1) * 0.5 + 0.5);
    features.add(_clip(_tail(tCompleted, 7).fold<double>(0, (a, b) => a + b) / 56.0, 0, 1));
    features.add(_clip(tCompleted.fold<double>(0, (a, b) => a + b) / 240.0, 0, 1));
    features.add(_clip(_slope(_tail(tOverdue, 7)) / 5.0, -1, 1));
    features.add(_clip(hStreak.last / math.max(hLongest.last, 1.0), 0, 1));
    features.add(_clip(_slope(_tail(hStreak, 7)) / 1.0, -1, 1));
    features.add(_clip(numHabits.last / 6.0, 0, 1));
    features.add(_tail(hRate, 7).where((r) => r >= 1.0).length / 7.0);

    // ---------------------------- MEDICATIONS (12)
    final List<double> compliance = <double>[
      for (int i = 0; i < window.length; i++)
        dosesSched[i] > 0 ? dosesTaken[i] / math.max(dosesSched[i], 1) : 1.0
    ];
    features.add(_mean(_tail(compliance, 7)));
    features.add(_mean(compliance));
    final double missed7 = <double>[
      for (int i = window.length - 7; i < window.length; i++)
        dosesSched[i] - dosesTaken[i]
    ].fold<double>(0, (a, b) => a + b);
    features.add(_clip(missed7 / 28.0, 0, 1));
    features.add(_clip(_mean(_tail(timOff, 7)) / 60.0, 0, 1));
    features.add(hasCrit.last);
    features.add(_clip(_slope(_tail(compliance, 7)), -1, 1) * 0.5 + 0.5);
    features.add(_clip(_tail(dosesTaken, 7).fold<double>(0, (a, b) => a + b) / 28.0, 0, 1));
    features.add(_clip(
      <double>[for (int i = 0; i < window.length; i++) dosesSched[i] - dosesTaken[i]]
              .fold<double>(0, (a, b) => a + b) /
          120.0,
      0,
      1,
    ));
    features.add(_clip(_mean(timOff) / 60.0, 0, 1));
    features.add(_clip(_std(_tail(compliance, 7)) / 0.5, 0, 1));
    features.add(_clip(dosesSched.last / 4.0, 0, 1));
    features.add(dosesSched.last > 0 ? 1.0 : 0.0);

    // ---------------------------- CROSS-MODULE (12)
    final List<double> sleepQ14 =
        sleepQ.sublist(math.max(0, sleepQ.length - 14));
    final List<double> mood14 = mood.sublist(math.max(0, mood.length - 14));
    final List<double> hRate14 = hRate.sublist(math.max(0, hRate.length - 14));
    final List<double> spend14 = spend.sublist(math.max(0, spend.length - 14));
    final List<double> water14 = water.sublist(math.max(0, water.length - 14));
    final List<double> tRate14 = tRate.sublist(math.max(0, tRate.length - 14));
    features.add(_pearson(sleepQ14, mood14) * 0.5 + 0.5);
    features.add(_pearson(sleepQ14, hRate14) * 0.5 + 0.5);
    features.add(_pearson(spend14, mood14) * 0.5 + 0.5);
    features.add(_pearson(water14, tRate14) * 0.5 + 0.5);
    // weekend perf delta
    final List<double> compositeWe = <double>[];
    final List<double> compositeWd = <double>[];
    for (int i = 0; i < window.length; i++) {
      final double v = (mood[i] + hRate[i] * 5 + tRate[i] * 5) / 3.0;
      if (isWeekend[i] > 0.5) {
        compositeWe.add(v);
      } else {
        compositeWd.add(v);
      }
    }
    final double perfDelta = (compositeWe.isNotEmpty && compositeWd.isNotEmpty)
        ? _mean(compositeWe) - _mean(compositeWd)
        : 0.0;
    features.add(_clip(perfDelta / 2.0, -1, 1) * 0.5 + 0.5);
    final double stress = _clip(
      ((5.0 - _mean(mood7)) / 5.0) * 0.4 +
          _clip(_safeDiv(_mean(spend7) - _mean(spend30), math.max(_mean(spend30), 1e-3)), 0, 2) * 0.3 +
          _clip((7.0 - _mean(sleepH7)) / 4.0, 0, 1) * 0.3,
      0,
      1,
    );
    features.add(stress);
    final double wellness = _clip(
      (_mean(mood7) / 5.0) * 0.4 +
          _clip(_mean(sleepH7) / 9.0, 0, 1) * 0.3 +
          _mean(hRate7List(hRate)) * 0.3,
      0,
      1,
    );
    features.add(wellness);
    final double consistency = _clip(
      1.0 - 0.5 * (_std(mood7) / 2.0) - 0.3 * (_std(sleepH7) / 3.0) - 0.2 * (_std(water7) / 4.0),
      0,
      1,
    );
    features.add(consistency);
    features.add(_pearson(mood14, spend14) * 0.5 + 0.5);
    features.add(_pearson(hRate14, mood14) * 0.5 + 0.5);
    final double weekendMoodDelta = (moodWe.isNotEmpty && moodWd.isNotEmpty)
        ? _mean(moodWe) - _mean(moodWd)
        : 0.0;
    features.add(_clip(weekendMoodDelta / 2.0, -1, 1) * 0.5 + 0.5);
    final double weekdayConsistency =
        moodWd.isNotEmpty ? 1.0 / (1.0 + _std(moodWd)) : 0.0;
    features.add(_clip(weekdayConsistency, 0, 1));

    // ---------------------------- TEMPORAL (12)
    final int dow = todayDate.weekday % 7; // Monday=1..Sunday=7 → 1..0; Python: Mon=0, Sun=6
    final int dowPy = todayDate.weekday - 1; // matches Python's .weekday() (Mon=0)
    features.add((math.sin(2 * math.pi * dowPy / 7.0) + 1.0) * 0.5);
    features.add((math.cos(2 * math.pi * dowPy / 7.0) + 1.0) * 0.5);
    features.add(todayDate.day / 31.0);
    features.add((math.sin(2 * math.pi * todayDate.month / 12.0) + 1.0) * 0.5);
    features.add((math.cos(2 * math.pi * todayDate.month / 12.0) + 1.0) * 0.5);
    features.add(dowPy >= 5 ? 1.0 : 0.0);
    features.add(todayDate.day <= 5 ? 1.0 : 0.0);
    features.add(todayDate.day > monthDays - 5 ? 1.0 : 0.0);
    features.add(_clip(daysSinceFirstUse / 365.0, 0, 1));
    final double completeness =
        (waterLog.last + weightLog.last + sleepLog.last + moodLog.last) / 4.0;
    features.add(completeness);
    features.add((todayDate.month % 12) ~/ 3 / 3.0);
    features.add(todayDate.month == 12 ? 1.0 : 0.0);

    assert(features.length == kFeatureDim, 'expected $kFeatureDim, got ${features.length}');
    return VivaFeatureVector(features);
  }
}

// Convenience for the wellness composite
List<double> hRate7List(List<double> hRate) =>
    hRate.sublist(math.max(0, hRate.length - 7));

/// Apply min-max normalisation using the saved [feature_metadata.json] params.
List<double> applyNormaliser(
  List<double> features,
  List<double> minimums,
  List<double> maximums,
) {
  assert(features.length == minimums.length);
  assert(features.length == maximums.length);
  final List<double> out = List<double>.filled(features.length, 0);
  for (int i = 0; i < features.length; i++) {
    final double span = math.max(maximums[i] - minimums[i], 1e-6);
    out[i] = ((features[i] - minimums[i]) / span).clamp(0.0, 1.0).toDouble();
  }
  return out;
}

/// Loads a parity fixture (from `parity_fixture.json`) — used by the Dart unit
/// test that verifies feature-vector parity with the Python pipeline.
class ParityFixture {
  ParityFixture._({
    required this.window,
    required this.todayDate,
    required this.daysSinceFirstUse,
    required this.monthlyIncome,
    required this.weightBaselineKg,
    required this.heightCm,
    required this.expectedFeatures,
  });

  final List<List<double>> window;
  final DateTime todayDate;
  final int daysSinceFirstUse;
  final double monthlyIncome;
  final double weightBaselineKg;
  final double heightCm;
  final List<double> expectedFeatures;

  static ParityFixture fromJsonString(String jsonStr) {
    final Map<String, dynamic> j = json.decode(jsonStr) as Map<String, dynamic>;
    final List<List<double>> window = (j['window'] as List<dynamic>)
        .map<List<double>>(
          (row) => (row as List<dynamic>).map((v) => (v as num).toDouble()).toList(),
        )
        .toList();
    return ParityFixture._(
      window: window,
      todayDate: DateTime.parse(j['today_iso'] as String),
      daysSinceFirstUse: j['days_since_first_use'] as int,
      monthlyIncome: (j['monthly_income'] as num).toDouble(),
      weightBaselineKg: (j['weight_baseline_kg'] as num).toDouble(),
      heightCm: (j['height_cm'] as num).toDouble(),
      expectedFeatures: (j['expected_features'] as List<dynamic>)
          .map<double>((v) => (v as num).toDouble())
          .toList(),
    );
  }
}
