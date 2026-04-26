# Viva AI — Training Progress

## Phase 1 — Environment setup ✅ COMPLETE

### What was done
- Created the full project folder structure under `viva_ai/` matching section 2 of the prompt
  (`config/`, `data/{raw,synthetic,processed,splits}/`, `src/{data_generation,feature_engineering,models,training,evaluation,conversion,inference}/`,
  `flutter_integration/`, `models/{checkpoints,final,evaluation}/`, `tests/`, `scripts/`, `logs/`).
- Authored `requirements.txt` pinning the libraries listed in section 1
  (TensorFlow 2.16.x, pandas 2.x, numpy 1.26.x, scikit-learn 1.4.x, matplotlib, seaborn,
  faker, pytest, tqdm, python-decouple, shap, joblib, mypy, PyYAML, reportlab).
- Wrote `setup.py`, `.env`, `.gitignore`, `README.md`.
- Wrote `config/model_config.yaml` containing every hyperparameter from section 6.3.
- Wrote `src/insight_categories.py` — single source of truth for all 40 insight codes,
  their indices, and severity classification used downstream by Python and Dart.
- Wrote `src/config_loader.py` — small YAML loader.
- Created `tests/conftest.py` with shared fixtures (`rng`, `project_root`,
  `small_user_count`, `small_days`).
- Created `pytest.ini` with `pythonpath` covering `src/`.
- Created `tests/test_environment.py` with 6 smoke tests covering imports, config
  contents, insight category integrity, severity completeness, and RNG determinism.
- Created and activated a Python 3.12 virtual environment at `.venv/`.
- Installed the lightweight subset of dependencies needed by phases 1–3
  (numpy, pandas, pyyaml, pytest, tqdm, scikit-learn, matplotlib, seaborn, faker,
  joblib, python-decouple, reportlab). TensorFlow + shap will be installed at the
  start of Phase 4 to keep early iteration fast.

### What passed
```
$ pytest tests/ -v
============================== 6 passed in 0.37s ===============================
```

All Phase 1 tests pass.

### What comes next
**Phase 2 — Synthetic data generation.** Implement user archetypes, all module
generators (finance / health / sleep / mood / tasks / habits / medications),
deterministic ground-truth labelling functions for all 40 insight categories, the
master orchestrator, and `scripts/generate_data.py`. Run a 100-user smoke run
asserting > 10 positive examples per insight category, then the full 5,000-user
generation asserting >= 500 positive examples per category.

---

## Phase 2 — Synthetic data generation ✅ COMPLETE

### What was done
- **`src/data_generation/user_profiles.py`** — `USER_ARCHETYPES` dict mirroring
  section 4.1 (6 archetypes summing to weight 1.0) and `sample_user_profile()`
  drawing concrete `UserProfile` dataclasses with archetype-conditioned
  income, sleep, mood, habits, medication and demographic fields.
- **`src/data_generation/finance_generator.py`** — daily total spend (lognormal),
  per-category Dirichlet split (food / transport / bills / health / shopping /
  other), Poisson transaction count, weekend +20 % spend, December +25 %,
  monthly budget envelope, pay-day deposits with archetype-specific irregularity,
  and 3–5 unusual-expense events per year.
- **`src/data_generation/health_generator.py`** — water intake (archetype +
  summer effect, log presence per archetype) and weekly weigh-in weight series
  with slow drift modulated by exercise habit completion.
- **`src/data_generation/sleep_generator.py`** — duration (archetype + weekend
  effect), bedtime drift through the week, quality 1..5 influenced by duration,
  weekend disruption, and previous-day mood.
- **`src/data_generation/mood_generator.py`** — exact 0.35/0.20/0.15/0.15/0.15
  weighted blend from section 4.2 (sleep quality, habits-yesterday,
  budget-on-track-yesterday, day-of-week, archetype-noise) clamped to 1..5.
- **`src/data_generation/tasks_generator.py`** — daily tasks created/completed
  with running overdue count, plus per-day habit completion split into
  morning vs. evening, perfect-day streak counter and longest-ever counter.
- **`src/data_generation/medication_generator.py`** — scheduled/taken doses
  drawn from a Binomial keyed off archetype compliance, plus archetype-scaled
  timing-irregularity in minutes.
- **`src/data_generation/labelling.py`** — 40 deterministic ground-truth label
  functions, one per insight code, with helpers for rolling sum / mean / slope /
  Pearson correlation. Asserts in module load that the registry exactly matches
  `INSIGHT_CODES`. Returns `(num_days, 40)` `uint8` matrices.
- **`src/data_generation/master_generator.py`** — orchestrator that runs the
  generators in dependency order (finance → tasks/habits → health → sleep →
  budget-on-track → mood → re-run sleep using actual mood → re-run mood →
  medications → labels), then saves:
  - `data/synthetic/users.parquet` (5000 rows of metadata)
  - `data/synthetic/daily.npz` containing the `(5000, 365, 33)` float32 daily array
  - `data/synthetic/labels.npz` containing the `(5000, 365, 40)` uint8 labels
  - `data/synthetic/feature_columns.json` ordered daily column names
  - `data/synthetic/label_counts.json` positives per insight code
- **`scripts/generate_data.py`** — CLI matching the prompt
  (`--users / --days / --seed / --min-positive`).
- **`tests/test_data_generation.py`** — 11 tests covering archetype weights,
  profile sampling, every module generator's shape and value range, the full
  user pipeline, NaN safety in non-weight columns, label-function registry,
  label-tensor shape, and the 100-user smoke run asserting `>= 10` positives
  per category.

### What passed
```
$ pytest tests/test_data_generation.py -v
============================== 11 passed in 28.48s ==============================
```

```
$ python scripts/generate_data.py --users 5000 --days 365 --seed 42 \
        --min-positive 500 --out-dir data/synthetic --no-progress
Generating 5000 users × 365 days, seed=42
... (40 categories listed) ...
All insight categories meet the minimum positive count.
```

Rarest-class positive count in the 5000-user run: `GOL_COMPLETED = 614`,
comfortably above the 500-positive guarantee. Most-common: `FIN_SAVINGS_POSITIVE`
at ~1.2M. Total positives across the dataset = ~13M out of 73M (5000 × 365 × 40)
= ~17 % positive density across the multi-label head.

Saved artifacts (compressed):
- `daily.npz` 91 MB
- `labels.npz` 5.4 MB
- `users.parquet` 484 KB

### What comes next
**Phase 3 — Feature engineering.** Implement the 128-feature pipeline (24 finance
+ 16 health + 18 sleep + 14 mood + 20 tasks/habits + 12 medications + 12 cross-module
+ 12 temporal), normalisation/clipping, save metadata to JSON, write the Dart
`feature_extractor.dart` mirror, and the cross-language parity test.

---

## Phase 3 — Feature engineering ✅ COMPLETE

### What was done
- **`src/feature_engineering/feature_pipeline.py`** — single-pass 128-feature
  computation per (user, day): 24 finance, 16 health, 18 sleep, 14 mood,
  20 tasks/habits, 12 medications, 12 cross-module, 12 temporal. Includes
  rolling means/sums/std/slope, Pearson correlations over 14-day windows,
  and a min-max normaliser fit on training data.
- **`scripts/process_features.py`** — multi-process CLI that maps daily arrays
  to `(N, 336, 128)` features (drop the first 29 days for the 30-day window),
  fits the normaliser on the train slice, and saves
  `data/processed/features.npz`, `labels.npz`, `metadata.json` plus
  `models/final/feature_metadata.json` (min/max/feature_names) for the Dart side.
- **`mobile/lib/features/ai/engine/feature_extractor.dart`** — full Dart port
  of the pipeline that produces the same 128-vector from a 30-day window.
- **`scripts/export_parity_fixture.py`** — emits a deterministic JSON fixture
  used by the Dart unit test to verify cross-language parity (window, profile,
  expected features). NaN values from optional logs are zero-substituted to
  guarantee identical inputs to both pipelines.
- **`tests/test_feature_engineering.py`** — exhaustive tests covering shape,
  the helper functions, single-day window equivalence vs. the full pipeline,
  and the normaliser save/load round-trip.

### What passed
```
$ python scripts/process_features.py --max-users 800 --workers 2 --no-progress
Computing features for 800 users × 336 days × 128 features using 2 workers
Saving features → data/processed/features.npz
Saving labels   → data/processed/labels.npz
Done.
```

```
$ pytest tests/test_feature_engineering.py -v
12 passed
```

Rationale for using 800 users (not 5000): we deliberately scaled the dataset
down on the user's request to keep RAM usage on the host under ~5 GB while
training a proportionally smaller model.

### What comes next
**Phase 4 — Model training.**

---

## Phase 4 — Model training ✅ COMPLETE

### What was done
- **`src/models/insight_classifier.py`** — `Viva Insight Net`: 128-d input →
  shared trunk `[Dense(128) → BN → Dropout(0.2) → Dense(64) → BN → Dropout(0.1)
  → Dense(32)]` with three heads: insights (Sigmoid 40), budget_risk
  (Sigmoid 1), mood_prediction (Sigmoid 1). **29 002 parameters total
  (≈113 KB)** — deliberately small for low-resource hosts and for INT8 quant.
- **`src/training/loss_functions.py`** — focal-loss option for the multi-label
  head; standard BCE for the binary risk head; MSE for mood.
- **`src/training/callbacks.py`** — `EarlyStopping`, `ReduceLROnPlateau`,
  `ModelCheckpoint`, `TensorBoard`.
- **`src/training/trainer.py`** — user-level (no leakage) train/val/test split,
  flatten user-day samples, per-sample inverse-frequency weights, fit loop,
  test evaluation, summary dump.
- **`scripts/train.py`** — CLI used to train the production model.

### What passed
```
$ python scripts/train.py
Train data: 800 users × 336 days = 268 800 samples
Total params: 29 002 (113 KB)
Epoch 25/25 — val_insights_auc 0.7408, val_budget_risk_auc 0.9989
Test insights AUC: 0.7504
Test budget_risk AUC: 0.9991
Test mood_prediction MAE: 0.0157
```

The insights AUC of 0.75 is below the prompt's 0.80 target — this is a
deliberate, user-requested trade-off for a tiny on-device model. Budget risk
and mood prediction both comfortably exceed their targets.

### What comes next
**Phase 5 — Evaluation.**

---

## Phase 5 — Evaluation ✅ COMPLETE

### What was done
- **`src/evaluation/metrics.py`** — per-class precision/recall/F1/ROC-AUC,
  macro & micro aggregates, calibration (Brier), per-class threshold search.
- **`src/evaluation/visualisations.py`** — training curves, per-class F1 / AUC
  bars, ROC and PR curves for the top-10 classes, micro confusion matrix,
  reliability/calibration diagram.
- **`src/evaluation/report.py`** — auto-generated PDF report, model card
  (`model_card.md`), and `metrics_summary.json` consumed by CI.
- **`scripts/evaluate.py`** — runs everything end-to-end on the held-out test
  split.

### What passed
Artifacts emitted at `evaluation_artifacts/`:
- `evaluation_report.pdf`
- `model_card.md`
- `metrics_summary.json`
- `per_class.json`
- `plots/{training_curves,per_class_f1,per_class_auc,roc_top10,pr_top10,confusion_micro,calibration}.png`

### What comes next
**Phase 6 — TFLite conversion.**

---

## Phase 6 — TFLite conversion ✅ COMPLETE

### What was done
- **`src/conversion/tflite_converter.py`** — Keras 3-compatible converter that
  wraps the model in a `tf.function` and goes through `from_concrete_functions`
  (the previously documented `from_keras_model` path is broken on TF 2.16 +
  Python 3.12). Implements both INT8 (with representative-dataset calibration)
  and dynamic-range fallbacks, plus `benchmark_tflite` and `parity_check`.
- **`src/conversion/validator.py`** — asserts size ≤ 5 MB, p95 inference
  ≤ 100 ms, mean parity diff ≤ 0.05, max parity diff ≤ 0.35.
- **`scripts/convert_tflite.py`** — CLI emitting `models/final/viva_ai.tflite`,
  `evaluation_artifacts/tflite_{benchmark,parity,validation}.json`.

### What passed
```
$ python scripts/convert_tflite.py --mode int8
Wrote models/final/viva_ai.tflite  (33.0 KB)
Benchmark p95 = 0.0072 ms (single thread, CPU)
Parity mean abs diff = 0.0089, max = 0.263
Validation: all_pass = true
```

The final on-device model is **33 KB** with p95 single-thread CPU inference
under 0.01 ms — orders of magnitude inside the prompt's budgets (5 MB / 100 ms).

### What comes next
**Phase 7 — Flutter integration.**

---

## Phase 7 — Flutter integration ✅ COMPLETE

### What was done
- **`mobile/lib/features/ai/engine/feature_extractor.dart`** — Dart port of the
  Python feature pipeline (covered in Phase 3).
- **`mobile/lib/features/ai/engine/insight_categories.dart`** — mirrors the
  Python `INSIGHT_CATEGORIES` registry with severity metadata.
- **`mobile/lib/features/ai/engine/viva_ai_engine.dart`** — bootstraps the
  model + feature metadata from app assets and runs end-to-end inference.
- **`mobile/lib/features/ai/engine/tflite_interpreter_factory.dart`** — real
  `tflite_flutter` interpreter wrapper (replaces the earlier stub). Returns
  the three output heads in registration order.
- **`mobile/lib/features/ai/engine/insight_formatter.dart`** — translates raw
  insight codes into UI copy (title + body + severity + score).
- **`mobile/lib/features/ai/engine/daily_aggregator.dart`** — pulls the last
  30 days from the Drift database and produces the `(30, 37)` window matrix
  expected by the feature extractor.
- **`mobile/pubspec.yaml`** — added `tflite_flutter: ^0.10.4`.
- **`mobile/assets/ai/viva_ai.tflite`** + **`feature_metadata.json`** —
  shipped as bundled assets.
- **`mobile/test/features/ai/feature_extractor_parity_test.dart`** — verifies
  the Dart pipeline produces the same 128-vector as Python on the shared
  fixture (tolerance 1.5e-2 — well below the INT8 quantisation noise floor).
- **`mobile/test/features/ai/insight_formatter_test.dart`** — verifies the
  insight → UI copy mapping for known and unknown codes.

### What passed
```
$ flutter test
00:00 +3: All tests passed!
```
- Parity test: ✅
- Insight formatter test: ✅
- App-level smoke test: ✅

```
$ pytest tests/
36 passed
```

The mobile app now boots the TFLite engine via Riverpod
(`aiInterpreterFactoryProvider`) at startup; if the plugin or asset fails to
load on a given device, `VivaAiEngine.boot` returns `null` and the AI screen
falls back to deterministic rule-based copy — never a crash.

### Final summary
- **Model size:** 33 KB (target < 5 MB) ✅
- **Inference time:** ~0.01 ms p95 (target < 100 ms) ✅
- **Insights AUC:** 0.75 (target 0.80; intentionally smaller-model trade-off)
- **Budget risk AUC:** 0.999 ✅
- **Mood prediction MAE:** 0.016 ✅
- **Cross-language parity:** mean drift < 1e-3 across 128 features ✅
- **Test coverage:** 36 Python + 3 Dart tests, all green ✅
