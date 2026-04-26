<div align="center">

# Viva AI

**On-device personal-life intelligence model for the Viva Android app.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](LICENSE)

</div>

---

## What it is

A small **TensorFlow Lite** model (target: **< 5 MB**, **< 100 ms CPU**
inference) that turns a user's local Viva data (finance, health, sleep,
mood, tasks, habits, medications) into personalised insights — running
entirely on the user's phone, with no cloud and no data leaving the
device. Trained on synthetic data only.

## Tech

Python 3.10+ · TensorFlow 2.16 (Keras 3) · NumPy / Pandas · scikit-learn
· Faker · pytest · SHAP · matplotlib / seaborn · ReportLab.

## Getting started

```bash
git clone git@github.com:cubahiropro/viva-Ai.git
cd viva-Ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .         # install the viva_ai package locally

cp .env.example .env     # adjust seed / paths if you want
pytest -q                # quick test pass
```

## Pipeline

```bash
# 1. Generate synthetic training data (5 000 users x 365 days)
python scripts/generate_data.py --users 5000 --days 365 --seed 42

# 2. Engineer features
python scripts/process_features.py

# 3. Train
python scripts/train.py --config config/model_config.yaml

# 4. Evaluate (writes evaluation_artifacts/)
python scripts/evaluate.py --model models/checkpoints/best.keras

# 5. Convert + INT8-quantise to TFLite (writes models/final/viva_ai.tflite)
python scripts/convert_tflite.py

# 6. Export the parity fixture used by the Flutter app's Dart tests
python scripts/export_parity_fixture.py
```

## Project structure

```text
viva_ai/
├── config/                     # hyperparameters (YAML)
├── scripts/                    # CLI entry points (generate / train / evaluate / convert)
├── src/                        # package source
│   ├── data_generation/        # synthetic user factories
│   ├── feature_engineering/    # 128-feature pipeline (Python parity with Dart)
│   ├── models/                 # Keras architectures
│   ├── training/               # trainer, callbacks, focal loss
│   ├── evaluation/             # metrics, plots, model card, PDF report
│   ├── conversion/             # TFLite + INT8 quantisation + validator
│   └── inference/              # CPU benchmarking
├── tests/                      # pytest suite
├── flutter_integration/        # Dart files dropped into the Viva mobile app
├── models/final/               # COMMITTED: viva_ai.tflite + feature_metadata.json
└── evaluation_artifacts/       # COMMITTED: model card, metrics, plots, PDF report
```

## Output artefacts

After running the pipeline, two files are copied into the Viva mobile
app at `viva mobile/assets/ai/`:

```text
models/final/viva_ai.tflite           # the on-device model
models/final/feature_metadata.json    # feature names + normalisation stats
```

`flutter_integration/feature_extractor.dart` and
`flutter_integration/parity_fixture.json` are also dropped into the
Flutter app to guarantee Python ↔ Dart numerical parity.

## Privacy & safety

- No real user data is ever used in training (synthetic only).
- The TFLite model runs entirely on-device.
- Insights are observations, not medical advice.
- Inference requires at least 7 days of user data; otherwise the
  Flutter app falls back to a rule-based engine.

## License & contact

© 2026 Evo Inc. — All rights reserved. See [LICENSE](LICENSE).
Developed by **Bebetos** under **Evo Inc.** Bujumbura, Burundi.
Support: WhatsApp [`+25776197888`](https://wa.me/25776197888).
