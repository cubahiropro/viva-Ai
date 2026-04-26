#!/usr/bin/env python3
"""Dump a tiny parity fixture used by the Dart unit tests.

The Dart port of the feature pipeline replays the same 30-day window and asserts
its 128-vector matches Python's to within 1e-4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data_generation.master_generator import DAILY_COLUMNS, generate_one_user
from feature_engineering.feature_pipeline import (
    FEATURE_NAMES,
    WINDOW_DAYS,
    compute_features_for_window,
)


def main() -> int:
    rng = np.random.default_rng(12345)
    days = np.arange(
        np.datetime64("2024-01-01", "D"),
        np.datetime64("2024-01-01", "D") + 60,
        dtype="datetime64[D]",
    )
    profile, daily, _ = generate_one_user(
        user_id=0, rng=rng, days=days, archetype="disciplined_professional"
    )
    target_day = len(days) - 1
    window = daily[target_day - WINDOW_DAYS + 1 : target_day + 1]
    # Sanitise the window the same way the Flutter `DailyAggregator` does:
    # missing logs come through as 0.0 (and an explicit `*_logged` flag), so
    # there are no NaN/Inf values flowing into features at inference time.
    window = np.where(np.isfinite(window), window, 0.0)
    features = compute_features_for_window(
        window,
        columns=DAILY_COLUMNS,
        today_date=days[target_day],
        days_since_first_use=target_day,
        monthly_income=profile["monthly_income"],
        weight_baseline_kg=profile["weight_baseline_kg"],
        height_cm=profile["height_cm"],
    )

    # Defensive: any residual NaN/Inf that snuck through feature math becomes 0.
    safe_features = np.where(np.isfinite(features), features, 0.0)

    out_path = PROJECT_ROOT / "flutter_integration" / "parity_fixture.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "columns": DAILY_COLUMNS,
        "feature_names": FEATURE_NAMES,
        "window": window.tolist(),
        "today_iso": str(days[target_day]),
        "days_since_first_use": int(target_day),
        "monthly_income": float(profile["monthly_income"]),
        "weight_baseline_kg": float(profile["weight_baseline_kg"]),
        "height_cm": float(profile["height_cm"]),
        "expected_features": safe_features.tolist(),
    }, indent=2, allow_nan=False))
    print(f"Wrote parity fixture → {out_path}  ({features.shape[0]} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
