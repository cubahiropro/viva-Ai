#!/usr/bin/env python3
"""CLI: build the (num_users × valid_days × 128) feature tensor + labels.

Output:
    data/processed/features.npz       float32 (num_users, valid_days, 128)
    data/processed/labels.npz         uint8   (num_users, valid_days, 40)
    models/final/feature_metadata.json   normaliser params + column ordering
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from feature_engineering.feature_pipeline import (  # noqa: E402
    FEATURE_NAMES,
    INPUT_DIM,
    WINDOW_DAYS,
    apply_normaliser,
    compute_user_features,
    fit_normaliser,
)
import pandas as pd  # noqa: E402

import multiprocessing as mp  # noqa: E402

# Globals set by initialiser for the worker pool
_DAILY: np.ndarray | None = None
_PROFILES: pd.DataFrame | None = None
_COLUMNS: list[str] | None = None


def _init_worker(daily_arr: np.ndarray, profiles: pd.DataFrame, columns: list[str]) -> None:
    global _DAILY, _PROFILES, _COLUMNS
    _DAILY = daily_arr
    _PROFILES = profiles
    _COLUMNS = columns


def _process_one(u: int) -> tuple[int, np.ndarray]:
    assert _DAILY is not None and _PROFILES is not None and _COLUMNS is not None
    profile_row = _PROFILES.iloc[u]
    feats = compute_user_features(
        _DAILY[u],
        columns=_COLUMNS,
        monthly_income=float(profile_row["monthly_income"]),
        weight_baseline_kg=float(profile_row["weight_baseline_kg"]),
        height_cm=float(profile_row["height_cm"]),
    )
    return u, feats[WINDOW_DAYS - 1 :].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Viva AI feature tensors.")
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=PROJECT_ROOT / "models" / "final" / "feature_metadata.json",
    )
    parser.add_argument("--max-users", type=int, default=None,
                        help="Optional cap for fast iteration during dev.")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    daily_npz = np.load(args.synthetic_dir / "daily.npz")
    daily = daily_npz["daily"]                        # (N, D, F_in)
    columns = json.loads(
        (args.synthetic_dir / "feature_columns.json").read_text()
    )
    labels = np.load(args.synthetic_dir / "labels.npz")["labels"]
    users_df = pd.read_parquet(args.synthetic_dir / "users.parquet")

    if args.max_users is not None:
        daily = daily[: args.max_users]
        labels = labels[: args.max_users]
        users_df = users_df.head(args.max_users).reset_index(drop=True)

    N, D, _ = daily.shape
    valid_days = D - WINDOW_DAYS + 1
    print(f"Computing features for {N} users × {valid_days} days × {INPUT_DIM} features "
          f"using {args.workers} workers")
    print(f"Estimated raw size: {N * valid_days * INPUT_DIM * 4 / (1024 ** 2):.1f} MB")

    feats_all = np.zeros((N, valid_days, INPUT_DIM), dtype=np.float32)
    labels_aligned = labels[:, WINDOW_DAYS - 1 :].astype(np.uint8)

    if args.workers <= 1:
        iterator = range(N)
        if not args.no_progress:
            iterator = tqdm(iterator, desc="Features")
        _init_worker(daily, users_df, columns)
        for u in iterator:
            _, f = _process_one(u)
            feats_all[u] = f
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(daily, users_df, columns),
        ) as pool:
            iterator = pool.imap_unordered(_process_one, range(N), chunksize=8)
            if not args.no_progress:
                iterator = tqdm(iterator, total=N, desc="Features")
            for u, f in iterator:
                feats_all[u] = f

    print("Fitting normaliser...")
    norm = fit_normaliser(feats_all)
    feats_all = apply_normaliser(feats_all, norm)
    norm.save(args.metadata_out)

    print(f"Saving features → {args.out_dir / 'features.npz'}")
    np.savez_compressed(args.out_dir / "features.npz", features=feats_all)
    print(f"Saving labels   → {args.out_dir / 'labels.npz'}")
    np.savez_compressed(args.out_dir / "labels.npz", labels=labels_aligned)

    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "num_users": int(N),
                "valid_days": int(valid_days),
                "input_dim": int(INPUT_DIM),
                "window_days": int(WINDOW_DAYS),
                "feature_names": FEATURE_NAMES,
            },
            fh,
            indent=2,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
