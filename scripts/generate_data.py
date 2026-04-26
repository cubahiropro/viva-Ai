#!/usr/bin/env python3
"""CLI: generate synthetic Viva AI training data.

Examples
--------
    python scripts/generate_data.py --users 100 --days 365 --seed 42
    python scripts/generate_data.py --users 5000 --days 365 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data_generation.master_generator import (  # noqa: E402
    generate_dataset,
    positive_counts_per_class,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Viva AI synthetic data.")
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=10,
        help=(
            "Assert each insight class has at least this many positive examples. "
            "Use 500 for the full 5000-user run, 10 for the 100-user smoke run."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.users} users × {args.days} days, seed={args.seed}")
    print(f"Output → {args.out_dir}")

    paths = generate_dataset(
        num_users=args.users,
        days_per_user=args.days,
        seed=args.seed,
        out_dir=args.out_dir,
        start_date=args.start_date,
        show_progress=not args.no_progress,
    )

    import numpy as np
    labels = np.load(paths["labels"])["labels"]
    counts = positive_counts_per_class(labels)
    counts_path = paths["users"].parent / "label_counts.json"
    counts_path.write_text(json.dumps(counts, indent=2))

    print("\nPositive counts per insight category:")
    for code, c in sorted(counts.items(), key=lambda kv: kv[1]):
        flag = "OK" if c >= args.min_positive else "LOW"
        print(f"  [{flag:3}] {code:30s} {c:>9d}")

    too_low = [c for c, n in counts.items() if n < args.min_positive]
    if too_low:
        print(
            f"\nWARNING: {len(too_low)} insight categories under "
            f"min_positive={args.min_positive}: {too_low}"
        )
        return 2
    print("\nAll insight categories meet the minimum positive count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
