#!/usr/bin/env python3
"""CLI: convert the trained Keras model to INT8 TFLite, validate, and benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from conversion.tflite_converter import (  # noqa: E402
    benchmark_tflite,
    convert_to_tflite_dynamic_range,
    convert_to_tflite_int8,
    discover_output_mapping,
    parity_check,
)
from conversion.validator import TFLiteAssertions, validate_tflite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Keras → TFLite INT8.")
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "best.keras",
    )
    parser.add_argument(
        "--features", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features.npz",
    )
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "models" / "final" / "viva_ai.tflite",
    )
    parser.add_argument("--mode", choices=["int8", "dynamic"], default="int8")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "evaluation_artifacts")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    print(f"Loading Keras model from {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print(f"Loading features for representative dataset from {args.features}")
    feats = np.load(args.features)["features"]
    flat = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(flat.shape[0])
    repr_samples = flat[perm[:1000]]

    print(f"Converting → {args.mode} TFLite ...")
    if args.mode == "int8":
        info = convert_to_tflite_int8(model, repr_samples, args.out)
    else:
        info = convert_to_tflite_dynamic_range(model, args.out)
    print(f"Wrote {info['path']}  ({info['size_bytes'] / 1024:.1f} KB)")

    print("Benchmarking TFLite inference (1 thread)...")
    bench = benchmark_tflite(args.out, repr_samples[:200], runs=200, warmup=20)
    print(json.dumps(bench, indent=2))
    (args.out_dir / "tflite_benchmark.json").write_text(json.dumps(bench, indent=2))

    print("Parity check vs Keras...")
    parity = parity_check(model, args.out, repr_samples[:50])
    print(json.dumps(parity, indent=2))
    (args.out_dir / "tflite_parity.json").write_text(json.dumps(parity, indent=2))

    print("Discovering TFLite output mapping (Keras name → TFLite index)...")
    output_mapping = discover_output_mapping(model, args.out, repr_samples[:8])
    print(json.dumps(output_mapping, indent=2))
    (args.out_dir / "tflite_output_mapping.json").write_text(
        json.dumps(output_mapping, indent=2)
    )

    # Merge mapping into the feature metadata file the Flutter engine consumes.
    fm_path = PROJECT_ROOT / "models" / "final" / "feature_metadata.json"
    if fm_path.exists():
        fm = json.loads(fm_path.read_text())
        fm["output_mapping"] = {k: int(v) for k, v in output_mapping.items()}
        fm_path.write_text(json.dumps(fm, indent=2))
        print(f"Merged output_mapping into {fm_path}")

    validation = validate_tflite(info, bench, parity, TFLiteAssertions())
    (args.out_dir / "tflite_validation.json").write_text(json.dumps(validation, indent=2))
    print("\nValidation:")
    print(json.dumps(validation, indent=2))
    if not validation["all_pass"]:
        print("WARNING: validation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
