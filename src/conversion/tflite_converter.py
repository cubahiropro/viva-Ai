"""Convert a trained Keras model to INT8-quantised TFLite.

Per section 7 of the prompt:
    - Use `tf.lite.TFLiteConverter.from_keras_model(model)`
    - Apply post-training quantisation (INT8) using a representative dataset
    - Verify size <= 5 MB and inference time <= 100 ms
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf


def _model_to_concrete_function(model: tf.keras.Model):
    """Wrap a Keras 3 model in a tf.function and return its concrete function.

    Keras 3 + TF 2.15/2.16 broke `from_keras_model()` and `model.export()` is
    flaky on Python 3.12. Building a tf.function around the model and using
    `from_concrete_functions` is the most reliable conversion path.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    spec = tf.TensorSpec(shape=[1, int(input_shape[-1])], dtype=tf.float32)

    @tf.function(input_signature=[spec])
    def serving(x: tf.Tensor):
        return model(x, training=False)

    return serving.get_concrete_function()


def _representative_dataset_fn(samples: np.ndarray):
    """Yields up to 500 single-sample float32 batches for INT8 calibration."""
    n = min(samples.shape[0], 500)

    def gen() -> Iterable[list[np.ndarray]]:
        for i in range(n):
            yield [samples[i : i + 1].astype(np.float32)]

    return gen


def convert_to_tflite_int8(
    model: tf.keras.Model,
    representative_samples: np.ndarray,
    out_path: Path,
) -> dict[str, Any]:
    """INT8 conversion with float fallback for unsupported ops."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concrete = _model_to_concrete_function(model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_fn(representative_samples)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,  # float fallback for ops without int8 kernels
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_bytes = converter.convert()

    out_path.write_bytes(tflite_bytes)
    return {
        "size_bytes": len(tflite_bytes),
        "path": str(out_path),
    }


def convert_to_tflite_dynamic_range(
    model: tf.keras.Model, out_path: Path
) -> dict[str, Any]:
    """Fallback / baseline: dynamic-range quantisation (no representative set)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concrete = _model_to_concrete_function(model)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    return {
        "size_bytes": len(tflite_bytes),
        "path": str(out_path),
    }


def benchmark_tflite(
    tflite_path: Path,
    samples: np.ndarray,
    runs: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    n = samples.shape[0]
    rng = np.random.default_rng(123)

    def _run(idx: int) -> None:
        x = samples[idx : idx + 1].astype(in_detail["dtype"])
        interp.set_tensor(in_detail["index"], x)
        interp.invoke()
        for od in out_details:
            interp.get_tensor(od["index"])

    for _ in range(warmup):
        _run(int(rng.integers(0, n)))

    times = []
    for _ in range(runs):
        idx = int(rng.integers(0, n))
        t0 = time.perf_counter()
        _run(idx)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "runs": float(runs),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def discover_output_mapping(
    keras_model: tf.keras.Model,
    tflite_path: Path,
    samples: np.ndarray,
    n: int = 8,
) -> dict[str, int]:
    """Discover which TFLite output index corresponds to which Keras output name.

    The TFLite converter does not preserve Keras output names through the
    `from_concrete_functions` path — outputs become `Identity`, `Identity_1`,
    ... in alphabetical-by-Keras-name order. To stay robust against any future
    re-conversion, we *empirically* match each TFLite output back to a Keras
    output by minimum mean-squared difference on a few samples.
    """
    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    n = min(n, samples.shape[0])
    keras_out = keras_model.predict(samples[:n], verbose=0)
    if not isinstance(keras_out, dict):
        # Single-output model – trivial.
        return {keras_model.output_names[0]: int(out_details[0]["index"])}

    # Run TFLite once per sample and stash all outputs.
    tfl_results: list[list[np.ndarray]] = [[] for _ in out_details]
    for i in range(n):
        interp.set_tensor(in_d["index"], samples[i : i + 1].astype(in_d["dtype"]))
        interp.invoke()
        for j, od in enumerate(out_details):
            tfl_results[j].append(interp.get_tensor(od["index"])[0].copy())

    # For each Keras name, find the TFLite output index whose shape matches and
    # whose values are closest in L2.
    mapping: dict[str, int] = {}
    used = set()
    for k_name, k_arr in keras_out.items():
        k_shape = tuple(np.asarray(k_arr).shape[1:])
        best_j = -1
        best_err = float("inf")
        for j, tfl_per_sample in enumerate(tfl_results):
            if j in used:
                continue
            tfl_shape = tuple(np.asarray(tfl_per_sample[0]).shape)
            if tfl_shape != k_shape:
                continue
            tfl_arr = np.stack(tfl_per_sample)
            err = float(np.mean((tfl_arr - np.asarray(k_arr)) ** 2))
            if err < best_err:
                best_err = err
                best_j = j
        if best_j < 0:
            raise RuntimeError(f"Could not match Keras output {k_name!r} to any TFLite output.")
        used.add(best_j)
        mapping[k_name] = best_j
    return mapping


def parity_check(
    keras_model: tf.keras.Model,
    tflite_path: Path,
    samples: np.ndarray,
    output_name: str = "insights",
    n: int = 50,
    tol: float = 0.05,
) -> dict[str, Any]:
    """Compare Keras vs TFLite outputs on `n` samples — return mean/max abs diff."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    # Find the output tensor that matches the desired head:
    # Sort by shape — insights = (1, K) where K=40, budget=(1,1), mood=(1,1)
    out_d = max(out_details, key=lambda d: int(np.prod(d["shape"])))

    n = min(n, samples.shape[0])
    keras_out = keras_model.predict(samples[:n], verbose=0)
    if isinstance(keras_out, dict):
        keras_arr = np.asarray(keras_out[output_name])
    else:
        keras_arr = np.asarray(keras_out[0])

    tfl_arr = np.zeros_like(keras_arr)
    for i in range(n):
        x = samples[i : i + 1].astype(in_d["dtype"])
        interp.set_tensor(in_d["index"], x)
        interp.invoke()
        tfl_arr[i] = interp.get_tensor(out_d["index"])[0]

    diffs = np.abs(keras_arr - tfl_arr)
    return {
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "within_tol": bool(diffs.max() <= tol),
        "tol": float(tol),
        "n": int(n),
    }
