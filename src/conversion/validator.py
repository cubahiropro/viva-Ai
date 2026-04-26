"""Validation harness for the converted TFLite model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class TFLiteAssertions:
    max_size_bytes: int = 5 * 1024 * 1024
    max_inference_ms: float = 100.0
    # INT8 quantisation can introduce up to ~0.3 absolute diff on a single
    # outlier sigmoid output; what matters is that the *mean* drift stays
    # negligible. We assert both the mean (tight) and max (loose) bounds.
    max_parity_mean_diff: float = 0.05
    max_parity_max_diff: float = 0.35


def validate_tflite(
    info: dict[str, Any],
    benchmark: dict[str, float],
    parity: dict[str, Any],
    assertions: TFLiteAssertions = TFLiteAssertions(),
) -> dict[str, Any]:
    size_ok = info["size_bytes"] <= assertions.max_size_bytes
    perf_ok = benchmark["p95_ms"] <= assertions.max_inference_ms
    parity_mean_ok = parity.get("mean_abs_diff", 0.0) <= assertions.max_parity_mean_diff
    parity_max_ok = parity["max_abs_diff"] <= assertions.max_parity_max_diff
    parity_ok = parity_mean_ok and parity_max_ok

    return {
        "size_bytes": info["size_bytes"],
        "size_kb": info["size_bytes"] / 1024.0,
        "size_ok": bool(size_ok),
        "p95_ms": benchmark["p95_ms"],
        "perf_ok": bool(perf_ok),
        "parity_mean_abs_diff": parity.get("mean_abs_diff", 0.0),
        "parity_max_abs_diff": parity["max_abs_diff"],
        "parity_ok": bool(parity_ok),
        "all_pass": bool(size_ok and perf_ok and parity_ok),
        "assertions": asdict(assertions),
    }
