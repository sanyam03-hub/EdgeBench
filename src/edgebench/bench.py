"""Latency timing loop and statistics; model file size.

Pure numpy: must not import torch or onnxruntime (backends are passed in).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgebench.backends.base import Backend


@dataclass(frozen=True)
class LatencyStats:
    """Summary of timed inference runs, in milliseconds. ``cv`` is a fraction (std / mean)."""

    n: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    iqr_ms: float
    cv: float
    noisy: bool


def summarize_latencies(samples_ms: Sequence[float], noisy_cv_threshold: float) -> LatencyStats:
    """Summarize every recorded timed run (warmup runs must already be excluded).

    Args:
        samples_ms: One latency per timed run, in milliseconds.
        noisy_cv_threshold: A CV strictly above this marks the measurement as noisy.

    Returns:
        LatencyStats with linear-interpolated percentiles and sample std (ddof=1).
        A single sample has no defined spread; its std and CV are reported as 0.0.

    Raises:
        ValueError: If ``samples_ms`` is empty or contains negative or non-finite values.
    """
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty latency list; record at least one timed run")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"Invalid latency samples (negative or non-finite): {arr.tolist()}")

    p25, p50, p75, p90, p99 = np.percentile(arr, [25, 50, 75, 90, 99])
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return LatencyStats(
        n=int(arr.size),
        p50_ms=float(p50),
        p90_ms=float(p90),
        p99_ms=float(p99),
        mean_ms=mean,
        std_ms=std,
        iqr_ms=float(p75 - p25),
        cv=cv,
        noisy=bool(cv > noisy_cv_threshold),
    )


def benchmark(
    backend: Backend,
    input_batch: np.ndarray,
    warmup: int,
    runs: int,
    noisy_cv_threshold: float,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> LatencyStats:
    """Time ``backend.predict(input_batch)``: ``warmup`` untimed calls, then ``runs`` timed ones.

    Only the predict call is inside the timed region; every timed run is recorded before
    statistics are computed.

    Args:
        clock: Nanosecond clock; injectable so tests do not depend on real timings.

    Raises:
        ValueError: If ``warmup`` < 0 or ``runs`` < 1.
    """
    if warmup < 0 or runs < 1:
        raise ValueError(f"Need warmup >= 0 and runs >= 1, got warmup={warmup}, runs={runs}")
    for _ in range(warmup):
        backend.predict(input_batch)
    samples_ms = []
    for _ in range(runs):
        start = clock()
        backend.predict(input_batch)
        samples_ms.append((clock() - start) / 1e6)
    return summarize_latencies(samples_ms, noisy_cv_threshold)


def file_size_mb(path: Path) -> float:
    """File size in MB (bytes / 1e6, Data_Schema.md §4)."""
    return path.stat().st_size / 1e6
