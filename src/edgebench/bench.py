"""Latency statistics. The timing loop is added in Phase 6.

The statistics part is pure numpy: it must not import torch or onnxruntime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
