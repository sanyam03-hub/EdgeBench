from __future__ import annotations

import math

import numpy as np
import pytest

from edgebench.bench import LatencyStats, summarize_latencies


def test_known_list_matches_numpy_percentiles() -> None:
    samples = [float(x) for x in range(1, 101)]
    stats = summarize_latencies(samples, noisy_cv_threshold=1.0)
    assert isinstance(stats, LatencyStats)
    assert stats.n == 100
    assert stats.p50_ms == pytest.approx(np.percentile(samples, 50))
    assert stats.p90_ms == pytest.approx(np.percentile(samples, 90))
    assert stats.p99_ms == pytest.approx(np.percentile(samples, 99))
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.mean_ms == pytest.approx(50.5)
    assert stats.std_ms == pytest.approx(np.std(samples, ddof=1))
    assert stats.iqr_ms == pytest.approx(np.percentile(samples, 75) - np.percentile(samples, 25))
    assert stats.cv == pytest.approx(stats.std_ms / stats.mean_ms)


def test_constant_latencies_have_zero_spread() -> None:
    stats = summarize_latencies([4.0] * 20, noisy_cv_threshold=0.1)
    assert stats.std_ms == 0.0
    assert stats.cv == 0.0
    assert stats.iqr_ms == 0.0
    assert stats.noisy is False


def test_highly_variable_list_is_noisy() -> None:
    stats = summarize_latencies([1.0, 10.0, 1.0, 10.0, 1.0, 10.0], noisy_cv_threshold=0.1)
    assert stats.cv > 0.1
    assert stats.noisy is True


def test_cv_equal_to_threshold_is_not_noisy() -> None:
    # noisy means strictly above the threshold
    stats = summarize_latencies([4.0] * 5, noisy_cv_threshold=0.0)
    assert stats.cv == 0.0
    assert stats.noisy is False


def test_single_sample_reports_zero_std() -> None:
    stats = summarize_latencies([3.5], noisy_cv_threshold=0.1)
    assert stats.n == 1
    assert stats.std_ms == 0.0
    assert stats.cv == 0.0
    assert stats.p50_ms == stats.p99_ms == 3.5


def test_empty_list_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_latencies([], noisy_cv_threshold=0.1)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_samples_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="latency"):
        summarize_latencies([1.0, bad], noisy_cv_threshold=0.1)


def test_fields_are_python_floats() -> None:
    stats = summarize_latencies(np.array([1.0, 2.0, 3.0]), noisy_cv_threshold=0.1)
    for name in ("p50_ms", "p90_ms", "p99_ms", "mean_ms", "std_ms", "iqr_ms", "cv"):
        assert type(getattr(stats, name)) is float
    assert type(stats.n) is int
    assert type(stats.noisy) is bool
