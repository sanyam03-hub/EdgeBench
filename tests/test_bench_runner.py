from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgebench.bench import benchmark, file_size_mb

PREDICT_NS = 2_000_000  # each fake predict "takes" 2 ms on the fake clock


class FakeClockBackend:
    """predict() advances a fake clock; nothing else does, so samples measure predict only."""

    variant = "fake"
    runtime = "fake"
    precision = "fp32"

    def __init__(self) -> None:
        self.now_ns = 0
        self.calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []

    def clock(self) -> int:
        return self.now_ns

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen_shapes.append(x.shape)
        self.now_ns += PREDICT_NS
        return np.zeros((x.shape[0], 10), dtype=np.float32)


def _batch(n: int = 4) -> np.ndarray:
    return np.zeros((n, 3, 8, 8), dtype=np.float32)


def test_times_only_predict_and_excludes_warmup() -> None:
    backend = FakeClockBackend()
    stats = benchmark(
        backend, _batch(), warmup=5, runs=20, noisy_cv_threshold=0.1, clock=backend.clock
    )
    assert backend.calls == 25
    assert stats.n == 20
    assert stats.p50_ms == pytest.approx(2.0)
    assert stats.std_ms == 0.0
    assert stats.noisy is False


def test_uses_the_given_batch() -> None:
    backend = FakeClockBackend()
    benchmark(backend, _batch(8), warmup=1, runs=2, noisy_cv_threshold=0.1, clock=backend.clock)
    assert set(backend.seen_shapes) == {(8, 3, 8, 8)}


def test_zero_warmup_allowed() -> None:
    backend = FakeClockBackend()
    stats = benchmark(
        backend, _batch(), warmup=0, runs=3, noisy_cv_threshold=0.1, clock=backend.clock
    )
    assert backend.calls == 3
    assert stats.n == 3


@pytest.mark.parametrize(("warmup", "runs"), [(-1, 3), (0, 0)])
def test_invalid_counts_raise(warmup: int, runs: int) -> None:
    backend = FakeClockBackend()
    with pytest.raises(ValueError):
        benchmark(backend, _batch(), warmup=warmup, runs=runs, noisy_cv_threshold=0.1)


def test_real_clock_produces_positive_samples() -> None:
    stats = benchmark(FakeClockBackend(), _batch(), warmup=0, runs=3, noisy_cv_threshold=0.1)
    assert stats.n == 3
    assert stats.p50_ms >= 0.0


def test_file_size_mb(tmp_path: Path) -> None:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"\0" * 2_500_000)
    assert file_size_mb(path) == pytest.approx(2.5)
