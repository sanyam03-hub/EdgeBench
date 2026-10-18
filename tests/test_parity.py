from __future__ import annotations

import math

import numpy as np
import pytest

from edgebench.parity import ParityReport, compare

FP32_TOL = {"max_abs_err": 1e-3, "min_cosine": 0.9999, "min_top1_agreement": 0.99}


def _logits(n: int = 10, c: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, c)).astype(np.float32)


def test_identical_arrays_are_perfect() -> None:
    ref = _logits()
    report = compare(ref, ref.copy(), FP32_TOL)
    assert isinstance(report, ParityReport)
    assert report.n_samples == 10
    assert report.max_abs_err == 0.0
    assert report.mean_abs_err == 0.0
    assert report.cosine_similarity == pytest.approx(1.0)
    assert report.top1_agreement == 1.0
    assert report.within_tolerance is True
    assert report.tolerance == FP32_TOL


def test_known_perturbation() -> None:
    ref = _logits().astype(np.float64)
    report = compare(ref, ref + 1e-3, FP32_TOL)
    assert report.max_abs_err == pytest.approx(1e-3)
    assert report.mean_abs_err == pytest.approx(1e-3)
    assert report.top1_agreement == 1.0


def test_one_flipped_argmax_out_of_ten() -> None:
    ref = np.zeros((10, 3))
    ref[:, 0] = 1.0
    cand = ref.copy()
    cand[4] = [0.0, 1.0, 0.0]
    report = compare(ref, cand, {"min_top1_agreement": 0.95})
    assert report.top1_agreement == pytest.approx(0.9)
    assert report.within_tolerance is False


def test_max_abs_err_above_tolerance_fails() -> None:
    ref = _logits().astype(np.float64)
    report = compare(ref, ref + 1e-2, FP32_TOL)
    assert report.top1_agreement == 1.0
    assert report.within_tolerance is False


def test_only_given_thresholds_are_checked() -> None:
    ref = _logits().astype(np.float64)
    report = compare(ref, ref + 0.5, {"min_top1_agreement": 0.9})
    assert report.max_abs_err == pytest.approx(0.5)
    assert report.within_tolerance is True


def test_shape_mismatch_mentions_both_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(10, 5\).*\(10, 4\)"):
        compare(_logits(10, 5), _logits(10, 4), FP32_TOL)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_non_2d_input_rejected(shape: tuple[int, ...]) -> None:
    arr = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        compare(arr, arr, FP32_TOL)


def test_empty_input_rejected() -> None:
    arr = np.zeros((0, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        compare(arr, arr, FP32_TOL)


def test_unknown_tolerance_key_rejected() -> None:
    ref = _logits()
    with pytest.raises(ValueError, match="max_rel_err"):
        compare(ref, ref, {"max_rel_err": 0.1})


def test_zero_rows_do_not_produce_nan() -> None:
    ref = np.zeros((2, 3))
    cand = np.zeros((2, 3))
    cand[1] = [1.0, 0.0, 0.0]
    report = compare(ref, cand, {})
    # both-zero row counts as identical (1.0), one-zero row as orthogonal (0.0)
    assert not math.isnan(report.cosine_similarity)
    assert report.cosine_similarity == pytest.approx(0.5)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_low_precision_inputs_computed_in_float64(dtype: type) -> None:
    ref = np.array([[1.0, 2.0, 3.0]], dtype=dtype)
    cand = np.array([[1.0, 2.0, 3.5]], dtype=dtype)
    report = compare(ref, cand, {})
    assert type(report.max_abs_err) is float
    assert report.max_abs_err == pytest.approx(0.5)
    assert report.mean_abs_err == pytest.approx(0.5 / 3)
