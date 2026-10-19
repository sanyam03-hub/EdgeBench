from __future__ import annotations

import numpy as np
import pytest

from edgebench.data import IMAGENETTE_WNID_TO_IMAGENET_IDX
from edgebench.evaluate import top1_accuracy

LABELS = tuple(IMAGENETTE_WNID_TO_IMAGENET_IDX.values())  # (0, 217, ..., 701)


class FixedBackend:
    """Fake backend that returns pre-set logits, one array per predict() call."""

    variant = "fake"
    runtime = "fake"
    precision = "fp32"

    def __init__(self, logits: list[np.ndarray]) -> None:
        self._logits = iter(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return next(self._logits)


def _x(n: int) -> np.ndarray:
    return np.zeros((n, 3, 4, 4), dtype=np.float32)


def _onehot_1000(targets: list[int]) -> np.ndarray:
    logits = np.zeros((len(targets), 1000), dtype=np.float32)
    logits[np.arange(len(targets)), targets] = 1.0
    return logits


def test_perfect_predictions_1000_classes() -> None:
    labels = np.array([0, 217, 701, 574])
    backend = FixedBackend([_onehot_1000(labels.tolist())])
    assert top1_accuracy(backend, [(_x(4), labels)], LABELS) == 100.0


def test_non_imagenette_logits_are_ignored() -> None:
    labels = np.array([217, 482])
    logits = _onehot_1000(labels.tolist())
    logits[:, 5] = 50.0  # class 5 is not an Imagenette class: must not win
    assert top1_accuracy(FixedBackend([logits]), [(_x(2), labels)], LABELS) == 100.0


def test_ten_output_model_maps_positions_to_labels() -> None:
    labels = np.array([LABELS[3], LABELS[9]])
    logits = np.zeros((2, 10), dtype=np.float32)
    logits[0, 3] = 1.0
    logits[1, 0] = 1.0  # wrong: predicts LABELS[0]
    assert top1_accuracy(FixedBackend([logits]), [(_x(2), labels)], LABELS) == 50.0


def test_accumulates_over_batches() -> None:
    labels_a, labels_b = np.array([0, 217]), np.array([482, 491])
    wrong = _onehot_1000([0, 0])
    backend = FixedBackend([_onehot_1000(labels_a.tolist()), wrong])
    result = top1_accuracy(backend, [(_x(2), labels_a), (_x(2), labels_b)], LABELS)
    assert result == pytest.approx(50.0)


def test_three_of_four() -> None:
    labels = np.array([0, 217, 482, 491])
    backend = FixedBackend([_onehot_1000([0, 217, 482, 0])])
    assert top1_accuracy(backend, [(_x(4), labels)], LABELS) == pytest.approx(75.0)


def test_empty_batches_raise() -> None:
    with pytest.raises(ValueError, match="no images"):
        top1_accuracy(FixedBackend([]), [], LABELS)


def test_unexpected_logit_width_raises() -> None:
    backend = FixedBackend([np.zeros((2, 7), dtype=np.float32)])
    with pytest.raises(ValueError, match="7"):
        top1_accuracy(backend, [(_x(2), np.array([0, 217]))], LABELS)


def test_label_outside_classes_raises() -> None:
    backend = FixedBackend([_onehot_1000([0])])
    with pytest.raises(ValueError, match="999"):
        top1_accuracy(backend, [(_x(1), np.array([999]))], LABELS)
