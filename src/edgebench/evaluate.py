"""Top-1 accuracy over the Imagenette eval subset (Data_Schema.md §5). Does not time anything."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from edgebench.backends.base import Backend


def top1_accuracy(
    backend: Backend,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    label_indices: Sequence[int],
) -> float:
    """10-way top-1 accuracy in percent: argmax over only the Imagenette class logits.

    Args:
        backend: Model to evaluate.
        batches: ``(inputs, labels)`` pairs; labels are ImageNet class indices. Consumed lazily,
            so the full eval set is never held in memory.
        label_indices: The ImageNet indices of the evaluated classes, in a fixed order. A model
            with 1000 outputs is restricted to these columns; a model with exactly
            ``len(label_indices)`` outputs is read as already restricted (output j = class j).

    Returns:
        Percentage of correct predictions (0-100).

    Raises:
        ValueError: No images, an unexpected number of logit columns, or a label that is not
            one of ``label_indices``.
    """
    classes = np.asarray(label_indices)
    correct = total = 0
    for inputs, labels in batches:
        logits = backend.predict(inputs)
        predicted = classes[np.argmax(_class_logits(logits, classes), axis=1)]
        unknown = set(np.asarray(labels).tolist()) - set(classes.tolist())
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} are not among the evaluated classes")
        correct += int(np.sum(predicted == labels))
        total += len(labels)
    if total == 0:
        raise ValueError("Cannot compute accuracy: no images were evaluated")
    return 100.0 * correct / total


def _class_logits(logits: np.ndarray, classes: np.ndarray) -> np.ndarray:
    width = logits.shape[1]
    if width == len(classes):
        return logits
    if width > int(classes.max()):
        return logits[:, classes]
    raise ValueError(
        f"Model has {width} logit columns; expected {len(classes)} or more than {classes.max()}"
    )
