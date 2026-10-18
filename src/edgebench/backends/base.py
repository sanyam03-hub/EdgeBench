"""The Backend protocol (Data_Schema.md §8) and shared input validation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from edgebench.errors import BackendError


class Backend(Protocol):
    """A loaded model that maps a float32 (N, 3, H, W) batch to float32 logits (N, C)."""

    variant: str
    runtime: str
    precision: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: float32, shape (N, 3, H, W). Returns float32 logits (N, C)."""
        ...


def validate_input(x: object, variant: str) -> np.ndarray:
    """Check ``x`` is a non-empty float32 (N, 3, H, W) numpy array; raise BackendError if not."""
    if not isinstance(x, np.ndarray):
        raise BackendError(f"{variant}: input must be a numpy array, got {type(x).__name__}")
    if x.dtype != np.float32:
        raise BackendError(f"{variant}: input dtype must be float32, got {x.dtype}")
    if x.ndim != 4 or x.shape[1] != 3:
        raise BackendError(f"{variant}: input must have shape (N, 3, H, W), got {x.shape}")
    if x.shape[0] == 0:
        raise BackendError(f"{variant}: input batch is empty")
    return x
