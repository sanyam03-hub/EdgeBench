"""Numerical parity between a reference output (PyTorch) and a converted model's output.

Pure numpy: must not import torch or onnxruntime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

TOLERANCE_KEYS = frozenset({"max_abs_err", "min_cosine", "min_top1_agreement"})


@dataclass(frozen=True)
class ParityReport:
    """Agreement metrics between two (N, C) logit arrays and whether they meet the tolerance."""

    n_samples: int
    max_abs_err: float
    mean_abs_err: float
    cosine_similarity: float
    top1_agreement: float
    within_tolerance: bool
    tolerance: dict[str, float]


def compare(
    reference: np.ndarray, candidate: np.ndarray, tolerance: Mapping[str, float]
) -> ParityReport:
    """Compare candidate logits against reference logits.

    Args:
        reference: (N, C) logits from the reference model.
        candidate: (N, C) logits from the converted model.
        tolerance: Thresholds to enforce; any subset of ``TOLERANCE_KEYS``. Only the
            thresholds present are checked (INT8 variants pass only ``min_top1_agreement``).

    Returns:
        A ParityReport. Metrics are computed in float64.

    Raises:
        ValueError: On non-2-D, empty or mismatched-shape inputs, or unknown tolerance keys.
    """
    _validate(reference, candidate, tolerance)
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)

    abs_err = np.abs(ref - cand)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    cosine = float(_row_cosine(ref, cand).mean())
    top1 = float(np.mean(ref.argmax(axis=1) == cand.argmax(axis=1)))

    within = (
        max_abs_err <= tolerance.get("max_abs_err", np.inf)
        and cosine >= tolerance.get("min_cosine", -np.inf)
        and top1 >= tolerance.get("min_top1_agreement", -np.inf)
    )
    return ParityReport(
        n_samples=ref.shape[0],
        max_abs_err=max_abs_err,
        mean_abs_err=mean_abs_err,
        cosine_similarity=cosine,
        top1_agreement=top1,
        within_tolerance=bool(within),
        tolerance=dict(tolerance),
    )


def _validate(reference: np.ndarray, candidate: np.ndarray, tolerance: Mapping[str, float]) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Parity shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    if reference.ndim != 2:
        raise ValueError(f"Parity expects 2-D (N, C) logits, got shape {reference.shape}")
    if reference.size == 0:
        raise ValueError(f"Parity got empty arrays of shape {reference.shape}")
    unknown = set(tolerance) - TOLERANCE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown parity tolerance keys {sorted(unknown)}; allowed: {sorted(TOLERANCE_KEYS)}"
        )


def _row_cosine(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. Both rows zero -> 1.0; exactly one zero -> 0.0 (never NaN)."""
    norms = np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1)
    dots = np.einsum("ij,ij->i", ref, cand)
    ref_zero = ~ref.any(axis=1)
    cand_zero = ~cand.any(axis=1)
    safe = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
    return np.where(ref_zero & cand_zero, 1.0, safe)
