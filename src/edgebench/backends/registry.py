"""Variant name -> backend. Runtimes are imported only when a variant needs them."""

from __future__ import annotations

import time
from pathlib import Path

from edgebench.backends.base import Backend
from edgebench.config import TargetConfig
from edgebench.errors import BackendError

# variant -> (runtime, precision), Data_Schema.md §6.
VARIANT_SPECS: dict[str, tuple[str, str]] = {
    "torch_fp32": ("torch", "fp32"),
    "onnx_fp32": ("onnxruntime", "fp32"),
    "onnx_int8_dynamic": ("onnxruntime", "int8_dynamic"),
    "onnx_int8_static": ("onnxruntime", "int8_static"),
    "tflite_fp32": ("litert", "fp32"),
    "tflite_int8_dynamic": ("litert", "int8_dynamic"),
}


def create_backend(
    variant: str,
    *,
    model_name: str,
    artifact: Path | None,
    target: TargetConfig,
    weights: str = "DEFAULT",
) -> tuple[Backend, float]:
    """Create the backend for ``variant`` and return it with its load time in ms.

    Args:
        variant: A key of ``VARIANT_SPECS``.
        model_name: Registered model; used to build the in-memory torch reference.
        artifact: Model file for ``onnx_*`` / ``tflite_*`` variants; None for ``torch_fp32``.
        target: Thread settings come from here.
        weights: torchvision weights for ``torch_fp32`` (must match what was exported).

    Raises:
        BackendError: Unknown variant, missing artifact, or the runtime could not load it.
    """
    if variant not in VARIANT_SPECS:
        raise BackendError(f"Unknown variant {variant!r}; known: {list(VARIANT_SPECS)}")
    runtime, precision = VARIANT_SPECS[variant]
    start = time.perf_counter()
    if runtime == "torch":
        backend = _torch(variant, model_name, target, weights)
    elif artifact is None:
        raise BackendError(f"{variant}: an artifact path is required")
    elif runtime == "onnxruntime":
        backend = _onnxruntime(variant, precision, artifact, target)
    else:
        backend = _tflite(variant, precision, artifact)
    return backend, (time.perf_counter() - start) * 1e3


def _torch(variant: str, model_name: str, target: TargetConfig, weights: str) -> Backend:
    from edgebench.backends.torch_backend import TorchBackend
    from edgebench.models import load_model

    module = load_model(model_name, weights).module
    return TorchBackend(module, num_threads=target.intra_op_threads, variant=variant)


def _onnxruntime(variant: str, precision: str, artifact: Path, target: TargetConfig) -> Backend:
    from edgebench.backends.ort_backend import OrtBackend

    return OrtBackend(
        artifact,
        variant=variant,
        precision=precision,
        intra_op_threads=target.intra_op_threads,
        inter_op_threads=target.inter_op_threads,
    )


def _tflite(variant: str, precision: str, artifact: Path) -> Backend:
    try:
        from edgebench.backends import tflite_backend  # type: ignore[attr-defined]
    except ImportError as exc:
        raise BackendError(
            f"{variant}: TFLite backend is not available ({exc}). It is added in Phase 8 and "
            'needs `pip install -e ".[tflite]"`.'
        ) from exc
    return tflite_backend.TFLiteBackend(artifact, variant=variant, precision=precision)
