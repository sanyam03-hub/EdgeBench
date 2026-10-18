"""ONNX Runtime CPU backend for ``onnx_*`` variants (FP32 and INT8)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.capi import onnxruntime_pybind11_state as _ort_state

from edgebench.backends.base import validate_input
from edgebench.errors import BackendError

# ORT's native errors share no base class below Exception; catch exactly the ones it defines.
_ORT_ERRORS = tuple(
    obj for obj in vars(_ort_state).values() if isinstance(obj, type) and issubclass(obj, Exception)
)


class OrtBackend:
    """An ``InferenceSession`` on CPUExecutionProvider with explicit threading."""

    runtime = "onnxruntime"

    def __init__(
        self,
        path: Path,
        *,
        variant: str,
        precision: str,
        intra_op_threads: int,
        inter_op_threads: int,
    ) -> None:
        if not path.is_file():
            raise BackendError(f"{variant}: model file not found: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except _ORT_ERRORS as exc:
            raise BackendError(
                f"{variant}: ONNX Runtime could not load {path}: {type(exc).__name__}: {exc}"
            ) from exc
        self.variant = variant
        self.precision = precision
        self._input_name = self._session.get_inputs()[0].name

    @property
    def session_options(self) -> ort.SessionOptions:
        """The options the session was created with (threads, optimization level)."""
        return self._session.get_session_options()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return float32 logits (N, C) for a float32 (N, 3, H, W) batch."""
        validate_input(x, self.variant)
        out = self._session.run(None, {self._input_name: x})[0]
        return np.asarray(out, dtype=np.float32)
