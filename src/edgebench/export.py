"""PyTorch -> ONNX FP32 export, validated with ``onnx.checker``. Does not quantize."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from edgebench.errors import ExportError

logger = logging.getLogger(__name__)

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
DEFAULT_OPSET = 17
# Legacy TorchScript exporter: the dynamo exporter needs onnxscript, which is not a project
# dependency (decision 2026-10-18 in Memory.md). torch is pinned, so its deprecation is stable.
USE_DYNAMO = False
# torch.onnx.errors.OnnxExporterError subclasses RuntimeError.
_EXPORT_ERRORS = (RuntimeError, TypeError, ValueError)


def export_onnx(
    model: nn.Module, sample_input: np.ndarray, path: Path, opset: int = DEFAULT_OPSET
) -> Path:
    """Export ``model`` to ONNX with a dynamic batch axis and validate the file.

    Args:
        model: Module in eval mode.
        sample_input: float32 (N, 3, H, W) array used to trace the graph.
        path: Output ``.onnx`` file; parent folders are created.
        opset: ONNX opset version.

    Returns:
        ``path``.

    Raises:
        ExportError: If tracing/export fails or the exported model fails ``onnx.checker``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (torch.from_numpy(sample_input),),
                str(path),
                dynamo=USE_DYNAMO,
                opset_version=opset,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            )
    except _EXPORT_ERRORS as exc:
        raise ExportError(f"ONNX export to {path} failed: {type(exc).__name__}: {exc}") from exc
    try:
        onnx.checker.check_model(str(path))
    except onnx.checker.ValidationError as exc:
        raise ExportError(f"Exported model {path} failed onnx.checker: {exc}") from exc
    logger.info("Exported ONNX opset %d to %s", opset, path)
    return path
