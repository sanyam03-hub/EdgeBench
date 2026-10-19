"""ONNX FP32 -> INT8 with ONNX Runtime: dynamic (weights only) and static QDQ (calibrated).

Does not benchmark.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_dynamic,
    quantize_static,
)

from edgebench.backends.ort_backend import ORT_ERRORS
from edgebench.errors import QuantizationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

CALIBRATE_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}
QUANT_TYPES = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
# What ORT quantization / onnx loading raise on bad models or failed calibration runs.
_QUANT_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    onnx.checker.ValidationError,
    *ORT_ERRORS,
)


def quantize_dynamic_int8(fp32_path: Path, out_path: Path) -> Path:
    """Quantize weights to int8 offline; activations are quantized at runtime.

    Raises:
        QuantizationError: If the FP32 model is missing or ORT quantization fails.
    """
    _require_file(fp32_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_dynamic(fp32_path, out_path, weight_type=QuantType.QInt8)
    except _QUANT_ERRORS as exc:
        raise _error("Dynamic", fp32_path, exc) from exc
    logger.info("Dynamic INT8 model written to %s", out_path)
    return out_path


def name_unnamed_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give every unnamed node the deterministic name ``<op_type>:<first output>``, in place.

    ORT's ``nodes_to_exclude`` matches node names, so an unnamed node (e.g. the HardSigmoid/Mul
    pairs of MobileNetV3's Hardswish) could otherwise never be kept in FP32.
    """
    taken = {node.name for node in model.graph.node if node.name}
    for node in model.graph.node:
        if node.name:
            continue
        base = f"{node.op_type}:{node.output[0]}"
        name, suffix = base, 1
        while name in taken:
            name, suffix = f"{base}_{suffix}", suffix + 1
        node.name = name
        taken.add(name)
    return model


def preprocess(fp32_path: Path, out_path: Path) -> Path:
    """ORT quantization pre-processing (shape inference + graph optimization), then name all nodes.

    Deterministic: the same FP32 file always gives the same node names, so names found by the
    per-layer analysis can be passed back as ``nodes_to_exclude``.

    Raises:
        QuantizationError: Missing/invalid FP32 model or pre-processing failed.
    """
    _require_file(fp32_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quant_pre_process(fp32_path, out_path)
        onnx.save(name_unnamed_nodes(onnx.load(str(out_path))), str(out_path))
    except _QUANT_ERRORS as exc:
        raise _error("Pre-processing for", fp32_path, exc) from exc
    return out_path


def quantize_static_int8(
    fp32_path: Path,
    out_path: Path,
    calibration_reader: CalibrationDataReader,
    *,
    per_channel: bool = True,
    calibrate_method: str = "minmax",
    activation_type: str = "quint8",
    weight_type: str = "qint8",
    nodes_to_exclude: Sequence[str] = (),
) -> Path:
    """Pre-process, calibrate and quantize to a QDQ INT8 model.

    Args:
        fp32_path: Exported FP32 ONNX model.
        out_path: Where the QDQ model is written.
        calibration_reader: Yields ``{input_name: batch}`` from calibration images (train/ only).
        per_channel: One weight scale per output channel instead of one per tensor.
        calibrate_method: ``minmax`` | ``entropy`` | ``percentile``.
        activation_type: ``quint8`` | ``qint8``.
        weight_type: ``quint8`` | ``qint8``.
        nodes_to_exclude: Node names (as named by ``preprocess``) to keep in FP32.

    Raises:
        QuantizationError: Unknown option or node name, missing model, or ORT
            pre-processing/quantization failed.
    """
    method = _lookup(CALIBRATE_METHODS, calibrate_method, "calibrate_method")
    act_type = _lookup(QUANT_TYPES, activation_type, "activation_type")
    w_type = _lookup(QUANT_TYPES, weight_type, "weight_type")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edgebench_quant_", ignore_cleanup_errors=True) as tmp:
        preprocessed = preprocess(fp32_path, Path(tmp) / "preprocessed.onnx")
        # ORT silently ignores unknown names, which would quietly quantize everything.
        known = {node.name for node in onnx.load(str(preprocessed)).graph.node}
        unknown = sorted(set(nodes_to_exclude) - known)
        if unknown:
            raise QuantizationError(f"nodes_to_exclude not found in {fp32_path}: {unknown}")
        try:
            quantize_static(
                preprocessed,
                out_path,
                calibration_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                activation_type=act_type,
                weight_type=w_type,
                calibrate_method=method,
                nodes_to_exclude=list(nodes_to_exclude),
            )
        except _QUANT_ERRORS as exc:
            raise _error("Static", fp32_path, exc) from exc
    logger.info("Static QDQ INT8 model written to %s", out_path)
    return out_path


def _lookup(table: dict[str, T], value: str, field: str) -> T:
    if value not in table:
        raise QuantizationError(f"Unknown {field} {value!r}; allowed: {list(table)}")
    return table[value]


def _require_file(path: Path) -> None:
    """Fail early, with the path in the message, if the FP32 model is missing or not valid ONNX."""
    if not path.is_file():
        raise QuantizationError(f"FP32 model not found: {path}. Export it first (onnx_fp32).")
    try:
        onnx.checker.check_model(str(path))
    except onnx.checker.ValidationError as exc:
        raise QuantizationError(f"FP32 model {path} is not a valid ONNX model: {exc}") from exc


def _error(kind: str, fp32_path: Path, exc: Exception) -> QuantizationError:
    return QuantizationError(
        f"{kind} INT8 quantization of {fp32_path} failed: {type(exc).__name__}: {exc}"
    )
