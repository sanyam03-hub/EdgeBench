"""ONNX FP32 -> INT8 with ONNX Runtime: dynamic (weights only) and static QDQ (calibrated).

Does not benchmark.
"""

from __future__ import annotations

import logging
import tempfile
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


def quantize_static_int8(
    fp32_path: Path,
    out_path: Path,
    calibration_reader: CalibrationDataReader,
    *,
    per_channel: bool = True,
    calibrate_method: str = "minmax",
    activation_type: str = "quint8",
    weight_type: str = "qint8",
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

    Raises:
        QuantizationError: Unknown option, missing model, or ORT pre-processing/quantization failed.
    """
    method = _lookup(CALIBRATE_METHODS, calibrate_method, "calibrate_method")
    act_type = _lookup(QUANT_TYPES, activation_type, "activation_type")
    w_type = _lookup(QUANT_TYPES, weight_type, "weight_type")
    _require_file(fp32_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edgebench_quant_", ignore_cleanup_errors=True) as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        try:
            quant_pre_process(fp32_path, preprocessed)
            quantize_static(
                preprocessed,
                out_path,
                calibration_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                activation_type=act_type,
                weight_type=w_type,
                calibrate_method=method,
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
