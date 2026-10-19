from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from edgebench.backends.ort_backend import OrtBackend
from edgebench.data import ImagenetteCalibrationReader, list_images
from edgebench.errors import QuantizationError
from edgebench.export import INPUT_NAME, export_onnx
from edgebench.models import LoadedModel
from edgebench.quantize import quantize_dynamic_int8, quantize_static_int8

N_CALIB_IMAGES = 8
CALIB_BATCH = 4


@pytest.fixture
def fp32_path(tiny_cnn: LoadedModel, tmp_path: Path) -> Path:
    sample = np.zeros((1, *tiny_cnn.input_shape), dtype=np.float32)
    return export_onnx(tiny_cnn.module, sample, tmp_path / "fp32" / "onnx_fp32.onnx")


@pytest.fixture
def reader(tiny_cnn: LoadedModel, synthetic_imagenette: Path) -> ImagenetteCalibrationReader:
    paths = list_images(synthetic_imagenette / "train")[:N_CALIB_IMAGES]
    return ImagenetteCalibrationReader(INPUT_NAME, paths, tiny_cnn.transforms, CALIB_BATCH)


def _op_types(path: Path) -> set[str]:
    return {node.op_type for node in onnx.load(str(path)).graph.node}


def _run(path: Path, variant: str, precision: str) -> np.ndarray:
    backend = OrtBackend(
        path, variant=variant, precision=precision, intra_op_threads=1, inter_op_threads=1
    )
    return backend.predict(np.random.default_rng(0).random((2, 3, 32, 32), dtype=np.float32))


def _weight_scale_sizes(path: Path) -> list[int]:
    """Sizes of the scale tensors of DequantizeLinear nodes that dequantize a stored weight."""
    graph = onnx.load(str(path)).graph
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    return [
        inits[node.input[1]].size
        for node in graph.node
        if node.op_type == "DequantizeLinear"
        and node.input[0] in inits
        and inits[node.input[0]].ndim == 4  # conv weight (out, in, kh, kw)
    ]


def test_dynamic_int8(fp32_path: Path, tmp_path: Path) -> None:
    out = quantize_dynamic_int8(fp32_path, tmp_path / "out" / "onnx_int8_dynamic.onnx")
    assert out.is_file()
    assert out.stat().st_size < fp32_path.stat().st_size
    int8_inits = [
        i for i in onnx.load(str(out)).graph.initializer if i.data_type == onnx.TensorProto.INT8
    ]
    assert int8_inits, "expected int8 weight initializers"
    assert _run(out, "onnx_int8_dynamic", "int8_dynamic").shape == (2, 10)


def test_static_int8_qdq(
    fp32_path: Path, tmp_path: Path, reader: ImagenetteCalibrationReader
) -> None:
    out_dir = tmp_path / "out"
    out = quantize_static_int8(fp32_path, out_dir / "onnx_int8_static.onnx", reader)
    assert out.is_file()
    assert out.stat().st_size < fp32_path.stat().st_size
    assert {"QuantizeLinear", "DequantizeLinear"} <= _op_types(out)
    assert _run(out, "onnx_int8_static", "int8_static").shape == (2, 10)
    assert reader.get_next() is None, "calibration should consume every batch"
    assert sorted(p.name for p in out_dir.iterdir()) == ["onnx_int8_static.onnx"]


@pytest.mark.parametrize(("per_channel", "expect_channelwise"), [(True, True), (False, False)])
def test_static_per_channel_weight_scales(
    fp32_path: Path,
    tmp_path: Path,
    reader: ImagenetteCalibrationReader,
    per_channel: bool,
    expect_channelwise: bool,
) -> None:
    out = quantize_static_int8(fp32_path, tmp_path / "q.onnx", reader, per_channel=per_channel)
    sizes = _weight_scale_sizes(out)
    assert sizes, "expected dequantized conv weights"
    assert all((size > 1) == expect_channelwise for size in sizes), sizes


def test_static_rejects_unknown_calibrate_method(
    fp32_path: Path, tmp_path: Path, reader: ImagenetteCalibrationReader
) -> None:
    with pytest.raises(QuantizationError, match="magic"):
        quantize_static_int8(fp32_path, tmp_path / "q.onnx", reader, calibrate_method="magic")


@pytest.mark.parametrize("which", ["dynamic", "static"])
def test_missing_fp32_model_raises(
    which: str, tmp_path: Path, reader: ImagenetteCalibrationReader
) -> None:
    missing = tmp_path / "nope.onnx"
    with pytest.raises(QuantizationError, match="not found"):
        if which == "dynamic":
            quantize_dynamic_int8(missing, tmp_path / "q.onnx")
        else:
            quantize_static_int8(missing, tmp_path / "q.onnx", reader)


@pytest.mark.parametrize("which", ["dynamic", "static"])
def test_corrupt_fp32_model_raises(
    which: str, tmp_path: Path, reader: ImagenetteCalibrationReader
) -> None:
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    with pytest.raises(QuantizationError, match="bad.onnx"):
        if which == "dynamic":
            quantize_dynamic_int8(bad, tmp_path / "q.onnx")
        else:
            quantize_static_int8(bad, tmp_path / "q.onnx", reader)
