from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgebench.backends.ort_backend import OrtBackend
from edgebench.backends.registry import VARIANT_SPECS, create_backend
from edgebench.backends.torch_backend import TorchBackend
from edgebench.config import VARIANTS, TargetConfig
from edgebench.errors import BackendError
from edgebench.export import export_onnx
from edgebench.models import LoadedModel

TARGET = TargetConfig(
    name="test",
    intra_op_threads=2,
    inter_op_threads=1,
    batch_sizes=(1,),
    warmup=0,
    runs=1,
    noisy_cv_threshold=0.1,
)


@pytest.fixture
def onnx_path(tiny_cnn: LoadedModel, tmp_path: Path) -> Path:
    sample = np.zeros((1, *tiny_cnn.input_shape), dtype=np.float32)
    return export_onnx(tiny_cnn.module, sample, tmp_path / "onnx_fp32.onnx")


def _x(n: int = 3) -> np.ndarray:
    return np.random.default_rng(0).random((n, 3, 32, 32), dtype=np.float32)


def test_variant_specs_cover_config_vocabulary() -> None:
    assert set(VARIANT_SPECS) == set(VARIANTS)


def test_torch_and_ort_same_shape_and_dtype(onnx_path: Path) -> None:
    torch_backend, _ = create_backend(
        "torch_fp32", model_name="tiny_cnn", artifact=None, target=TARGET
    )
    ort_backend, _ = create_backend(
        "onnx_fp32", model_name="tiny_cnn", artifact=onnx_path, target=TARGET
    )
    a, b = torch_backend.predict(_x()), ort_backend.predict(_x())
    assert a.shape == b.shape == (3, 10)
    assert a.dtype == b.dtype == np.float32


@pytest.mark.parametrize(
    "variant", ["torch_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"]
)
def test_registry_returns_right_class(variant: str, onnx_path: Path) -> None:
    artifact = None if variant == "torch_fp32" else onnx_path
    backend, load_ms = create_backend(
        variant, model_name="tiny_cnn", artifact=artifact, target=TARGET
    )
    expected = TorchBackend if variant == "torch_fp32" else OrtBackend
    assert isinstance(backend, expected)
    assert (backend.runtime, backend.precision) == VARIANT_SPECS[variant]
    assert backend.variant == variant
    assert isinstance(load_ms, float) and load_ms >= 0.0


def test_ort_session_uses_target_threads(onnx_path: Path) -> None:
    backend, _ = create_backend(
        "onnx_fp32", model_name="tiny_cnn", artifact=onnx_path, target=TARGET
    )
    assert isinstance(backend, OrtBackend)
    options = backend.session_options
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1


@pytest.mark.parametrize(
    ("x", "match"),
    [
        (_x().astype(np.float64), "float32"),
        (np.zeros((3, 32, 32), dtype=np.float32), r"\(N, 3, H, W\)"),
        (np.zeros((2, 1, 32, 32), dtype=np.float32), r"\(N, 3, H, W\)"),
        (np.zeros((0, 3, 32, 32), dtype=np.float32), "empty"),
        ([[1.0]], "numpy"),
    ],
)
@pytest.mark.parametrize("variant", ["torch_fp32", "onnx_fp32"])
def test_predict_rejects_bad_input(variant: str, x: object, match: str, onnx_path: Path) -> None:
    artifact = None if variant == "torch_fp32" else onnx_path
    backend, _ = create_backend(variant, model_name="tiny_cnn", artifact=artifact, target=TARGET)
    with pytest.raises(BackendError, match=match):
        backend.predict(x)  # type: ignore[arg-type]


def test_unknown_variant_raises() -> None:
    with pytest.raises(BackendError, match="onnx_fp16"):
        create_backend("onnx_fp16", model_name="tiny_cnn", artifact=None, target=TARGET)


def test_onnx_variant_without_artifact_raises() -> None:
    with pytest.raises(BackendError, match="artifact"):
        create_backend("onnx_fp32", model_name="tiny_cnn", artifact=None, target=TARGET)


def test_missing_artifact_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="not found"):
        create_backend(
            "onnx_fp32", model_name="tiny_cnn", artifact=tmp_path / "nope.onnx", target=TARGET
        )


def test_corrupt_artifact_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    with pytest.raises(BackendError, match="InvalidProtobuf"):
        create_backend("onnx_fp32", model_name="tiny_cnn", artifact=bad, target=TARGET)


def test_tflite_unavailable_is_backend_error(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="TFLite"):
        create_backend(
            "tflite_fp32", model_name="tiny_cnn", artifact=tmp_path / "m.tflite", target=TARGET
        )
