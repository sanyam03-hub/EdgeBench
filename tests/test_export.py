from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from edgebench.errors import ExportError
from edgebench.export import INPUT_NAME, OUTPUT_NAME, export_onnx
from edgebench.models import LoadedModel
from edgebench.parity import compare

FP32_TOL = {"max_abs_err": 1e-3, "min_cosine": 0.9999, "min_top1_agreement": 0.99}


def _batch(model: LoadedModel, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, *model.input_shape), dtype=np.float32)


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_export_writes_valid_onnx(tiny_cnn: LoadedModel, tmp_path: Path) -> None:
    path = export_onnx(tiny_cnn.module, _batch(tiny_cnn, 1), tmp_path / "m" / "onnx_fp32.onnx")
    assert path.is_file()
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    assert [i.name for i in proto.graph.input] == [INPUT_NAME]
    assert [o.name for o in proto.graph.output] == [OUTPUT_NAME]
    assert {op.domain: op.version for op in proto.opset_import}[""] == 17


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dynamic_batch_and_fp32_parity(
    tiny_cnn: LoadedModel, tmp_path: Path, batch_size: int
) -> None:
    path = export_onnx(tiny_cnn.module, _batch(tiny_cnn, 1), tmp_path / "m.onnx")
    x = _batch(tiny_cnn, batch_size, seed=1)
    ort_out = _session(path).run(None, {INPUT_NAME: x})[0]
    with torch.inference_mode():
        torch_out = tiny_cnn.module(torch.from_numpy(x)).numpy()
    assert ort_out.shape == (batch_size, 10)
    assert compare(torch_out, ort_out, FP32_TOL).within_tolerance


def test_export_failure_raises_export_error(
    tiny_cnn: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unsupported operator")

    monkeypatch.setattr(torch.onnx, "export", boom)
    with pytest.raises(ExportError, match="RuntimeError: unsupported operator"):
        export_onnx(tiny_cnn.module, _batch(tiny_cnn, 1), tmp_path / "m.onnx")


def test_checker_failure_raises_export_error(
    tiny_cnn: LoadedModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_model(*args: object, **kwargs: object) -> None:
        raise onnx.checker.ValidationError("bad graph")

    monkeypatch.setattr(onnx.checker, "check_model", bad_model)
    with pytest.raises(ExportError, match="bad graph"):
        export_onnx(tiny_cnn.module, _batch(tiny_cnn, 1), tmp_path / "m.onnx")
