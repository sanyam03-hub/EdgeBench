from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

WNIDS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
IMAGES_PER_CLASS = 3


@pytest.fixture
def sample_results() -> Any:
    """Small valid RunResults (synthetic placeholder values, not measurements)."""
    from edgebench.bench import LatencyStats
    from edgebench.parity import ParityReport
    from edgebench.results import AccuracyResult, Environment, RunResults, VariantResult

    def latency(p50: float, noisy: bool = False) -> dict[str, LatencyStats]:
        def stats(scale: float) -> LatencyStats:
            v = p50 * scale
            return LatencyStats(
                n=10, p50_ms=v, p90_ms=v * 1.2, p99_ms=v * 1.5, mean_ms=v, std_ms=v * 0.05,
                iqr_ms=v * 0.1, cv=0.15 if noisy else 0.05, noisy=noisy,
            )  # fmt: skip

        return {"1": stats(1.0), "8": stats(6.0)}

    def parity(agree: float, max_abs: float) -> ParityReport:
        return ParityReport(
            n_samples=16, max_abs_err=max_abs, mean_abs_err=max_abs / 10, cosine_similarity=0.99,
            top1_agreement=agree, within_tolerance=agree >= 0.9,
            tolerance={"min_top1_agreement": 0.9},
        )  # fmt: skip

    def row(variant: str, runtime: str, precision: str, **kw: Any) -> VariantResult:
        base = dict(
            model="tiny_cnn", variant=variant, runtime=runtime, precision=precision, status="ok",
            error=None, artifact=None, size_mb=None, load_ms=1.5, parity=None, accuracy=None,
            latency=None,
        )  # fmt: skip
        base.update(kw)
        return VariantResult(**base)

    def acc(top1: float, delta: float | None) -> AccuracyResult:
        return AccuracyResult(top1=top1, n_images=64, delta_vs_onnx_fp32_pts=delta)

    variants = (
        row("torch_fp32", "torch", "fp32", accuracy=acc(50.0, None), latency=latency(4.0)),
        row("onnx_fp32", "onnxruntime", "fp32", artifact="artifacts/tiny_cnn/onnx_fp32.onnx",
            size_mb=0.02, parity=parity(1.0, 3.1e-6), accuracy=acc(50.0, 0.0),
            latency=latency(2.0)),
        row("onnx_int8_dynamic", "onnxruntime", "int8_dynamic", status="error",
            error="QuantizationError: boom | bad"),
        row("onnx_int8_static", "onnxruntime", "int8_static",
            artifact="artifacts/tiny_cnn/onnx_int8_static.onnx", size_mb=0.01,
            parity=parity(0.95, 0.2), accuracy=acc(48.4375, -1.5625),
            latency=latency(1.0, noisy=True)),
    )  # fmt: skip
    return RunResults(
        schema_version=1,
        run_id="20260101T000000Z_ab12cd34",
        created_at="2026-01-01T00:00:00Z",
        edgebench_version="0.1.0",
        config_hash="ab12cd34",
        config={
            "model": {
                "model": "tiny_cnn",
                "eval": {"n_images": 64},
                "quantization": {"calibration": {"n_images": 32}},
            },
            "target": {
                "name": "ci",
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "batch_sizes": [1, 8],
                "warmup": 2,
                "runs": 10,
            },
        },  # fmt: skip
        environment=Environment(
            os="Linux",
            kernel="6.0",
            cpu_model="Test CPU",
            logical_cores=4,
            python="3.11.4",
            packages={
                "torch": "2.14.1",
                "torchvision": "0.29.1",
                "onnx": "1.23.2",
                "onnxruntime": "1.31.0",
                "numpy": "2.4.6",
            },
        ),  # fmt: skip
        variants=variants,
    )


@pytest.fixture
def tiny_cnn() -> Any:
    """The seeded random-init tiny CNN (10 classes, 32x32 input); no download."""
    from edgebench.models import load_model

    return load_model("tiny_cnn")


@pytest.fixture
def synthetic_imagenette(tmp_path: Path) -> Path:
    """A fake ``imagenette2-160`` root: train/ and val/ with the 10 real WNID folders.

    Each folder holds a few small random RGB JPEGs (fixed seed) plus one non-image file.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "data" / "imagenette2-160"
    for split in ("train", "val"):
        for wnid in WNIDS:
            folder = root / split / wnid
            folder.mkdir(parents=True)
            for i in range(IMAGES_PER_CLASS):
                pixels = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{split}_{wnid}_{i}.JPEG")
            (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    return root
