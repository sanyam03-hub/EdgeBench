from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgebench.bench import LatencyStats
from edgebench.errors import SchemaError
from edgebench.parity import ParityReport
from edgebench.results import (
    SCHEMA_VERSION,
    AccuracyResult,
    Environment,
    RunResults,
    VariantResult,
    load_results,
    save_results,
)

# Synthetic placeholder values for schema tests only; not measurements.
_LAT = LatencyStats(
    n=3,
    p50_ms=1.0,
    p90_ms=1.5,
    p99_ms=1.9,
    mean_ms=1.1,
    std_ms=0.1,
    iqr_ms=0.2,
    cv=0.09,
    noisy=False,
)


def _sample_results() -> RunResults:
    ok = VariantResult(
        model="tiny_cnn",
        variant="onnx_int8_static",
        runtime="onnxruntime",
        precision="int8_static",
        status="ok",
        error=None,
        artifact="artifacts/tiny_cnn/onnx_int8_static.onnx",
        size_mb=0.5,
        load_ms=2.0,
        parity=ParityReport(
            n_samples=4,
            max_abs_err=0.1,
            mean_abs_err=0.01,
            cosine_similarity=0.99,
            top1_agreement=1.0,
            within_tolerance=True,
            tolerance={"min_top1_agreement": 0.9},
        ),
        accuracy=AccuracyResult(top1=50.0, n_images=8, delta_vs_onnx_fp32_pts=-0.5),
        latency={"1": _LAT, "8": _LAT},
    )
    failed = VariantResult(
        model="tiny_cnn",
        variant="onnx_int8_dynamic",
        runtime="onnxruntime",
        precision="int8_dynamic",
        status="error",
        error="QuantizationError: boom",
        artifact=None,
        size_mb=None,
        load_ms=None,
        parity=None,
        accuracy=None,
        latency=None,
    )
    return RunResults(
        schema_version=SCHEMA_VERSION,
        run_id="20260101T120000Z_ab12cd34",
        created_at="2026-01-01T12:00:00Z",
        edgebench_version="0.1.0",
        config_hash="ab12cd34",
        config={"model": {"model": "tiny_cnn"}, "target": {"name": "ci"}},
        environment=Environment(
            os="Windows",
            kernel="10.0.26200",
            cpu_model="test-cpu",
            logical_cores=4,
            python="3.11.4",
            packages={"numpy": "x.y.z"},
        ),
        variants=(ok, failed),
    )


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_round_trip(tmp_path: Path) -> None:
    original = _sample_results()
    path = tmp_path / "run.json"
    save_results(original, path)
    assert load_results(path) == original


def test_to_dict_matches_schema_field_names() -> None:
    data = _sample_results().to_dict()
    assert data["schema_version"] == 1
    row = data["variants"][0]
    assert set(row) == {
        "model", "variant", "runtime", "precision", "status", "error", "artifact",
        "size_mb", "load_ms", "parity", "accuracy", "latency",
    }  # fmt: skip
    assert row["latency"]["1"]["p50_ms"] == 1.0
    assert row["accuracy"]["delta_vs_onnx_fp32_pts"] == -0.5
    assert data["variants"][1]["parity"] is None


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    data = _sample_results().to_dict()
    data["schema_version"] = 2
    with pytest.raises(SchemaError, match=r"expected 1.*found 2"):
        load_results(_write_json(tmp_path / "run.json", data))


def test_missing_top_level_field_named(tmp_path: Path) -> None:
    data = _sample_results().to_dict()
    del data["run_id"]
    with pytest.raises(SchemaError, match="run_id"):
        load_results(_write_json(tmp_path / "run.json", data))


def test_missing_nested_field_named(tmp_path: Path) -> None:
    data = _sample_results().to_dict()
    del data["variants"][0]["latency"]["8"]["p99_ms"]
    with pytest.raises(SchemaError, match=r"variants\[0\]\.latency\.8\.p99_ms"):
        load_results(_write_json(tmp_path / "run.json", data))


def test_unknown_status_rejected(tmp_path: Path) -> None:
    data = _sample_results().to_dict()
    data["variants"][0]["status"] = "maybe"
    with pytest.raises(SchemaError, match="status"):
        load_results(_write_json(tmp_path / "run.json", data))


def test_invalid_json_raises_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaError, match="run.json"):
        load_results(path)


def test_file_is_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    save_results(_sample_results(), path)
    text = path.read_bytes().decode("utf-8")
    expected = json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert text == expected
    assert '\n  "config"' in text
