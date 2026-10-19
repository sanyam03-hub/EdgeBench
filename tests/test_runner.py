from __future__ import annotations

import dataclasses
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edgebench import runner
from edgebench.config import RunConfig, load_config
from edgebench.errors import DatasetError, ExportError, QuantizationError
from edgebench.results import RunResults, load_results
from edgebench.runner import RunPaths, config_hash, config_snapshot, execute_run, make_run_id

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs" / "smoke_tiny.yaml"
CI_TARGET = REPO_ROOT / "configs" / "targets" / "ci.yaml"


@pytest.fixture
def smoke_run() -> RunConfig:
    return load_config(SMOKE, CI_TARGET)


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return RunPaths(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        results_dir=tmp_path / "results",
        project_root=tmp_path,
    )


def _rows(results: RunResults) -> dict[str, object]:
    return {row.variant: row for row in results.variants}


def test_run_id_format_and_hash(smoke_run: RunConfig) -> None:
    snapshot = config_snapshot(smoke_run)
    digest = config_hash(snapshot)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert config_hash(config_snapshot(smoke_run)) == digest
    other = dataclasses.replace(smoke_run.target, runs=11)
    assert config_hash(config_snapshot(dataclasses.replace(smoke_run, target=other))) != digest
    run_id = make_run_id(digest, datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert run_id == f"20260102T030405Z_{digest[:8]}"


def test_smoke_run_all_variants_ok(smoke_run: RunConfig, paths: RunPaths) -> None:
    messages: list[str] = []
    results, path = execute_run(smoke_run, paths, progress=messages.append)
    assert path.parent == paths.results_dir and path.name == f"{results.run_id}.json"
    assert load_results(path) == results
    rows = _rows(results)
    assert list(rows) == list(smoke_run.model.variants)
    for variant, row in rows.items():
        assert row.status == "ok", (variant, row.error)
        assert row.accuracy is not None and row.accuracy.n_images == 64
        assert set(row.latency) == {"1"} and row.latency["1"].n == smoke_run.target.runs
        assert row.load_ms is not None and row.load_ms >= 0.0
    assert rows["torch_fp32"].parity is None
    assert rows["torch_fp32"].artifact is None
    assert rows["torch_fp32"].accuracy.delta_vs_onnx_fp32_pts is None
    assert rows["onnx_fp32"].accuracy.delta_vs_onnx_fp32_pts == 0.0
    assert rows["onnx_int8_static"].artifact == "artifacts/tiny_cnn/onnx_int8_static.onnx"
    assert rows["onnx_int8_static"].parity.tolerance == smoke_run.model.parity.int8
    assert rows["onnx_fp32"].parity.tolerance == smoke_run.model.parity.fp32
    assert results.environment.cpu_model
    assert messages, "progress lines expected"


def test_quantization_failure_is_isolated(
    smoke_run: RunConfig, paths: RunPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise QuantizationError("calibration exploded")

    monkeypatch.setattr(runner, "quantize_static_int8", boom)
    rows = _rows(execute_run(smoke_run, paths)[0])
    failed = rows["onnx_int8_static"]
    assert failed.status == "error"
    assert failed.error == "QuantizationError: calibration exploded"
    assert failed.parity is None and failed.accuracy is None and failed.latency is None
    for variant in ("torch_fp32", "onnx_fp32", "onnx_int8_dynamic"):
        assert rows[variant].status == "ok"


def test_export_failure_skips_dependent_variants(
    smoke_run: RunConfig, paths: RunPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise ExportError("unsupported op")

    monkeypatch.setattr(runner, "export_onnx", boom)
    rows = _rows(execute_run(smoke_run, paths)[0])
    assert rows["torch_fp32"].status == "ok"
    assert rows["onnx_fp32"].status == "error"
    assert rows["onnx_fp32"].error == "ExportError: unsupported op"
    for variant in ("onnx_int8_dynamic", "onnx_int8_static"):
        assert rows[variant].status == "skipped"
        assert "onnx_fp32" in rows[variant].error


def test_export_reused_unless_forced(smoke_run: RunConfig, paths: RunPaths) -> None:
    execute_run(smoke_run, paths)
    exported = paths.artifacts_dir / "tiny_cnn" / "onnx_fp32.onnx"
    first = exported.stat().st_mtime_ns
    execute_run(smoke_run, paths)
    assert exported.stat().st_mtime_ns == first
    execute_run(smoke_run, paths, force=True)
    assert exported.stat().st_mtime_ns != first


def test_missing_dataset_fails_fast(paths: RunPaths) -> None:
    run = load_config(REPO_ROOT / "configs" / "mobilenet_v3_small.yaml", CI_TARGET)
    with pytest.raises(DatasetError, match="imagenette2-160"):
        execute_run(run, paths)
    assert not paths.results_dir.exists()
