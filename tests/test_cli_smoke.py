from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from edgebench.cli import main
from edgebench.results import load_results

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(REPO_ROOT / "configs" / "smoke_tiny.yaml")
CI_TARGET = str(REPO_ROOT / "configs" / "targets" / "ci.yaml")


@pytest.fixture(scope="module")
def smoke_results_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One smoke run shared by the report/gate tests (results JSON path)."""
    from edgebench.config import load_config
    from edgebench.runner import RunPaths, execute_run

    root = tmp_path_factory.mktemp("smoke")
    paths = RunPaths(root / "data", root / "artifacts", root / "results", root)
    _, path = execute_run(load_config(Path(SMOKE), Path(CI_TARGET)), paths)
    return path


def test_report_creates_files(smoke_results_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "--results", str(smoke_results_path)]) == 0
    out_dir = smoke_results_path.with_suffix("")
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "pareto.png").stat().st_size > 0
    assert str(out_dir / "report.md") in capsys.readouterr().out


def test_gate_passes_against_baseline_from_same_run(
    smoke_results_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "ci.json"
    args = ["gate", "--results", str(smoke_results_path), "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    assert baseline.is_file()
    capsys.readouterr()
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "GATE PASSED" in out
    assert "FAIL" not in out.replace("GATE PASSED", "")


def test_gate_fails_on_impossible_baseline(
    smoke_results_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "impossible.json"
    args = ["gate", "--results", str(smoke_results_path), "--baseline", str(baseline)]
    assert main([*args, "--write-baseline"]) == 0
    data = json.loads(baseline.read_text(encoding="utf-8"))
    data["entries"][0]["top1"] = 101.0
    baseline.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert main(args) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "GATE FAILED (1 checks)" in out


def test_gate_missing_baseline_exits_2(
    smoke_results_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "nope.json")
    assert main(["gate", "--results", str(smoke_results_path), "--baseline", missing]) == 2
    assert capsys.readouterr().err.startswith("ERROR:")


def test_report_bad_results_path_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "--results", "nope.json"]) == 2
    assert capsys.readouterr().err.startswith("ERROR:")


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    out = {name: tmp_path / name for name in ("data", "artifacts", "results")}
    monkeypatch.setenv("EDGEBENCH_DATA_DIR", str(out["data"]))
    monkeypatch.setenv("EDGEBENCH_ARTIFACTS_DIR", str(out["artifacts"]))
    monkeypatch.setenv("EDGEBENCH_RESULTS_DIR", str(out["results"]))
    return out


def test_help_via_module_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "edgebench.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "run" in proc.stdout


def test_run_smoke_writes_valid_results(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--config", SMOKE, "--target", CI_TARGET]) == 0
    files = list(dirs["results"].glob("*.json"))
    assert len(files) == 1
    results = load_results(files[0])
    assert {row.status for row in results.variants} == {"ok"}
    out = capsys.readouterr().out
    assert str(files[0]) in out
    for variant in ("torch_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"):
        assert variant in out


def test_bad_config_path_exits_2(dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", "nope.yaml", "--target", CI_TARGET]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:")
    assert "nope.yaml" in err


def test_missing_dataset_exits_2_with_download_command(
    dirs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config = str(REPO_ROOT / "configs" / "mobilenet_v3_small.yaml")
    assert main(["run", "--config", config, "--target", CI_TARGET]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:")
    assert "imagenette2-160.tgz" in err
    assert not dirs["results"].exists()


def test_run_requires_config_and_target(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run"])
    assert exc_info.value.code == 2
