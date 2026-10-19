"""End-to-end S1 study on tiny_cnn + synthetic Imagenette (mechanics only: random model)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from edgebench.config import RunConfig, SubsetConfig, load_config
from edgebench.export import export_onnx
from edgebench.models import LoadedModel
from edgebench.s1_study import StudySettings, run_study, short_name, write_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = StudySettings(ks=(0, 1, 2), random_seeds=2, analysis_images=4, analysis_batch=2)
METHODS = ["local_sqnr", "sensitivity"]


def _run(n_eval: int, n_calib: int) -> RunConfig:
    """smoke_tiny + ci target, pointed at the (synthetic) Imagenette folder layout."""
    run = load_config(REPO_ROOT / "configs/smoke_tiny.yaml", REPO_ROOT / "configs/targets/ci.yaml")
    quant = dataclasses.replace(
        run.model.quantization, calibration=SubsetConfig(n_images=n_calib, batch_size=4, seed=0)
    )
    model = dataclasses.replace(
        run.model, dataset="imagenette2-160", quantization=quant,
        eval=SubsetConfig(n_images=n_eval, batch_size=4, seed=0),
    )  # fmt: skip
    return dataclasses.replace(run, model=model)


def _fp32(tiny_cnn: LoadedModel, tmp_path: Path) -> Path:
    sample = np.zeros((1, *tiny_cnn.input_shape), dtype=np.float32)
    return export_onnx(tiny_cnn.module, sample, tmp_path / "onnx_fp32.onnx")


@pytest.fixture
def study(tiny_cnn: LoadedModel, synthetic_imagenette: Path, tmp_path: Path) -> dict:
    fp32 = _fp32(tiny_cnn, tmp_path)
    return run_study(_run(20, 8), synthetic_imagenette, fp32, tmp_path / "work", SETTINGS)


def test_rankings(study: dict) -> None:
    assert list(study["rankings"]) == METHODS
    for ranking in study["rankings"].values():
        scores = [r["sqnr_db"] for r in ranking]
        assert scores == sorted(scores) and len(scores) <= study["n_eligible_nodes"]
    # Sensitivity scores every quantized node, from the logits.
    sensitivity = study["rankings"]["sensitivity"]
    assert len(sensitivity) == study["n_eligible_nodes"]
    assert {r["source"] for r in sensitivity} == {"logits"}


def test_sweep_and_report(study: dict) -> None:
    assert study["settings"]["n_search"] == 10 and study["settings"]["n_report"] == 10
    sweep = study["sweep"]
    assert {m: list(c) for m, c in sweep["ranked"].items()} == dict.fromkeys(
        METHODS, ["0", "1", "2"]
    )
    assert list(sweep["random"]) == ["1", "2"]
    assert all(len(v) == 2 for v in sweep["random"].values())
    for method in METHODS:
        k = study["chosen_k"][method]
        assert k in (0, 1, 2)
        expected = [r["node"] for r in study["rankings"][method][:k]]
        assert study["excluded_nodes"][method] == expected
    report = study["report"]
    rows = [report["fp32"], report["int8"], *report["mixed"].values()]
    for row in rows:
        assert 0 <= row["top1"] <= 100 and row["size_mb"] > 0
        assert row["latency_b1"]["n"] == 10  # ci target: 10 timed runs
    assert report["int8"]["size_mb"] < report["fp32"]["size_mb"]
    assert all(len(runs) == 2 for runs in report["random_mixed"].values())


def test_outputs_written(study: dict, tmp_path: Path) -> None:
    paths = write_outputs(study, tmp_path / "out")
    assert sorted(p.name for p in paths) == [
        "ranking_local_sqnr.png", "ranking_sensitivity.png", "recovery.png", "report.md",
        "s1.json", "sqnr_depth.png",
    ]  # fmt: skip
    assert all(p.stat().st_size > 0 for p in paths)
    assert json.loads((tmp_path / "out" / "s1.json").read_text(encoding="utf-8")) == study
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Report half" in text and "Random baseline" in text and "sensitive" in text


def test_rankings_only(tiny_cnn: LoadedModel, synthetic_imagenette: Path, tmp_path: Path) -> None:
    fp32 = _fp32(tiny_cnn, tmp_path)
    settings = dataclasses.replace(SETTINGS, sweep=False, sensitivity=False)
    result = run_study(_run(10, 4), synthetic_imagenette, fp32, tmp_path / "w", settings)
    assert list(result["rankings"]) == ["local_sqnr"] and result["rankings"]["local_sqnr"]
    assert result["sweep"] is None and result["report"] is None
    names = sorted(p.name for p in write_outputs(result, tmp_path / "out"))
    assert names == ["ranking_local_sqnr.png", "report.md", "s1.json", "sqnr_depth.png"]


def test_short_name() -> None:
    conv = "/features/features.1/block/block.0/block.0.0/Conv"
    assert short_name(conv) == "features.1/block.0.0/Conv"
    mul = "Mul:/features/features.4/block/block.0/Mul_output_0"
    assert short_name(mul) == "Mul:features.4/block.0/Mul_output_0"
    assert short_name("/avgpool/GlobalAveragePool") == "avgpool/GlobalAveragePool"
    assert short_name("HardSigmoid:_inlfunc_X") == "HardSigmoid:_inlfunc_X"
