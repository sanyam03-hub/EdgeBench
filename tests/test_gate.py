from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from edgebench.bench import LatencyStats
from edgebench.errors import SchemaError
from edgebench.gate import (
    Baseline,
    BaselineEntry,
    baseline_from_results,
    evaluate_gate,
    gate_passed,
    load_baseline,
    save_baseline,
)
from edgebench.parity import ParityReport
from edgebench.results import AccuracyResult, Environment, RunResults, VariantResult

# Synthetic placeholder values for gate tests only; not measurements.


def _latency(p50: float) -> dict[str, LatencyStats]:
    stats = LatencyStats(
        n=10, p50_ms=p50, p90_ms=p50, p99_ms=p50, mean_ms=p50, std_ms=0.0, iqr_ms=0.0, cv=0.0,
        noisy=False,
    )  # fmt: skip
    return {"1": stats}


def _row(
    variant: str, top1: float, *, parity_ok: bool | None = True, p50: float = 1.0
) -> VariantResult:
    parity = None
    if parity_ok is not None:
        parity = ParityReport(
            n_samples=16, max_abs_err=0.0, mean_abs_err=0.0, cosine_similarity=1.0,
            top1_agreement=1.0 if parity_ok else 0.5, within_tolerance=parity_ok,
            tolerance={"min_top1_agreement": 0.9},
        )  # fmt: skip
    return VariantResult(
        model="tiny_cnn", variant=variant, runtime="onnxruntime", precision="fp32", status="ok",
        error=None, artifact=f"artifacts/tiny_cnn/{variant}.onnx", size_mb=0.02, load_ms=1.0,
        parity=parity, accuracy=AccuracyResult(top1=top1, n_images=64, delta_vs_onnx_fp32_pts=0.0),
        latency=_latency(p50),
    )  # fmt: skip


def _results(*rows: VariantResult, target: str = "ci") -> RunResults:
    return RunResults(
        schema_version=1, run_id="20260101T000000Z_ab12cd34", created_at="2026-01-01T00:00:00Z",
        edgebench_version="0.1.0", config_hash="ab12cd34",
        config={"model": {"model": "tiny_cnn"}, "target": {"name": target}},
        environment=Environment(
            os="Linux", kernel="x", cpu_model="test", logical_cores=2, python="3.11",
            packages={},
        ),
        variants=tuple(rows),
    )  # fmt: skip


def _baseline(max_drop: float = 0.5) -> Baseline:
    return Baseline(
        schema_version=1,
        target="ci",
        created_from_run_id="20260101T000000Z_ab12cd34",
        created_at="2026-01-01T00:00:00Z",
        max_accuracy_drop_pts=max_drop,
        entries=(
            BaselineEntry(model="tiny_cnn", variant="onnx_fp32", top1=50.0, require_parity=True),
            BaselineEntry(
                model="tiny_cnn", variant="onnx_int8_static", top1=50.0, require_parity=True
            ),
        ),
    )


def _failed(checks: list) -> list[tuple[str, str]]:
    return [(c.variant, c.check) for c in checks if not c.passed]


def test_all_checks_pass() -> None:
    results = _results(_row("onnx_fp32", 50.0), _row("onnx_int8_static", 50.0))
    checks = evaluate_gate(results, _baseline())
    assert gate_passed(checks)
    kinds = [(c.variant, c.check) for c in checks]
    assert kinds == [
        ("onnx_fp32", "present"), ("onnx_fp32", "status"), ("onnx_fp32", "accuracy"),
        ("onnx_fp32", "parity"), ("onnx_fp32", "latency_info"),
        ("onnx_int8_static", "present"), ("onnx_int8_static", "status"),
        ("onnx_int8_static", "accuracy"), ("onnx_int8_static", "parity"),
        ("onnx_int8_static", "latency_info"),
    ]  # fmt: skip


def test_accuracy_drop_above_threshold_fails() -> None:
    results = _results(_row("onnx_fp32", 50.0), _row("onnx_int8_static", 49.4))
    checks = evaluate_gate(results, _baseline(max_drop=0.5))
    assert not gate_passed(checks)
    assert _failed(checks) == [("onnx_int8_static", "accuracy")]


def test_accuracy_drop_exactly_at_threshold_passes() -> None:
    results = _results(_row("onnx_fp32", 50.0), _row("onnx_int8_static", 49.5))
    assert gate_passed(evaluate_gate(results, _baseline(max_drop=0.5)))


def test_accuracy_improvement_passes() -> None:
    results = _results(_row("onnx_fp32", 60.0), _row("onnx_int8_static", 55.0))
    assert gate_passed(evaluate_gate(results, _baseline()))


def test_parity_outside_tolerance_fails() -> None:
    results = _results(_row("onnx_fp32", 50.0), _row("onnx_int8_static", 50.0, parity_ok=False))
    checks = evaluate_gate(results, _baseline())
    assert _failed(checks) == [("onnx_int8_static", "parity")]


def test_parity_not_required_is_not_checked() -> None:
    baseline = dataclasses.replace(
        _baseline(),
        entries=(BaselineEntry("tiny_cnn", "onnx_fp32", top1=50.0, require_parity=False),),
    )
    results = _results(_row("onnx_fp32", 50.0, parity_ok=False))
    checks = evaluate_gate(results, baseline)
    assert gate_passed(checks)
    assert "parity" not in [c.check for c in checks]


def test_missing_parity_when_required_fails() -> None:
    results = _results(_row("onnx_fp32", 50.0, parity_ok=None), _row("onnx_int8_static", 50.0))
    assert _failed(evaluate_gate(results, _baseline())) == [("onnx_fp32", "parity")]


def test_missing_variant_fails_and_stops_its_chain() -> None:
    results = _results(_row("onnx_fp32", 50.0))
    checks = evaluate_gate(results, _baseline())
    assert _failed(checks) == [("onnx_int8_static", "present")]
    assert [c.check for c in checks if c.variant == "onnx_int8_static"] == ["present"]


def test_errored_variant_fails() -> None:
    broken = dataclasses.replace(
        _row("onnx_int8_static", 50.0), status="error", error="QuantizationError: x",
        parity=None, accuracy=None, latency=None,
    )  # fmt: skip
    checks = evaluate_gate(_results(_row("onnx_fp32", 50.0), broken), _baseline())
    assert _failed(checks) == [("onnx_int8_static", "status")]
    assert "QuantizationError" in next(c.detail for c in checks if not c.passed)


def test_latency_ten_times_worse_still_passes_but_is_reported() -> None:
    results = _results(_row("onnx_fp32", 50.0, p50=10.0), _row("onnx_int8_static", 50.0, p50=10.0))
    checks = evaluate_gate(results, _baseline())
    assert gate_passed(checks)
    info = [c for c in checks if c.check == "latency_info"]
    assert info and all(c.passed for c in info)
    assert "10.00 ms" in info[0].detail


def test_target_mismatch_raises_schema_error() -> None:
    results = _results(_row("onnx_fp32", 50.0), _row("onnx_int8_static", 50.0), target="cpu_4t")
    with pytest.raises(SchemaError, match="cpu_4t"):
        evaluate_gate(results, _baseline())


def test_baseline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "baselines" / "ci.json"
    save_baseline(_baseline(), path)
    assert load_baseline(path) == _baseline()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["thresholds"] == {"max_accuracy_drop_pts": 0.5}
    assert data["entries"][0] == {
        "model": "tiny_cnn", "variant": "onnx_fp32", "top1": 50.0, "require_parity": True,
    }  # fmt: skip


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.pop("target"), "target"),
        (lambda d: d["thresholds"].pop("max_accuracy_drop_pts"), "max_accuracy_drop_pts"),
        (lambda d: d["entries"][0].pop("top1"), "top1"),
    ],
)
def test_invalid_baseline_raises(tmp_path: Path, mutate, match: str) -> None:
    path = tmp_path / "ci.json"
    save_baseline(_baseline(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SchemaError, match=match):
        load_baseline(path)


def test_baseline_from_results_uses_ok_variants() -> None:
    torch_row = dataclasses.replace(_row("torch_fp32", 50.0, parity_ok=None), runtime="torch")
    broken = dataclasses.replace(
        _row("onnx_int8_dynamic", 50.0), status="error", error="x", parity=None, accuracy=None,
        latency=None,
    )  # fmt: skip
    results = _results(torch_row, _row("onnx_fp32", 51.0), broken)
    baseline = baseline_from_results(
        results, max_accuracy_drop_pts=0.5, created_at="2026-01-02T00:00:00Z"
    )
    assert baseline.target == "ci"
    assert baseline.created_from_run_id == results.run_id
    assert baseline.entries == (
        BaselineEntry("tiny_cnn", "torch_fp32", top1=50.0, require_parity=False),
        BaselineEntry("tiny_cnn", "onnx_fp32", top1=51.0, require_parity=True),
    )
