"""Results file schema (Data_Schema.md §4): typed dataclasses and the JSON boundary.

Must not import torch or onnxruntime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from edgebench.bench import LatencyStats
from edgebench.errors import SchemaError
from edgebench.parity import ParityReport

SCHEMA_VERSION = 1
STATUSES = frozenset({"ok", "error", "skipped"})


@dataclass(frozen=True)
class Environment:
    """Machine and software the run was measured on."""

    os: str
    kernel: str
    cpu_model: str
    logical_cores: int
    python: str
    packages: dict[str, str]


@dataclass(frozen=True)
class AccuracyResult:
    """10-way Imagenette top-1 in percent (Data_Schema.md §5)."""

    top1: float
    n_images: int
    delta_vs_onnx_fp32_pts: float | None


@dataclass(frozen=True)
class VariantResult:
    """One (model, variant) row. Parity, accuracy and latency are None unless status is ok."""

    model: str
    variant: str
    runtime: str
    precision: str
    status: str
    error: str | None
    artifact: str | None
    size_mb: float | None
    load_ms: float | None
    parity: ParityReport | None
    accuracy: AccuracyResult | None
    latency: dict[str, LatencyStats] | None


@dataclass(frozen=True)
class RunResults:
    """A complete results file."""

    schema_version: int
    run_id: str
    created_at: str
    edgebench_version: str
    config_hash: str
    config: dict[str, Any]
    environment: Environment
    variants: tuple[VariantResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-ready dicts and lists."""
        data = asdict(self)
        data["variants"] = list(data["variants"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResults:
        """Build from parsed JSON, validating the version and every required field.

        Raises:
            SchemaError: On a version mismatch, missing field or unknown status.
        """
        version = _require(data, "schema_version", "")
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"Unsupported results schema_version: expected {SCHEMA_VERSION}, found {version}"
            )
        values = _fields(cls, data, "", skip={"environment", "variants"})
        values["environment"] = _build(
            Environment, _require(data, "environment", ""), "environment"
        )
        rows = _require(data, "variants", "")
        values["variants"] = tuple(_variant(row, f"variants[{i}]") for i, row in enumerate(rows))
        return cls(**values)


def save_results(results: RunResults, path: Path) -> None:
    """Write results as UTF-8 JSON with sorted keys and 2-space indent (stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def load_results(path: Path) -> RunResults:
    """Load and validate a results file.

    Raises:
        SchemaError: If the file is not valid JSON or does not match the schema.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a JSON object at the top level")
    return RunResults.from_dict(data)


def _variant(row: dict[str, Any], where: str) -> VariantResult:
    values = _fields(VariantResult, row, where, skip={"parity", "accuracy", "latency"})
    if values["status"] not in STATUSES:
        raise SchemaError(f"Field '{where}.status' must be one of {sorted(STATUSES)}")
    latency = _require(row, "latency", where)
    values["parity"] = _optional(ParityReport, _require(row, "parity", where), f"{where}.parity")
    values["accuracy"] = _optional(
        AccuracyResult, _require(row, "accuracy", where), f"{where}.accuracy"
    )
    values["latency"] = (
        None
        if latency is None
        else {
            bs: _build(LatencyStats, stats, f"{where}.latency.{bs}")
            for bs, stats in latency.items()
        }
    )
    return VariantResult(**values)


def _optional(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    return None if data is None else _build(cls, data, where)


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise SchemaError(f"Field '{where}' must be an object")
    return cls(**_fields(cls, data, where))


def _fields(
    cls: type, data: dict[str, Any], where: str, skip: frozenset[str] | set[str] = frozenset()
) -> dict[str, Any]:
    """Collect ``cls``'s fields from ``data`` (except ``skip``), requiring each one."""
    return {f.name: _require(data, f.name, where) for f in fields(cls) if f.name not in skip}


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        name = f"{where}.{key}" if where else key
        raise SchemaError(f"Missing required field '{name}' in results file")
    return data[key]
