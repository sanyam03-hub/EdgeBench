"""Regression gate: compare a results file with a stored baseline (Architecture.md Journey 3).

Per baseline entry: present -> status ok -> top-1 not below baseline minus the allowed drop ->
parity within tolerance (if required). Latency is reported, never gated.
Pure Python: must not import torch or onnxruntime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edgebench.errors import SchemaError
from edgebench.results import RunResults, VariantResult

SCHEMA_VERSION = 1
# Float slack so "exactly at the threshold" passes despite rounding (e.g. 50.0 - 0.5).
_EPS = 1e-9


@dataclass(frozen=True)
class BaselineEntry:
    """Expected top-1 for one (model, variant) and whether its parity must hold."""

    model: str
    variant: str
    top1: float
    require_parity: bool


@dataclass(frozen=True)
class Baseline:
    """A gate baseline for one target (Data_Schema.md §7)."""

    schema_version: int
    target: str
    created_from_run_id: str
    created_at: str
    max_accuracy_drop_pts: float
    entries: tuple[BaselineEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON layout of Data_Schema.md §7 (threshold nested under ``thresholds``)."""
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "created_from_run_id": self.created_from_run_id,
            "created_at": self.created_at,
            "thresholds": {"max_accuracy_drop_pts": self.max_accuracy_drop_pts},
            "entries": [asdict(entry) for entry in self.entries],
        }


@dataclass(frozen=True)
class GateCheck:
    """One check outcome (Data_Schema.md §8)."""

    model: str
    variant: str
    check: str  # "present" | "status" | "accuracy" | "parity" | "latency_info"
    passed: bool
    detail: str


def evaluate_gate(results: RunResults, baseline: Baseline) -> list[GateCheck]:
    """Run every check for every baseline entry, in order.

    Raises:
        SchemaError: The results were measured on a different target than the baseline.
    """
    target = _results_target(results)
    if target != baseline.target:
        raise SchemaError(
            f"Results are for target {target!r} but the baseline is for {baseline.target!r}; "
            "baselines are per target and must not be compared across machines."
        )
    rows = {(row.model, row.variant): row for row in results.variants}
    checks: list[GateCheck] = []
    for entry in baseline.entries:
        row = rows.get((entry.model, entry.variant))
        checks.extend(_entry_checks(entry, row, baseline.max_accuracy_drop_pts))
    return checks


def gate_passed(checks: list[GateCheck]) -> bool:
    """True only if every check passed."""
    return all(check.passed for check in checks)


def _entry_checks(
    entry: BaselineEntry, row: VariantResult | None, max_drop: float
) -> list[GateCheck]:
    """The check chain for one entry; it stops at the first failure."""

    def check(name: str, passed: bool, detail: str) -> GateCheck:
        return GateCheck(entry.model, entry.variant, name, passed, detail)

    if row is None:
        return [check("present", False, "variant missing from results")]
    out = [check("present", True, "variant present")]
    if row.status != "ok":
        out.append(check("status", False, f"status {row.status}: {row.error}"))
        return out
    out.append(check("status", True, "status ok"))
    if row.accuracy is None:
        out.append(check("accuracy", False, "no accuracy recorded"))
        return out
    top1, minimum = row.accuracy.top1, entry.top1 - max_drop
    passed = top1 >= minimum - _EPS
    detail = (
        f"top-1 {top1:.2f}% vs baseline {entry.top1:.2f}% "
        f"(minimum {minimum:.2f}%, change {top1 - entry.top1:+.2f} pts)"
    )
    out.append(check("accuracy", passed, detail))
    if not passed:
        return out
    if entry.require_parity:
        parity = row.parity
        ok = parity is not None and parity.within_tolerance
        detail = (
            "no parity recorded"
            if parity is None
            else f"top-1 agreement {parity.top1_agreement:.3f}, max abs err "
            f"{parity.max_abs_err:.1e} (tolerance {parity.tolerance})"
        )
        out.append(check("parity", ok, detail))
        if not ok:
            return out
    out.append(check("latency_info", True, _latency_detail(row)))
    return out


def _latency_detail(row: VariantResult) -> str:
    if not row.latency:
        return "no latency recorded (informational, not gated)"
    parts = ", ".join(f"b={b} p50 {s.p50_ms:.2f} ms" for b, s in row.latency.items())
    return f"{parts} (informational, not gated)"


def baseline_from_results(
    results: RunResults, *, max_accuracy_drop_pts: float, created_at: str
) -> Baseline:
    """A baseline with one entry per ``ok`` variant; parity required where it was measured."""
    entries = tuple(
        BaselineEntry(
            model=row.model,
            variant=row.variant,
            top1=row.accuracy.top1,
            require_parity=row.parity is not None,
        )
        for row in results.variants
        if row.status == "ok" and row.accuracy is not None
    )
    return Baseline(
        schema_version=SCHEMA_VERSION,
        target=_results_target(results),
        created_from_run_id=results.run_id,
        created_at=created_at,
        max_accuracy_drop_pts=max_accuracy_drop_pts,
        entries=entries,
    )


def save_baseline(baseline: Baseline, path: Path) -> None:
    """Write UTF-8 JSON with sorted keys and 2-space indent (stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(baseline.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def load_baseline(path: Path) -> Baseline:
    """Load and validate a baseline file.

    Raises:
        SchemaError: Missing file, invalid JSON, wrong ``schema_version`` or missing field.
    """
    if not path.is_file():
        raise SchemaError(f"Baseline file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc
    if _require(data, "schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"Unsupported baseline schema_version: expected {SCHEMA_VERSION}, "
            f"found {data['schema_version']}"
        )
    thresholds = _require(data, "thresholds")
    entries = tuple(
        BaselineEntry(
            model=_require(e, "model", f"entries[{i}]"),
            variant=_require(e, "variant", f"entries[{i}]"),
            top1=float(_require(e, "top1", f"entries[{i}]")),
            require_parity=bool(_require(e, "require_parity", f"entries[{i}]")),
        )
        for i, e in enumerate(_require(data, "entries"))
    )
    return Baseline(
        schema_version=SCHEMA_VERSION,
        target=_require(data, "target"),
        created_from_run_id=_require(data, "created_from_run_id"),
        created_at=_require(data, "created_at"),
        max_accuracy_drop_pts=float(_require(thresholds, "max_accuracy_drop_pts", "thresholds")),
        entries=entries,
    )


def _results_target(results: RunResults) -> str:
    try:
        return str(results.config["target"]["name"])
    except (KeyError, TypeError) as exc:
        raise SchemaError("Results file has no config.target.name") from exc


def _require(data: Any, key: str, where: str = "") -> Any:
    if not isinstance(data, dict) or key not in data:
        name = f"{where}.{key}" if where else key
        raise SchemaError(f"Missing required field '{name}' in baseline file")
    return data[key]
