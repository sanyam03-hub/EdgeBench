"""Orchestration of ``edgebench run`` (Architecture.md Journey 1).

Prepares artifacts, then for each variant: load, parity vs torch, top-1, latency. A failure in
one variant is recorded in its row and the run continues (the only broad ``except``s, Rules §5).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from edgebench import __version__
from edgebench.backends.registry import VARIANT_SPECS, create_backend
from edgebench.backends.torch_backend import TorchBackend
from edgebench.bench import benchmark, file_size_mb
from edgebench.config import RunConfig
from edgebench.data import (
    IMAGENETTE_WNID_TO_IMAGENET_IDX,
    ArrayCalibrationReader,
    ImagenetteCalibrationReader,
    calibration_subset,
    eval_subset,
    load_batch,
    locate_imagenette,
    synthetic_split,
)
from edgebench.env import capture_environment
from edgebench.errors import DatasetError
from edgebench.evaluate import top1_accuracy
from edgebench.export import INPUT_NAME, export_onnx
from edgebench.models import LoadedModel, load_model
from edgebench.parity import compare
from edgebench.quantize import quantize_dynamic_int8, quantize_static_int8
from edgebench.results import (
    SCHEMA_VERSION,
    AccuracyResult,
    RunResults,
    VariantResult,
    save_results,
)

Progress = Callable[[str], None]
LABEL_INDICES = tuple(IMAGENETTE_WNID_TO_IMAGENET_IDX.values())


@dataclass(frozen=True)
class RunPaths:
    """Where inputs are read and outputs written; paths in results are relative to the root."""

    data_dir: Path
    artifacts_dir: Path
    results_dir: Path
    project_root: Path


@dataclass(frozen=True)
class _Unavailable:
    """A variant whose artifact could not be produced: recorded without running it."""

    status: str  # "error" | "skipped"
    message: str


def config_snapshot(run: RunConfig) -> dict[str, Any]:
    """The full model + target config as JSON-ready data (tuples become lists)."""
    raw = {"model": dataclasses.asdict(run.model), "target": dataclasses.asdict(run.target)}
    return json.loads(json.dumps(raw))


def config_hash(snapshot: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``snapshot``."""
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_run_id(digest: str, now: datetime) -> str:
    """``YYYYMMDDTHHMMSSZ_<first 8 hex chars of the config hash>``."""
    return f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}_{digest[:8]}"


def execute_run(
    run: RunConfig, paths: RunPaths, *, force: bool = False, progress: Progress = lambda _: None
) -> tuple[RunResults, Path]:
    """Run every configured variant and write ``results/<run_id>.json``.

    Raises:
        DatasetError: Dataset missing (checked before any model work) or pretrained weights
            unavailable.
        ConfigError: Unknown model or weights.
    """
    cfg = run.model
    data_root = None if cfg.dataset == "synthetic" else locate_imagenette(paths.data_dir)
    started = datetime.now(UTC)
    loaded = load_model(cfg.model, cfg.weights)
    data = _RunData(run, loaded, data_root)
    reference = TorchBackend(loaded.module, num_threads=run.target.intra_op_threads)
    ref_logits = reference.predict(data.parity_batch())
    artifacts = _prepare_artifacts(run, loaded, data, paths.artifacts_dir, force, progress)
    rows = [
        _run_variant(run, variant, artifacts.get(variant), data, ref_logits, paths, progress, i)
        for i, variant in enumerate(cfg.variants, start=1)
    ]
    snapshot = config_snapshot(run)
    digest = config_hash(snapshot)
    results = RunResults(
        schema_version=SCHEMA_VERSION,
        run_id=make_run_id(digest, started),
        created_at=f"{started:%Y-%m-%dT%H:%M:%SZ}",
        edgebench_version=__version__,
        config_hash=digest,
        config=snapshot,
        environment=capture_environment(),
        variants=tuple(_with_deltas(rows)),
    )
    path = paths.results_dir / f"{results.run_id}.json"
    save_results(results, path)
    return results, path


class _RunData:
    """Eval, parity, benchmark and calibration inputs for one run (streamed for Imagenette)."""

    def __init__(self, run: RunConfig, loaded: LoadedModel, root: Path | None) -> None:
        self._cfg = run.model
        self._transforms = loaded.transforms
        self._shape = loaded.input_shape
        self._root = root
        if root is None:
            self._eval = synthetic_split(
                self._cfg.eval.n_images, self._shape, self._cfg.eval.seed, "val"
            )
        else:
            self._eval_paths = eval_subset(root, self._cfg.eval)

    def eval_batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        size = self._cfg.eval.batch_size
        for start in range(0, self._cfg.eval.n_images, size):
            yield self._first(start + size, start)

    def parity_batch(self) -> np.ndarray:
        return self._first(self._cfg.parity.n_images)[0]

    def bench_input(self, batch_size: int) -> np.ndarray:
        if batch_size > self._cfg.eval.n_images:
            raise DatasetError(
                f"Benchmark batch size {batch_size} exceeds eval subset ({self._cfg.eval.n_images})"
            )
        return self._first(batch_size)[0]

    def calibration_reader(self) -> ArrayCalibrationReader | ImagenetteCalibrationReader:
        calib = self._cfg.quantization.calibration
        if self._root is None:
            images, _ = synthetic_split(calib.n_images, self._shape, calib.seed, "train")
            return ArrayCalibrationReader(INPUT_NAME, images, calib.batch_size)
        paths = calibration_subset(self._root, calib)
        return ImagenetteCalibrationReader(INPUT_NAME, paths, self._transforms, calib.batch_size)

    def _first(self, stop: int, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if self._root is None:
            return self._eval[0][start:stop], self._eval[1][start:stop]
        return load_batch(self._eval_paths[start:stop], self._transforms)


def _prepare_artifacts(
    run: RunConfig,
    loaded: LoadedModel,
    data: _RunData,
    artifacts_dir: Path,
    force: bool,
    progress: Progress,
) -> dict[str, Path | _Unavailable]:
    variants = run.model.variants
    out_dir = artifacts_dir / run.model.model
    result: dict[str, Path | _Unavailable] = {}
    for variant in variants:
        if variant.startswith("tflite_"):
            path = out_dir / f"{variant}.tflite"
            missing = _Unavailable("skipped", "run scripts/convert_tflite.py first")
            result[variant] = path if path.is_file() else missing
    onnx_variants = [v for v in variants if v.startswith("onnx_")]
    if not onnx_variants:
        return result
    fp32 = out_dir / "onnx_fp32.onnx"
    try:
        if force or not fp32.is_file():
            progress(f"exporting {run.model.model} to ONNX (opset {run.model.export.opset})")
            export_onnx(loaded.module, data.parity_batch()[:1], fp32, opset=run.model.export.opset)
    except Exception as exc:  # per-variant boundary: export failure is recorded, run continues
        fp32.unlink(missing_ok=True)  # never reuse a half-written export
        result["onnx_fp32"] = _Unavailable("error", _describe(exc))
        for variant in onnx_variants:
            if variant != "onnx_fp32":
                reason = f"onnx_fp32 export failed: {_describe(exc)}"
                result[variant] = _Unavailable("skipped", reason)
        return result
    result["onnx_fp32"] = fp32
    for variant in onnx_variants:
        if variant != "onnx_fp32":
            result[variant] = _quantize(variant, run, fp32, out_dir, data, progress)
    return result


def _quantize(
    variant: str, run: RunConfig, fp32: Path, out_dir: Path, data: _RunData, progress: Progress
) -> Path | _Unavailable:
    out = out_dir / f"{variant}.onnx"
    progress(f"quantizing {variant}")
    try:
        if variant == "onnx_int8_dynamic":
            return quantize_dynamic_int8(fp32, out)
        static = run.model.quantization.static
        return quantize_static_int8(
            fp32,
            out,
            data.calibration_reader(),
            per_channel=static.per_channel,
            calibrate_method=static.calibrate_method,
            activation_type=static.activation_type,
            weight_type=static.weight_type,
        )
    except Exception as exc:  # per-variant boundary: quantization failure is recorded
        return _Unavailable("error", _describe(exc))


def _run_variant(
    run: RunConfig,
    variant: str,
    artifact: Path | _Unavailable | None,
    data: _RunData,
    ref_logits: np.ndarray,
    paths: RunPaths,
    progress: Progress,
    index: int,
) -> VariantResult:
    cfg, target = run.model, run.target
    runtime, precision = VARIANT_SPECS[variant]
    tag = f"[{index}/{len(cfg.variants)}] {variant}"
    row = VariantResult(
        model=cfg.model, variant=variant, runtime=runtime, precision=precision, status="ok",
        error=None, artifact=None, size_mb=None, load_ms=None, parity=None, accuracy=None,
        latency=None,
    )  # fmt: skip
    if isinstance(artifact, _Unavailable):
        progress(f"{tag}: {artifact.status} ({artifact.message})")
        return dataclasses.replace(row, status=artifact.status, error=artifact.message)
    if artifact is not None:
        row = dataclasses.replace(
            row, artifact=_relative(artifact, paths.project_root), size_mb=file_size_mb(artifact)
        )
    try:
        backend, load_ms = create_backend(
            variant, model_name=cfg.model, artifact=artifact, target=target, weights=cfg.weights
        )
        parity = None
        if variant != "torch_fp32":
            tolerance = cfg.parity.fp32 if precision == "fp32" else cfg.parity.int8
            parity = compare(ref_logits, backend.predict(data.parity_batch()), tolerance)
            verdict = "ok" if parity.within_tolerance else "FAIL"
            progress(f"{tag}: parity {verdict} (top-1 agreement {parity.top1_agreement:.3f})")
        top1 = top1_accuracy(backend, data.eval_batches(), LABEL_INDICES)
        progress(f"{tag}: top-1 {top1:.2f}% - benchmarking...")
        latency = {
            str(b): benchmark(
                backend, data.bench_input(b), target.warmup, target.runs, target.noisy_cv_threshold
            )
            for b in target.batch_sizes
        }
    except Exception as exc:  # per-variant boundary (Rules §5): record and continue
        progress(f"{tag}: error ({_describe(exc)})")
        return dataclasses.replace(row, status="error", error=_describe(exc))
    for b, stats in latency.items():
        noisy = " (noisy)" if stats.noisy else ""
        progress(f"{tag}: batch {b} p50 {stats.p50_ms:.2f} ms, CV {stats.cv * 100:.1f}%{noisy}")
    return dataclasses.replace(
        row,
        load_ms=load_ms,
        parity=parity,
        accuracy=AccuracyResult(top1=top1, n_images=cfg.eval.n_images, delta_vs_onnx_fp32_pts=None),
        latency=latency,
    )


def _with_deltas(rows: list[VariantResult]) -> list[VariantResult]:
    """Fill ``delta_vs_onnx_fp32_pts`` (None for torch_fp32 or when onnx_fp32 has no accuracy)."""
    base = next((r for r in rows if r.variant == "onnx_fp32" and r.accuracy is not None), None)
    if base is None or base.accuracy is None:
        return rows
    out = []
    for row in rows:
        if row.accuracy is not None and row.variant != "torch_fp32":
            delta = row.accuracy.top1 - base.accuracy.top1
            row = dataclasses.replace(
                row, accuracy=dataclasses.replace(row.accuracy, delta_vs_onnx_fp32_pts=delta)
            )
        out.append(row)
    return out


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
