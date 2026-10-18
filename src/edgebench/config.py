"""Load and validate model and target YAML configs (Data_Schema.md §2, §3) into frozen dataclasses.

Every validation error raises ConfigError naming the field as a quoted dotted path,
e.g. ``'eval.n_images'``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from edgebench.errors import ConfigError
from edgebench.parity import TOLERANCE_KEYS

SCHEMA_VERSION = 1
VARIANTS = (
    "torch_fp32",
    "onnx_fp32",
    "onnx_int8_dynamic",
    "onnx_int8_static",
    "tflite_fp32",
    "tflite_int8_dynamic",
)
DATASETS = ("imagenette2-160", "synthetic")
CALIBRATE_METHODS = ("minmax", "entropy", "percentile")
QUANT_TYPES = ("quint8", "qint8")
# Names become path components (artifacts/<model>/, baselines/<target>.json).
_SAFE_NAME = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class ExportConfig:
    opset: int


@dataclass(frozen=True)
class StaticQuantConfig:
    per_channel: bool
    calibrate_method: str
    activation_type: str
    weight_type: str


@dataclass(frozen=True)
class SubsetConfig:
    """A deterministic image subset: calibration (from train/) or eval (from val/)."""

    n_images: int
    batch_size: int
    seed: int


@dataclass(frozen=True)
class QuantizationConfig:
    static: StaticQuantConfig
    calibration: SubsetConfig


@dataclass(frozen=True)
class ParityConfig:
    n_images: int
    fp32: dict[str, float]
    int8: dict[str, float]


@dataclass(frozen=True)
class ModelConfig:
    model: str
    weights: str
    dataset: str
    variants: tuple[str, ...]
    export: ExportConfig
    quantization: QuantizationConfig
    eval: SubsetConfig
    parity: ParityConfig


@dataclass(frozen=True)
class TargetConfig:
    name: str
    intra_op_threads: int
    inter_op_threads: int
    batch_sizes: tuple[int, ...]
    warmup: int
    runs: int
    noisy_cv_threshold: float


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    target: TargetConfig


MODEL_KEYS = ("schema_version", *ModelConfig.__dataclass_fields__)
TARGET_KEYS = ("schema_version", *TargetConfig.__dataclass_fields__)


def load_config(model_path: Path, target_path: Path) -> RunConfig:
    """Load and validate a model config and a target profile."""
    return RunConfig(model=load_model_config(model_path), target=load_target_config(target_path))


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate ``configs/<model>.yaml``. Raises ConfigError."""
    data = _read_yaml(path)
    _keys(data, MODEL_KEYS, "")
    _schema_version(data)
    export = _section(data, "export", ["opset"])
    quant = _section(data, "quantization", ["static", "calibration"])
    parity = _section(data, "parity", ["n_images", "fp32", "int8"])
    static = _section(quant, "static", list(StaticQuantConfig.__dataclass_fields__), "quantization")
    return ModelConfig(
        model=_name(data["model"], "model"),
        weights=_str(data["weights"], "weights"),
        dataset=_choice(data["dataset"], "dataset", DATASETS),
        variants=_variants(data["variants"]),
        export=ExportConfig(opset=_int(export["opset"], "export.opset", 1)),
        quantization=QuantizationConfig(
            static=_static(static),
            calibration=_subset(quant, "calibration", "quantization"),
        ),
        eval=_subset(data, "eval", ""),
        parity=ParityConfig(
            n_images=_int(parity["n_images"], "parity.n_images", 1),
            fp32=_tolerance(parity["fp32"], "parity.fp32"),
            int8=_tolerance(parity["int8"], "parity.int8"),
        ),
    )


def load_target_config(path: Path) -> TargetConfig:
    """Load and validate ``configs/targets/<target>.yaml``. Raises ConfigError."""
    data = _read_yaml(path)
    _keys(data, TARGET_KEYS, "")
    _schema_version(data)
    batch_sizes = data["batch_sizes"]
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise ConfigError("Field 'batch_sizes' must be a non-empty list of positive integers")
    return TargetConfig(
        name=_name(data["name"], "name"),
        intra_op_threads=_int(data["intra_op_threads"], "intra_op_threads", 1),
        inter_op_threads=_int(data["inter_op_threads"], "inter_op_threads", 1),
        batch_sizes=tuple(_int(b, "batch_sizes", 1) for b in batch_sizes),
        warmup=_int(data["warmup"], "warmup", 0),
        runs=_int(data["runs"], "runs", 1),
        noisy_cv_threshold=_positive_float(data["noisy_cv_threshold"], "noisy_cv_threshold"),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must be a YAML mapping at the top level")
    return data


def _schema_version(data: dict[str, Any]) -> None:
    if data["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(
            f"Field 'schema_version' must be {SCHEMA_VERSION}, found {data['schema_version']!r}"
        )


def _keys(data: dict[str, Any], required: Iterable[str], where: str) -> None:
    """Require exactly ``required`` keys: none missing, none unknown."""
    required = list(required)
    for key in required:
        if key not in data:
            raise ConfigError(f"Missing required field '{_join(where, key)}'")
    for key in data:
        if key not in required:
            raise ConfigError(f"Unknown field '{_join(where, key)}'; allowed: {required}")


def _section(
    data: dict[str, Any], key: str, required: list[str], where: str = ""
) -> dict[str, Any]:
    path = _join(where, key)
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"Field '{path}' must be a mapping")
    _keys(value, required, path)
    return value


def _static(section: dict[str, Any]) -> StaticQuantConfig:
    where = "quantization.static"
    return StaticQuantConfig(
        per_channel=_bool(section["per_channel"], f"{where}.per_channel"),
        calibrate_method=_choice(
            section["calibrate_method"], f"{where}.calibrate_method", CALIBRATE_METHODS
        ),
        activation_type=_choice(
            section["activation_type"], f"{where}.activation_type", QUANT_TYPES
        ),
        weight_type=_choice(section["weight_type"], f"{where}.weight_type", QUANT_TYPES),
    )


def _subset(data: dict[str, Any], key: str, where: str) -> SubsetConfig:
    path = _join(where, key)
    section = _section(data, key, ["n_images", "batch_size", "seed"], where)
    return SubsetConfig(
        n_images=_int(section["n_images"], f"{path}.n_images", 1),
        batch_size=_int(section["batch_size"], f"{path}.batch_size", 1),
        seed=_int(section["seed"], f"{path}.seed", 0),
    )


def _variants(value: Any) -> tuple[str, ...]:
    allowed = list(VARIANTS)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Field 'variants' must be a non-empty list from {allowed}")
    for variant in value:
        if variant not in VARIANTS:
            raise ConfigError(
                f"Field 'variants' has unknown variant {variant!r}; allowed: {allowed}"
            )
    if len(set(value)) != len(value):
        raise ConfigError(f"Field 'variants' contains duplicates: {value}")
    return tuple(value)


def _tolerance(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"Field '{where}' must be a mapping")
    for key, threshold in value.items():
        if key not in TOLERANCE_KEYS:
            raise ConfigError(f"Unknown field '{where}.{key}'; allowed: {sorted(TOLERANCE_KEYS)}")
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ConfigError(f"Field '{where}.{key}' must be a number, got {threshold!r}")
    return {key: float(threshold) for key, threshold in value.items()}


def _int(value: Any, where: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"Field '{where}' must be an integer >= {minimum}, got {value!r}")
    return value


def _positive_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"Field '{where}' must be a number > 0, got {value!r}")
    return float(value)


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"Field '{where}' must be true or false, got {value!r}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Field '{where}' must be a non-empty string, got {value!r}")
    return value


def _choice(value: Any, where: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ConfigError(f"Field '{where}' must be one of {list(allowed)}, got {value!r}")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ConfigError(
            f"Field '{where}' must contain only letters, digits and underscores, got {value!r}"
        )
    return value


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key
