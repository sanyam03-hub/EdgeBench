from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from edgebench.config import (
    ModelConfig,
    RunConfig,
    TargetConfig,
    load_config,
    load_model_config,
    load_target_config,
)
from edgebench.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL: dict[str, Any] = {
    "schema_version": 1,
    "model": "mobilenet_v3_small",
    "weights": "DEFAULT",
    "dataset": "imagenette2-160",
    "variants": ["torch_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"],
    "export": {"opset": 17},
    "quantization": {
        "static": {
            "per_channel": True,
            "calibrate_method": "minmax",
            "activation_type": "quint8",
            "weight_type": "qint8",
        },
        "calibration": {"n_images": 256, "batch_size": 32, "seed": 0},
    },
    "eval": {"n_images": 1000, "batch_size": 32, "seed": 0},
    "parity": {
        "n_images": 32,
        "fp32": {"max_abs_err": 1.0e-3, "min_cosine": 0.9999, "min_top1_agreement": 0.99},
        "int8": {"min_top1_agreement": 0.90},
    },
}

TARGET: dict[str, Any] = {
    "schema_version": 1,
    "name": "cpu_4t",
    "intra_op_threads": 4,
    "inter_op_threads": 1,
    "batch_sizes": [1, 8],
    "warmup": 20,
    "runs": 200,
    "noisy_cv_threshold": 0.10,
}


def _write(tmp_path: Path, data: Any, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _names(field: str) -> str:
    """Regex: the quoted dotted field name, or a quoted child of it (e.g. 'parity.fp32.x')."""
    return "'" + re.escape(field) + "[.']"


def _edited(base: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Copy ``base`` and set (or delete, if value is ...) the field at ``dotted``."""
    data = copy.deepcopy(base)
    *parents, last = dotted.split(".")
    node = data
    for key in parents:
        node = node[key]
    if value is ...:
        del node[last]
    else:
        node[last] = value
    return data


def test_valid_model_config_loads(tmp_path: Path) -> None:
    cfg = load_model_config(_write(tmp_path, MODEL))
    assert isinstance(cfg, ModelConfig)
    assert cfg.model == "mobilenet_v3_small"
    assert cfg.variants == ("torch_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static")
    assert cfg.export.opset == 17
    assert cfg.quantization.static.per_channel is True
    assert cfg.quantization.calibration.n_images == 256
    assert cfg.eval.seed == 0
    assert cfg.parity.n_images == 32
    assert cfg.parity.fp32["min_cosine"] == 0.9999
    assert cfg.parity.int8 == {"min_top1_agreement": 0.90}


def test_valid_target_config_loads(tmp_path: Path) -> None:
    cfg = load_target_config(_write(tmp_path, TARGET))
    assert isinstance(cfg, TargetConfig)
    assert cfg.name == "cpu_4t"
    assert cfg.batch_sizes == (1, 8)
    assert cfg.noisy_cv_threshold == pytest.approx(0.10)


def test_configs_are_frozen(tmp_path: Path) -> None:
    cfg = load_target_config(_write(tmp_path, TARGET))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.runs = 1  # type: ignore[misc]


def test_load_config_combines_both(tmp_path: Path) -> None:
    run = load_config(_write(tmp_path, MODEL, "m.yaml"), _write(tmp_path, TARGET, "t.yaml"))
    assert isinstance(run, RunConfig)
    assert run.model.model == "mobilenet_v3_small"
    assert run.target.name == "cpu_4t"


@pytest.mark.parametrize(
    ("dotted", "value", "field"),
    [
        ("variants", ["torch_fp32", "onnx_fp16"], "variants"),
        ("variants", [], "variants"),
        ("variants", ["onnx_fp32", "onnx_fp32"], "variants"),
        ("model", ..., "model"),
        ("model", "../evil", "model"),
        ("model", "a/b", "model"),
        ("schema_version", 2, "schema_version"),
        ("export.opset", 0, "export.opset"),
        ("eval.n_images", -5, "eval.n_images"),
        ("eval.batch_size", True, "eval.batch_size"),
        ("eval.seed", -1, "eval.seed"),
        ("quantization.static.calibrate_method", "magic", "quantization.static.calibrate_method"),
        ("quantization.static.weight_type", "int4", "quantization.static.weight_type"),
        ("quantization.static.per_channel", "yes", "quantization.static.per_channel"),
        ("dataset", "cifar10", "dataset"),
        ("quantization.calibration", ..., "quantization.calibration"),
        ("parity.fp32", {"max_rel_err": 0.1}, "parity.fp32"),
        ("parity.int8", {"min_top1_agreement": "high"}, "parity.int8"),
        ("extra_key", 1, "extra_key"),
        ("eval.shuffle", True, "eval.shuffle"),
    ],
)
def test_invalid_model_config_names_field(
    tmp_path: Path, dotted: str, value: Any, field: str
) -> None:
    with pytest.raises(ConfigError, match=_names(field)):
        load_model_config(_write(tmp_path, _edited(MODEL, dotted, value)))


@pytest.mark.parametrize(
    ("dotted", "value", "field"),
    [
        ("runs", 0, "runs"),
        ("runs", -10, "runs"),
        ("batch_sizes", [], "batch_sizes"),
        ("batch_sizes", [1, 0], "batch_sizes"),
        ("warmup", -1, "warmup"),
        ("intra_op_threads", 0, "intra_op_threads"),
        ("noisy_cv_threshold", 0, "noisy_cv_threshold"),
        ("name", ..., "name"),
        ("name", "cpu/4t", "name"),
        ("gpu", True, "gpu"),
    ],
)
def test_invalid_target_config_names_field(
    tmp_path: Path, dotted: str, value: Any, field: str
) -> None:
    with pytest.raises(ConfigError, match=_names(field)):
        load_target_config(_write(tmp_path, _edited(TARGET, dotted, value)))


def test_warmup_zero_is_allowed(tmp_path: Path) -> None:
    assert load_target_config(_write(tmp_path, _edited(TARGET, "warmup", 0))).warmup == 0


def test_python_object_tag_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("schema_version: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(ConfigError, match="evil.yaml"):
        load_target_config(path)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_target_config(_write(tmp_path, [1, 2, 3]))


@pytest.mark.parametrize(
    "path",
    ["configs/mobilenet_v3_small.yaml", "configs/resnet18.yaml"],
)
def test_shipped_model_configs_load(path: str) -> None:
    load_model_config(REPO_ROOT / path)


@pytest.mark.parametrize(
    "path",
    ["configs/targets/cpu_1t.yaml", "configs/targets/cpu_4t.yaml", "configs/targets/ci.yaml"],
)
def test_shipped_target_configs_load(path: str) -> None:
    load_target_config(REPO_ROOT / path)
