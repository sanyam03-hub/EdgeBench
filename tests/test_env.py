from __future__ import annotations

import dataclasses
import getpass
import json
import platform

from edgebench.env import PACKAGES, capture_environment, parse_cpuinfo_model
from edgebench.results import Environment


def test_capture_environment_fields() -> None:
    env = capture_environment()
    assert isinstance(env, Environment)
    assert env.os and env.kernel and env.cpu_model and env.python
    assert env.logical_cores >= 1
    assert env.python == platform.python_version()
    assert set(env.packages) == set(PACKAGES) == {
        "torch", "torchvision", "onnx", "onnxruntime", "numpy",
    }  # fmt: skip
    assert all(env.packages.values())


def test_no_hostname_or_username_recorded() -> None:
    text = json.dumps(dataclasses.asdict(capture_environment())).lower()
    for secret in (platform.node(), getpass.getuser()):
        assert secret.lower() not in text


def test_parse_cpuinfo_model() -> None:
    text = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Example CPU @ 2.0GHz\n"
    assert parse_cpuinfo_model(text) == "Example CPU @ 2.0GHz"


def test_parse_cpuinfo_model_missing() -> None:
    assert parse_cpuinfo_model("processor\t: 0\n") is None
