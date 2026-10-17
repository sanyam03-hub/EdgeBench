from __future__ import annotations

import pytest

import edgebench
from edgebench import cli, errors


def test_version_is_set() -> None:
    assert edgebench.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "name",
    [
        "ConfigError",
        "DatasetError",
        "ExportError",
        "QuantizationError",
        "BackendError",
        "SchemaError",
    ],
)
def test_project_errors_derive_from_base(name: str) -> None:
    exc_type = getattr(errors, name)
    assert issubclass(exc_type, errors.EdgeBenchError)
    assert issubclass(errors.EdgeBenchError, Exception)


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    assert "usage: edgebench" in capsys.readouterr().out


def test_cli_without_args_prints_help_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "usage: edgebench" in capsys.readouterr().out
