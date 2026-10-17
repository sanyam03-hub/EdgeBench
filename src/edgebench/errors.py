"""Project exception hierarchy. The CLI maps these to exit codes (Rules.md §5)."""

from __future__ import annotations


class EdgeBenchError(Exception):
    """Base class for all EdgeBench errors."""


class ConfigError(EdgeBenchError):
    """A model or target config is missing, malformed or invalid."""


class DatasetError(EdgeBenchError):
    """The dataset is missing or has an unexpected layout."""


class ExportError(EdgeBenchError):
    """PyTorch to ONNX export or ONNX validation failed."""


class QuantizationError(EdgeBenchError):
    """ONNX Runtime quantization failed."""


class BackendError(EdgeBenchError):
    """A backend could not be created or received invalid input."""


class SchemaError(EdgeBenchError):
    """A results or baseline JSON file does not match the expected schema."""
