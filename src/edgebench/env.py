"""Capture the machine and software a run was measured on.

Records only CPU model, core count, OS/kernel and versions — never hostname, username,
home paths or environment variables (Security.md §1).
"""

from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path

from edgebench.results import Environment

PACKAGES = ("torch", "torchvision", "onnx", "onnxruntime", "numpy")
_WINDOWS_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


def capture_environment() -> Environment:
    """Return the current environment block for a results file."""
    system = platform.system()
    return Environment(
        os=system,
        kernel=platform.version() if system == "Windows" else platform.release(),
        cpu_model=_cpu_model(system),
        logical_cores=os.cpu_count() or 0,
        python=platform.python_version(),
        packages={name: _version(name) for name in PACKAGES},
    )


def parse_cpuinfo_model(text: str) -> str | None:
    """The first ``model name`` value from ``/proc/cpuinfo`` text, or None."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return value.strip()
    return None


def _cpu_model(system: str) -> str:
    model: str | None = None
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            model = parse_cpuinfo_model(cpuinfo.read_text(encoding="utf-8", errors="replace"))
    elif system == "Windows":
        model = _windows_cpu_model()
    return (model or platform.processor() or "unknown").strip()


def _windows_cpu_model() -> str | None:
    import winreg  # Windows-only stdlib module

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_CPU_KEY) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        return None
    return str(value)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"
