from __future__ import annotations

import subprocess
import sys

PURE_MODULES = ("edgebench.parity", "edgebench.bench", "edgebench.results", "edgebench.gate")


def test_pure_modules_do_not_import_heavy_dependencies() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['onnxruntime'] = None\n"
        f"import {', '.join(PURE_MODULES)}\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
