"""Command-line entry point: ``edgebench``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from edgebench import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser. Subcommands are added in later phases."""
    parser = argparse.ArgumentParser(
        prog="edgebench",
        description=(
            "Convert PyTorch models to ONNX Runtime (FP32/INT8), check parity against the "
            "PyTorch reference, benchmark latency/size/accuracy and gate regressions."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
