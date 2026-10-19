"""Command-line entry point: ``edgebench``. Argument parsing, printing and exit codes only.

Exit codes (Rules.md §5): 0 success, 1 variant failure / regression, 2 usage/config/data error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from edgebench import __version__
from edgebench.errors import ConfigError, DatasetError, EdgeBenchError, SchemaError
from edgebench.results import RunResults, VariantResult

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (``report`` and ``gate`` are added in Phase 7)."""
    parser = argparse.ArgumentParser(
        prog="edgebench",
        description=(
            "Convert PyTorch models to ONNX Runtime (FP32/INT8), check parity against the "
            "PyTorch reference, benchmark latency/size/accuracy and gate regressions."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="export, quantize, check parity, evaluate and benchmark")
    run.add_argument("--config", required=True, type=Path, help="model config YAML")
    run.add_argument("--target", required=True, type=Path, help="target profile YAML")
    run.add_argument("--force", action="store_true", help="re-export even if the ONNX file exists")
    report = sub.add_parser("report", help="write report.md and pareto.png for a results file")
    report.add_argument("--results", required=True, type=Path, help="results/<run_id>.json")
    gate = sub.add_parser("gate", help="check a results file against a baseline (exit 1 on fail)")
    gate.add_argument("--results", required=True, type=Path, help="results/<run_id>.json")
    gate.add_argument("--baseline", required=True, type=Path, help="baselines/<target>.json")
    gate.add_argument(
        "--write-baseline", action="store_true", help="create the baseline from the results"
    )
    gate.add_argument(
        "--max-accuracy-drop", type=float, default=0.5, help="pts, with --write-baseline"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    # Third-party libraries (ORT quantization logs via the root logger) stay at WARNING;
    # EdgeBench's own messages at INFO, or DEBUG with --verbose.
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr
    )
    logging.getLogger("edgebench").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    commands = {"run": _run, "report": _report, "gate": _gate}
    try:
        return commands[args.command](args)
    except (ConfigError, DatasetError, SchemaError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except EdgeBenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE


def _run(args: argparse.Namespace) -> int:
    from edgebench.config import load_config
    from edgebench.runner import RunPaths, execute_run

    run = load_config(args.config, args.target)
    paths = RunPaths(
        data_dir=Path(os.environ.get("EDGEBENCH_DATA_DIR", "data")),
        artifacts_dir=Path(os.environ.get("EDGEBENCH_ARTIFACTS_DIR", "artifacts")),
        results_dir=Path(os.environ.get("EDGEBENCH_RESULTS_DIR", "results")),
        project_root=Path.cwd(),
    )
    _say(f"EdgeBench run: {run.model.model} on {run.target.name}")
    results, path = execute_run(run, paths, force=args.force, progress=_say)
    _say("")
    _say(_summary_table(results))
    if any(row.latency and any(s.noisy for s in row.latency.values()) for row in results.variants):
        _say("* noisy: CV above the target's noisy_cv_threshold")
    _say(f"\nResults: {path}")
    failed = [row.variant for row in results.variants if row.status == "error"]
    if failed:
        print(f"WARNING: {len(failed)} variant(s) failed: {', '.join(failed)}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


def _report(args: argparse.Namespace) -> int:
    from edgebench.report import write_report
    from edgebench.results import load_results

    results = load_results(args.results)
    md_path, png_path = write_report(results, args.results.with_suffix(""))
    _say(f"Report: {md_path}")
    _say(f"Plot:   {png_path}")
    return EXIT_OK


def _gate(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from edgebench.gate import (
        baseline_from_results,
        evaluate_gate,
        gate_passed,
        load_baseline,
        save_baseline,
    )
    from edgebench.results import load_results

    results = load_results(args.results)
    if args.write_baseline:
        baseline = baseline_from_results(
            results,
            max_accuracy_drop_pts=args.max_accuracy_drop,
            created_at=f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        )
        save_baseline(baseline, args.baseline)
        _say(f"Baseline written: {args.baseline} ({len(baseline.entries)} entries)")
        return EXIT_OK
    checks = evaluate_gate(results, load_baseline(args.baseline))
    for check in checks:
        verdict = "PASS" if check.passed else "FAIL"
        _say(f"{verdict}  {check.model}/{check.variant}  {check.check}: {check.detail}")
    if gate_passed(checks):
        _say("GATE PASSED")
        return EXIT_OK
    _say(f"GATE FAILED ({sum(not c.passed for c in checks)} checks)")
    return EXIT_FAILURE


def _say(message: str) -> None:
    """Print a progress/summary line immediately (stdout is block-buffered when redirected)."""
    print(message, flush=True)


# ASCII only: output must survive Windows code pages when piped to a file or CI log.
_COLUMNS = (
    ("Variant", 18), ("Status", 7), ("Top-1 %", 8), ("Delta", 7), ("Agree", 6),
    ("Max abs err", 11), ("p50 ms", 8), ("CV %", 6), ("Size MB", 8),
)  # fmt: skip


def _summary_table(results: RunResults) -> str:
    """Fixed-width summary (Design.md §2); latency columns use the first batch size."""
    header = "  ".join(f"{name:<{width}}" for name, width in _COLUMNS)
    rows = [
        "  ".join(f"{cell:<{width}}" for cell, (_, width) in zip(_cells(r), _COLUMNS, strict=True))
        for r in results.variants
    ]
    return "\n".join([header, "-" * len(header), *rows])


def _cells(row: VariantResult) -> list[str]:
    dash = "-"
    acc, par = row.accuracy, row.parity
    stats = next(iter(row.latency.values())) if row.latency else None
    delta = acc.delta_vs_onnx_fp32_pts if acc else None
    cv = f"{stats.cv * 100:.1f}{'*' if stats.noisy else ''}" if stats else dash
    return [
        row.variant,
        row.status,
        f"{acc.top1:.2f}" if acc else dash,
        f"{delta:+.2f}" if delta is not None else dash,
        f"{par.top1_agreement:.3f}" if par else dash,
        f"{par.max_abs_err:.1e}" if par else dash,
        f"{stats.p50_ms:.2f}" if stats else dash,
        cv,
        f"{row.size_mb:.2f}" if row.size_mb is not None else dash,
    ]


if __name__ == "__main__":
    raise SystemExit(main())
