from __future__ import annotations

from pathlib import Path

from edgebench.report import COLUMNS, plot_pareto, render_markdown, write_report
from edgebench.results import RunResults


def _table_rows(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    start = lines.index("| " + " | ".join(COLUMNS) + " |")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


def test_columns_match_design_order() -> None:
    assert COLUMNS == (
        "Variant", "Status", "Top-1 (%)", "Δ vs FP32 (pts)", "Top-1 agreement vs torch",
        "Max abs err", "p50 ms (b=1)", "p90 ms (b=1)", "CV (%)", "p50 ms (b=8)", "Size (MB)",
        "Load (ms)",
    )  # fmt: skip


def test_markdown_title_and_one_row_per_variant(sample_results: RunResults) -> None:
    md = render_markdown(sample_results)
    assert md.startswith("# EdgeBench report — tiny_cnn on ci")
    assert sample_results.run_id in md
    rows = _table_rows(md)
    assert len(rows) == len(sample_results.variants)
    for row, variant in zip(rows, sample_results.variants, strict=True):
        assert row.startswith(f"| {variant.variant} |")


def test_number_formats(sample_results: RunResults) -> None:
    rows = {
        r.split(" | ")[0][2:]: r.split(" | ") for r in _table_rows(render_markdown(sample_results))
    }
    fp32 = rows["onnx_fp32"]
    assert fp32[2] == "50.00"  # top-1, 2 decimals
    assert fp32[3] == "+0.00"  # delta
    assert fp32[5] == "3.1e-06"  # max abs err, scientific
    assert fp32[6] == "2.00"  # p50 b=1
    assert fp32[10] == "0.02"  # size
    static = rows["onnx_int8_static"]
    assert static[3] == "-1.56"
    assert static[8] == "15.0 (noisy)"


def test_errored_variant_shows_status_and_dashes(sample_results: RunResults) -> None:
    rows = {r.split(" | ")[0][2:]: r for r in _table_rows(render_markdown(sample_results))}
    row = rows["onnx_int8_dynamic"]
    assert "error: QuantizationError: boom" in row
    assert "\\|" in row  # pipe in the error message is escaped, table stays intact
    cells = [c.strip() for c in row.strip("|").split(" | ")]
    assert cells[2:] == ["—"] * 10


def test_environment_methodology_and_limitations(sample_results: RunResults) -> None:
    md = render_markdown(sample_results)
    for text in ("## Environment", "Test CPU", "onnxruntime 1.31.0", "torch 2.14.1",
                 "## Methodology", "10-way", "## Limitations", "not representative"):  # fmt: skip
        assert text in md


def test_pareto_png_created(sample_results: RunResults, tmp_path: Path) -> None:
    path = tmp_path / "pareto.png"
    plot_pareto(sample_results, path)
    assert path.is_file() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_report_creates_both_files(sample_results: RunResults, tmp_path: Path) -> None:
    md_path, png_path = write_report(sample_results, tmp_path / "out")
    assert md_path == tmp_path / "out" / "report.md"
    assert png_path == tmp_path / "out" / "pareto.png"
    assert md_path.read_text(encoding="utf-8").startswith("# EdgeBench report")
    assert png_path.stat().st_size > 0
