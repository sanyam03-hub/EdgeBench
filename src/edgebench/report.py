"""Results JSON -> ``report.md`` + ``pareto.png`` (Design.md §3-4). Never re-runs a model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from edgebench.bench import LatencyStats
from edgebench.results import RunResults, VariantResult

COLUMNS = (
    "Variant", "Status", "Top-1 (%)", "Δ vs FP32 (pts)", "Top-1 agreement vs torch",
    "Max abs err", "p50 ms (b=1)", "p90 ms (b=1)", "CV (%)", "p50 ms (b=8)", "Size (MB)",
    "Load (ms)",
)  # fmt: skip
DASH = "—"
# Okabe-Ito (colour-blind safe); same mapping in every plot (Design.md §4).
RUNTIME_COLOURS = {"torch": "#0072B2", "onnxruntime": "#E69F00", "litert": "#009E73"}
OTHER_COLOUR = "#CC79A7"
PRECISION_MARKERS = {"fp32": "o", "int8_dynamic": "^", "int8_static": "s"}
_MAX_REASON = 60


def write_report(results: RunResults, out_dir: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``pareto.png`` into ``out_dir``; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, png_path = out_dir / "report.md", out_dir / "pareto.png"
    with md_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(results))
    plot_pareto(results, png_path)
    return md_path, png_path


def render_markdown(results: RunResults) -> str:
    """The full markdown report: title, results table, environment, methodology, limitations."""
    model, target = _names(results)
    lines = [
        f"# EdgeBench report — {model} on {target}",
        "",
        f"Run ID: `{results.run_id}` · created {results.created_at} · "
        f"EdgeBench {results.edgebench_version}",
        "",
        "## Results",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
        *("| " + " | ".join(_cells(row)) + " |" for row in results.variants),
        "",
        "Δ vs FP32 is measured against `onnx_fp32` (same runtime). Top-1 agreement and max abs err "
        "compare each variant with the PyTorch FP32 reference on the parity batch. "
        "`(noisy)`: CV above the target's threshold.",
        "",
        *_environment(results),
        *_methodology(results),
        "## Limitations",
        "",
        "- CPU only (PyTorch eager and ONNX Runtime CPUExecutionProvider) on one machine; "
        "not representative of NPU/DSP or on-device mobile performance.",
        "- Latency depends on the CPU, thread count and background load; compare runs only "
        "within the same target profile and machine.",
        "",
    ]
    return "\n".join(lines)


def _cells(row: VariantResult) -> list[str]:
    if row.status != "ok":
        reason = (row.error or "").replace("|", "\\|")
        if len(reason) > _MAX_REASON:
            reason = reason[:_MAX_REASON] + "…"
        return [row.variant, f"{row.status}: {reason}", *[DASH] * (len(COLUMNS) - 2)]
    acc, par = row.accuracy, row.parity
    b1, b8 = _stats(row, "1"), _stats(row, "8")
    delta = acc.delta_vs_onnx_fp32_pts if acc else None
    cv = f"{b1.cv * 100:.1f}{' (noisy)' if b1.noisy else ''}" if b1 else DASH
    return [
        row.variant,
        row.status,
        f"{acc.top1:.2f}" if acc else DASH,
        f"{delta:+.2f}" if delta is not None else DASH,
        f"{par.top1_agreement:.3f}" if par else DASH,
        f"{par.max_abs_err:.1e}" if par else DASH,
        f"{b1.p50_ms:.2f}" if b1 else DASH,
        f"{b1.p90_ms:.2f}" if b1 else DASH,
        cv,
        f"{b8.p50_ms:.2f}" if b8 else DASH,
        f"{row.size_mb:.2f}" if row.size_mb is not None else DASH,
        f"{row.load_ms:.1f}" if row.load_ms is not None else DASH,
    ]


def _environment(results: RunResults) -> list[str]:
    env, target = results.environment, _config(results, "target")
    pkgs = env.packages
    versions = ", ".join(f"{name} {pkgs[name]}" for name in sorted(pkgs))
    return [
        "## Environment",
        "",
        f"- CPU: {env.cpu_model} ({env.logical_cores} logical cores)",
        f"- OS: {env.os} {env.kernel}",
        f"- Python {env.python}; {versions}",
        f"- Threads: intra-op {target.get('intra_op_threads', '?')}, "
        f"inter-op {target.get('inter_op_threads', '?')}",
        f"- Batch sizes: {target.get('batch_sizes', '?')}; warmup {target.get('warmup', '?')}, "
        f"timed runs {target.get('runs', '?')}",
        "",
    ]


def _methodology(results: RunResults) -> list[str]:
    model_cfg, target = _config(results, "model"), _config(results, "target")
    n_eval = model_cfg.get("eval", {}).get("n_images", "?")
    n_calib = model_cfg.get("quantization", {}).get("calibration", {}).get("n_images", "?")
    return [
        "## Methodology",
        "",
        f"- Accuracy: 10-way top-1 among the Imagenette classes (argmax over their 10 logits of "
        f"the 1000-class output) on {n_eval} images from `val/`. Not ImageNet top-1; do not "
        "compare with published ImageNet numbers. Meaningless for the random `tiny_cnn` "
        "smoke model.",
        f"- Calibration (INT8 static): {n_calib} images from `train/` only, disjoint from eval.",
        f"- Timing: `time.perf_counter_ns` around the inference call only; "
        f"{target.get('warmup', '?')} warmup runs discarded, {target.get('runs', '?')} timed runs "
        "recorded; p50/p90 by linear interpolation; CV = std/mean.",
        "- Gated (`edgebench gate`): presence, status, top-1 drop vs baseline, parity. "
        "Latency is reported, never gated.",
        "",
    ]


def plot_pareto(results: RunResults, path: Path) -> None:
    """Accuracy vs batch-1 p50 latency, whiskers to p90; colour = runtime, marker = precision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    model, target = _names(results)
    points = [
        (row, stats, row.accuracy.top1)
        for row in results.variants
        if row.accuracy is not None and (stats := _stats(row, "1")) is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    # Round number at least 5 pts below the minimum, so no point sits on the axis edge.
    low = max(0, math.floor((min(p[2] for p in points) - 5) / 10) * 10) if points else 0
    span = 100 - low
    # Alternate labels above/below in latency order, so x-neighbours never share a side;
    # points in the top 10 % of the range are labelled below to stay clear of the frame.
    for i, (row, stats, top1) in enumerate(sorted(points, key=lambda p: p[1].p50_ms)):
        colour = RUNTIME_COLOURS.get(row.runtime, OTHER_COLOUR)
        ax.errorbar(
            stats.p50_ms, top1, xerr=[[0.0], [stats.p90_ms - stats.p50_ms]],
            fmt=PRECISION_MARKERS.get(row.precision, "D"), color=colour, ecolor=colour,
            capsize=3, markersize=7,
        )  # fmt: skip
        below = i % 2 == 1 or top1 > 100 - 0.1 * span
        ax.annotate(row.variant, (stats.p50_ms, top1), textcoords="offset points",
                    xytext=(6, -14 if below else 8), fontsize=8)  # fmt: skip
    # Headroom of 3 % of the range keeps markers off the frame; ticks stop at 100.
    ax.set_ylim(low, 100 + 0.03 * span)
    ax.set_yticks([t for t in ax.get_yticks() if low <= t <= 100])
    ax.margins(x=0.15)
    if not points:
        ax.text(0.5, 0.5, "no variant with accuracy and batch-1 latency", ha="center",
                transform=ax.transAxes)  # fmt: skip
    ax.set_xlabel("Latency p50, batch 1 (ms)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"{model} — {target}", fontsize=12)
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="", label=r)
        for r, c in RUNTIME_COLOURS.items()
        if any(p[0].runtime == r for p in points)
    ]
    handles += [
        Line2D([], [], color="grey", marker=m, linestyle="", label=p)
        for p, m in PRECISION_MARKERS.items()
        if any(q[0].precision == p for q in points)
    ]
    if handles:  # outside the axes, so it never hides a data point
        ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.01, 0.01, f"{results.environment.cpu_model} · run {results.run_id} · "
             f"y-axis {low}–100 % · whiskers: p50 to p90", fontsize=7)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _stats(row: VariantResult, batch: str) -> LatencyStats | None:
    return row.latency.get(batch) if row.latency else None


def _config(results: RunResults, section: str) -> dict[str, Any]:
    value = results.config.get(section, {})
    return value if isinstance(value, dict) else {}


def _names(results: RunResults) -> tuple[str, str]:
    model = _config(results, "model").get("model") or (
        results.variants[0].model if results.variants else "?"
    )
    return str(model), str(_config(results, "target").get("name", "?"))
