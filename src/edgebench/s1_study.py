"""Stretch S1 study: rank layers by INT8 error, keep the worst k in FP32, measure the recovery.

Protocol (docs/Phases.md, S1):
- INT8 static calibration uses the model config's calibration subset (train/ only), exactly as
  ``edgebench run`` does;
- the eval subset (val/) is split once, seeded, into a *search* half and a *report* half;
- the rankings and the choice of k use only the search half;
- the final numbers (FP32, full INT8, each method's mixed-precision model and a random-exclusion
  baseline with the same k) come from the report half, which nothing was tuned on.

Two rankings of the quantized nodes:
- ``local_sqnr``: a node's worst local SQNR over its weights and the activations it reads
  (cheap: one pass over the analysis images; see layer_error.py);
- ``sensitivity``: SQNR of the model's logits vs FP32 when *only that node* is quantized and
  everything else stays FP32 (one quantization per node; measures the effect on the output).

Output ``s1.json`` (written by ``write_outputs``):
``{"schema": "edgebench.s1.v2", "model", "created_at", "settings", "environment",
"n_eligible_nodes", "tensors": [TensorError], "weights": [WeightError],
"rankings": {method: [NodeScore]}, "sweep": {"fp32_search_top1", "ranked": {method: {k: top1}},
"random": {k: [top1 per seed]}} | null, "chosen_k": {method: k} | null,
"excluded_nodes": {method: [node]} | null, "report": {"fp32", "int8": {"top1", "size_mb",
"latency_b1"}, "mixed": {method: {...}}, "random_mixed": {method: [{"seed", "top1"}]}} | null}``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from edgebench.backends.ort_backend import OrtBackend
from edgebench.bench import benchmark, file_size_mb
from edgebench.config import RunConfig
from edgebench.data import ArrayCalibrationReader, calibration_subset, eval_subset, load_batch
from edgebench.env import capture_environment
from edgebench.evaluate import top1_accuracy
from edgebench.export import INPUT_NAME
from edgebench.layer_error import (
    NodeScore,
    activation_errors,
    array_sqnr_db,
    choose_k,
    quantized_nodes,
    random_exclusions,
    rank_nodes,
    split_halves,
    weight_errors,
)
from edgebench.models import load_model
from edgebench.quantize import preprocess, quantize_static_int8
from edgebench.runner import LABEL_INDICES

SCHEMA = "edgebench.s1.v2"
Progress = Callable[[str], None]
METHOD_LABELS = {
    "local_sqnr": "worst local SQNR",
    "sensitivity": "most sensitive (logit SQNR)",
}
# Okabe-Ito, as in report.py.
SOURCE_COLOURS = {"weight": "#0072B2", "activation": "#E69F00", "logits": "#009E73"}
METHOD_COLOURS = {"local_sqnr": "#E69F00", "sensitivity": "#009E73"}
RANDOM_COLOUR = "#0072B2"


@dataclasses.dataclass(frozen=True)
class StudySettings:
    ks: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)  # numbers of top-ranked nodes kept in FP32
    random_seeds: int = 3  # random-exclusion draws per k
    analysis_images: int = 32  # search-half images for both rankings
    analysis_batch: int = 4
    max_drop_pts: float = 1.0  # choose the smallest k within this of FP32 (search half)
    sensitivity: bool = True  # False: local-SQNR ranking only (no per-node quantizations)
    sweep: bool = True  # False: rankings only (no re-quantization sweep)


def run_study(
    run: RunConfig,
    root: Path,
    fp32: Path,
    work: Path,
    settings: StudySettings | None = None,
    progress: Progress = lambda _: None,
) -> dict[str, Any]:
    """Run the S1 protocol for one model and return the ``s1.json`` content (JSON-native).

    Args:
        run: Model + target config; the target sets the threads and the latency runs.
        root: Imagenette root (``train/`` and ``val/``).
        fp32: The exported ONNX FP32 model (``edgebench run`` writes it).
        work: Scratch folder for the intermediate ONNX models.
    """
    settings = settings or StudySettings()
    cfg, target = run.model, run.target
    transforms = load_model(cfg.model, cfg.weights).transforms
    static, calib = cfg.quantization.static, cfg.quantization.calibration
    # Decoded once: every quantization below calibrates on exactly these arrays.
    calib_images = _load(calibration_subset(root, calib), calib.batch_size, transforms)

    def quantize(out: Path, exclude: Sequence[str]) -> Path:
        reader = ArrayCalibrationReader(INPUT_NAME, calib_images, calib.batch_size)
        return quantize_static_int8(
            fp32, out, reader,
            per_channel=static.per_channel, calibrate_method=static.calibrate_method,
            activation_type=static.activation_type, weight_type=static.weight_type,
            nodes_to_exclude=exclude,
        )  # fmt: skip

    def backend(model: Path) -> OrtBackend:
        return OrtBackend(
            model, variant=model.stem, precision="mixed",
            intra_op_threads=target.intra_op_threads, inter_op_threads=target.inter_op_threads,
        )  # fmt: skip

    def top1(model: Path, paths: Sequence[Path]) -> float:
        return top1_accuracy(backend(model), _batches(paths, cfg.eval.batch_size, transforms),
                             LABEL_INDICES)  # fmt: skip

    search, report = split_halves(eval_subset(root, cfg.eval), seed=cfg.eval.seed)
    progress(f"eval split: {len(search)} search / {len(report)} report images")
    pre = preprocess(fp32, work / "preprocessed.onnx")
    base = quantize(work / "int8_k0.onnx", [])
    eligible = quantized_nodes(base)

    analysis = search[: settings.analysis_images]
    progress(f"local SQNR on {len(analysis)} search images")
    images = (load_batch(c, transforms)[0] for c in _chunks(analysis, settings.analysis_batch))
    tensors = activation_errors(pre, base, images, INPUT_NAME, work / "augmented")
    weights = weight_errors(pre, base)
    rankings = {"local_sqnr": rank_nodes(tensors, weights, eligible)}

    if settings.sensitivity:
        sample = _load(analysis, settings.analysis_batch, transforms)
        reference = backend(fp32).predict(sample)
        all_nodes = [node.name for node in onnx.load(str(pre)).graph.node]
        scores = []
        for i, node in enumerate(sorted(eligible), start=1):
            single = quantize(work / "int8_single.onnx", [n for n in all_nodes if n != node])
            value = array_sqnr_db(reference, backend(single).predict(sample))
            scores.append(NodeScore(node, eligible[node], value, "logits"))
            if i % 10 == 0 or i == len(eligible):
                progress(f"sensitivity: {i}/{len(eligible)} nodes")
        rankings["sensitivity"] = sorted(scores, key=lambda s: (s.sqnr_db, s.node))

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "model": cfg.model,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {
            **dataclasses.asdict(settings),
            "n_search": len(search), "n_report": len(report), "eval_seed": cfg.eval.seed,
            "target": target.name, "calibration_images": calib.n_images,
        },  # fmt: skip
        "environment": dataclasses.asdict(capture_environment()),
        "n_eligible_nodes": len(eligible),
        "tensors": [dataclasses.asdict(t) for t in tensors],
        "weights": [dataclasses.asdict(w) for w in weights],
        "rankings": {m: [dataclasses.asdict(s) for s in r] for m, r in rankings.items()},
        "sweep": None, "chosen_k": None, "excluded_nodes": None, "report": None,
    }  # fmt: skip
    if settings.sweep:
        result.update(_sweep(rankings, eligible, settings, base, work, quantize, top1, progress,
                             fp32, search, report, transforms, backend, target))  # fmt: skip
    return json.loads(json.dumps(result))  # tuples -> lists: equal to what s1.json holds


def _sweep(
    rankings: dict[str, list[NodeScore]],
    eligible: dict[str, str],
    settings: StudySettings,
    base: Path,
    work: Path,
    quantize: Callable[[Path, Sequence[str]], Path],
    top1: Callable[[Path, Sequence[Path]], float],
    progress: Progress,
    fp32: Path,
    search: Sequence[Path],
    report: Sequence[Path],
    transforms: Any,
    backend: Callable[[Path], OrtBackend],
    target: Any,
) -> dict[str, Any]:
    """Search-half sweep over k for each ranking + random baseline; report-half final numbers."""
    order = {m: [s.node for s in r] for m, r in rankings.items()}

    def mixed_path(method: str, k: int) -> Path:
        return base if k == 0 else work / f"int8_{method}_top{k}.onnx"

    def random_top1(k: int, paths: Sequence[Path]) -> list[float]:
        return [
            top1(quantize(work / "int8_random.onnx", random_exclusions(eligible, k, seed)), paths)
            for seed in range(settings.random_seeds)
        ]

    fp32_search = top1(fp32, search)
    ranked: dict[str, dict[int, float]] = {m: {} for m in order}
    rand: dict[int, list[float]] = {}
    for k in (k for k in settings.ks if k <= len(eligible)):
        for method, nodes in order.items():
            model = base if k == 0 else quantize(mixed_path(method, k), nodes[:k])
            ranked[method][k] = top1(model, search)
        if k > 0:
            rand[k] = random_top1(k, search)
        progress(
            f"k={k}: " + ", ".join(f"{m} {ranked[m][k]:.2f}%" for m in order)
            + (f", random {', '.join(f'{v:.2f}' for v in rand[k])}%" if k else "")
            + f" (FP32 {fp32_search:.2f}%)"
        )  # fmt: skip
    chosen = {m: choose_k(ranked[m], fp32_search, settings.max_drop_pts) for m in order}
    progress(f"chosen k: {chosen} (search half); measuring on the report half")

    bench_input = load_batch(report[:1], transforms)[0]

    def measure(model: Path) -> dict[str, Any]:
        stats = benchmark(
            backend(model), bench_input, target.warmup, target.runs, target.noisy_cv_threshold
        )
        return {"top1": top1(model, report), "size_mb": file_size_mb(model),
                "latency_b1": dataclasses.asdict(stats)}  # fmt: skip

    report_random: dict[int, list[float]] = {}
    for k in set(chosen.values()):
        report_random[k] = random_top1(k, report)
    final = {
        "fp32": measure(fp32),
        "int8": measure(base),
        "mixed": {m: measure(mixed_path(m, k)) for m, k in chosen.items()},
        "random_mixed": {
            m: [{"seed": s, "top1": v} for s, v in enumerate(report_random[k])]
            for m, k in chosen.items()
        },
    }
    return {
        "sweep": {
            "fp32_search_top1": fp32_search,
            "ranked": {m: {str(k): v for k, v in c.items()} for m, c in ranked.items()},
            "random": {str(k): v for k, v in rand.items()},
        },
        "chosen_k": chosen,
        "excluded_nodes": {m: order[m][:k] for m, k in chosen.items()},
        "report": final,
    }


def _chunks(items: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _batches(
    paths: Sequence[Path], size: int, transforms: Any
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for chunk in _chunks(paths, size):
        yield load_batch(chunk, transforms)


def _load(paths: Sequence[Path], size: int, transforms: Any) -> np.ndarray:
    return np.concatenate([images for images, _ in _batches(paths, size, transforms)])


def write_outputs(result: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write ``s1.json``, ``report.md`` and the plots into ``out_dir``; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "s1.json", out_dir / "report.md"]
    with paths[0].open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with paths[1].open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(result))
    for method in result["rankings"]:
        paths.append(_plot_ranking(result, method, out_dir / f"ranking_{method}.png"))
    paths.append(_plot_depth(result, out_dir / "sqnr_depth.png"))
    if result["sweep"] is not None:
        paths.append(_plot_recovery(result, out_dir / "recovery.png"))
    return paths


def short_name(node: str) -> str:
    """Block + last two segments of an ONNX node name: ``features.1/block.0.0/Conv``.

    The block (second segment) is kept because the last segments repeat in every block.
    """
    op, sep, rest = node.partition(":")
    parts = (rest if sep else node).strip("/").split("/")
    tail = "/".join([parts[1], *parts[-2:]] if len(parts) > 3 else parts)
    return f"{op}:{tail}" if sep else tail


def render_markdown(result: dict[str, Any]) -> str:
    s = result["settings"]
    env = result["environment"]
    lines = [
        f"# S1 — per-layer INT8 error, {result['model']}",
        "",
        f"Created {result['created_at']} · target `{s['target']}` · {env['cpu_model']} · "
        f"onnxruntime {env['packages'].get('onnxruntime', '?')}",
        "",
        f"Static QDQ INT8 (calibration: {s['calibration_images']} train/ images). Eval subset "
        f"split (seed {s['eval_seed']}) into {s['n_search']} search / {s['n_report']} report "
        f"images. Rankings on {min(s['analysis_images'], s['n_search'])} search images; "
        f"{result['n_eligible_nodes']} quantized nodes can be kept in FP32.",
        "",
    ]
    for method, ranking in result["rankings"].items():
        lines += [
            f"## Top 10 by {METHOD_LABELS[method]} (lower SQNR = keep in FP32 first)",
            "",
            "| Rank | Node | Op | SQNR (dB) | Source |",
            "|---|---|---|---|---|",
            *(
                f"| {i} | `{short_name(r['node'])}` | {r['op_type']} | {r['sqnr_db']:.1f} "
                f"| {r['source']} |"
                for i, r in enumerate(ranking[:10], start=1)
            ),
            "",
        ]
    if result["sweep"] is not None:
        lines += _sweep_markdown(result)
    return "\n".join(lines)


def _sweep_markdown(result: dict[str, Any]) -> list[str]:
    s, sweep, rep = result["settings"], result["sweep"], result["report"]
    methods = list(sweep["ranked"])
    ks = list(sweep["ranked"][methods[0]])
    lines = [
        f"## Search half: keep the top-k nodes in FP32 (FP32 top-1 "
        f"{sweep['fp32_search_top1']:.2f}%)",
        "",
        "| k | " + " | ".join(f"{METHOD_LABELS[m]} (%)" for m in methods)
        + " | random k nodes (%), per seed |",
        "|---|" + "---|" * (len(methods) + 1),
    ]  # fmt: skip
    for k in ks:
        cells = [f"{sweep['ranked'][m][k]:.2f}" for m in methods]
        rand = ", ".join(f"{x:.2f}" for x in sweep["random"].get(k, [])) or "—"
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {rand} |")
    lines += [
        "",
        "Chosen k (smallest k within "
        f"{s['max_drop_pts']} pts of FP32 on the search half, else the best): "
        + ", ".join(f"{METHOD_LABELS[m]} k = {k}" for m, k in result["chosen_k"].items())
        + ".",
        "",
        "## Report half (not used for any choice)",
        "",
        "| Model | Top-1 (%) | Size (MB) | p50 ms (b=1) | p90 ms (b=1) | CV (%) |",
        "|---|---|---|---|---|---|",
        _report_row("ONNX FP32", rep["fp32"]),
        _report_row("INT8 static, all nodes", rep["int8"]),
        *(
            _report_row(f"INT8, top {result['chosen_k'][m]} by {METHOD_LABELS[m]} in FP32", row)
            for m, row in rep["mixed"].items()
        ),
        "",
        *(
            f"Random baseline for {METHOD_LABELS[m]} (same k = {result['chosen_k'][m]}, report "
            "half): " + ", ".join(f"{r['top1']:.2f}%" for r in runs) + "."
            for m, runs in rep["random_mixed"].items()
        ),
        "",
    ]  # fmt: skip
    return lines


def _report_row(name: str, row: dict[str, Any]) -> str:
    lat = row["latency_b1"]
    noisy = " (noisy)" if lat["noisy"] else ""
    return (
        f"| {name} | {row['top1']:.2f} | {row['size_mb']:.2f} | {lat['p50_ms']:.2f} | "
        f"{lat['p90_ms']:.2f} | {lat['cv'] * 100:.1f}{noisy} |"
    )


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish(fig: Any, plt: Any, result: dict[str, Any], extra: str, path: Path) -> Path:
    fig.text(0.01, 0.01, f"{result['environment']['cpu_model']} · {result['created_at']} · {extra}",
             fontsize=7)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _plot_ranking(result: dict[str, Any], method: str, path: Path, top: int = 20) -> Path:
    plt = _pyplot()
    rows = result["rankings"][method][:top][::-1]  # worst at the top
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    # Bars by position: equal labels must not merge into one categorical bar.
    ax.barh(
        range(len(rows)),
        [r["sqnr_db"] for r in rows],
        color=[SOURCE_COLOURS[r["source"]] for r in rows],
    )
    ax.set_yticks(range(len(rows)), [short_name(r["node"]) for r in rows], fontsize=6)
    ax.axvline(0, color="grey", linewidth=0.5)
    if method == "sensitivity":
        ax.set_xlabel("Logit SQNR vs FP32 with only this node in INT8 (dB)")
    else:
        ax.set_xlabel("Worst local SQNR of the node's weights / inputs (dB)")
    title = "most sensitive nodes" if method == "sensitivity" else "worst local SQNR"
    ax.set_title(f"{result['model']} — top {len(rows)}, {title}", fontsize=12)
    sources = list(dict.fromkeys(r["source"] for r in rows))
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLOURS[s]) for s in sources]
    # Worst (shortest) bars are at the top, so the upper right is free.
    ax.legend(handles, sources, fontsize=8, loc="upper right")
    return _finish(fig, plt, result, "lower SQNR = more error", path)


def _plot_depth(result: dict[str, Any], path: Path) -> Path:
    plt = _pyplot()
    rows = sorted((t for t in result["tensors"] if t["depth"] is not None),
                  key=lambda t: t["depth"])  # fmt: skip
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    local = [(t["depth"], t["local_sqnr_db"]) for t in rows if t["local_sqnr_db"] is not None]
    model = [(t["depth"], t["model_sqnr_db"]) for t in rows if t["model_sqnr_db"] is not None]
    ax.plot([d for d, _ in local], [v for _, v in local], "o", markersize=3, color="#E69F00",
            label="local (this tensor's own quantization)")  # fmt: skip
    ax.plot([d for d, _ in model], [v for _, v in model], "-", color="#0072B2", linewidth=1,
            label="FP32 model vs INT8 model (accumulated)")  # fmt: skip
    values = [v for _, v in local + model]
    top = min(max(values, default=60.0), 80.0)  # exact (clamped ~300 dB) tensors would flatten it
    ax.set_ylim(min(min(values, default=0.0), 0.0) - 2, top + 5)
    ax.set_xlabel("Node index in the FP32 graph (input → output)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title(f"{result['model']} — where INT8 diverges from FP32", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    return _finish(fig, plt, result, f"y-axis capped at {top + 5:.0f} dB", path)


def _plot_recovery(result: dict[str, Any], path: Path) -> Path:
    plt = _pyplot()
    sweep = result["sweep"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ks: list[int] = []
    for method, curve in sweep["ranked"].items():
        ks = [int(k) for k in curve]
        ax.plot(ks, list(curve.values()), "o-", color=METHOD_COLOURS[method],
                label=f"top-k by {METHOD_LABELS[method]}")  # fmt: skip
    for k, values in sweep["random"].items():
        ax.plot([int(k)] * len(values), values, "s", color=RANDOM_COLOUR, markersize=4, alpha=0.8)
    ax.plot([], [], "s", color=RANDOM_COLOUR, markersize=4, label="random k nodes (one per seed)")
    ax.axhline(sweep["fp32_search_top1"], color="grey", linestyle="--", linewidth=1,
               label="ONNX FP32")  # fmt: skip
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(-0.3, max(ks, default=1) * 1.3)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Nodes kept in FP32 (k)")
    ax.set_ylabel("Top-1 accuracy, search half (%)")
    ax.set_title(f"{result['model']} — accuracy recovery vs k", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    n = result["settings"]["n_search"]
    return _finish(fig, plt, result, f"{n} search-half images · y-axis 0–100 %", path)
