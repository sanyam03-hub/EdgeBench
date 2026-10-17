# Design — Output, Report and Visual Conventions

> EdgeBench has no app UI. Its "visual side" is: terminal output, the generated markdown report, the plots, and the README. These conventions keep them consistent and readable for reviewers.

## 1. Aesthetic direction
Plain, technical and dense with information — like a good benchmark report or a research paper table. No emoji, no colours that carry meaning on their own, no decorative elements.

## 2. Terminal output (`cli.py` only)
- Plain text, no colour libraries. Must read correctly when piped to a file or shown in CI logs.
- One line per step while running: `[3/8] onnx_int8_static: parity ok (top-1 agreement 0.998) — benchmarking...` (illustrative values only).
- Warnings start with `WARNING:`; errors with `ERROR:` and always include what to do next.
- End of `run`: a compact summary table (fixed-width columns) and the path of the results JSON.
- End of `gate`: one line per check, `PASS` / `FAIL`, then `GATE PASSED` or `GATE FAILED (n checks)`.
- `--verbose` switches logging to DEBUG; default INFO.

## 3. Markdown report (`report.md`)
Section order:
1. Title: `EdgeBench report — <model> on <target>` and the run ID.
2. **Results table**, one row per variant, columns in this order:

| Variant | Status | Top-1 (%) | Δ vs FP32 (pts) | Top-1 agreement vs torch | Max abs err | p50 ms (b=1) | p90 ms (b=1) | CV (%) | p50 ms (b=8) | Size (MB) | Load (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|

   - Numbers: accuracy 2 decimals; latency 2 decimals; max abs err in scientific notation (`3.1e-06`); size 2 decimals.
   - Δ vs FP32 is computed against `onnx_fp32` (the same runtime), not torch.
   - Noisy rows (CV above threshold) get `(noisy)` after the CV value.
   - Errored/skipped variants show the status and a short reason; other cells `—`.
3. **Environment** block: CPU model, cores, OS, Python, torch, onnxruntime versions, thread count, batch sizes, warmup/runs.
4. **Methodology** (fixed text): eval subset size, accuracy definition, calibration source, timing method, what is and isn't gated.
5. **Limitations** (fixed text): CPU only; not representative of NPU/DSP performance.

## 4. Plots (matplotlib)
- Backend `Agg`; saved as PNG, 150 dpi, 7×4.5 inches; white background (readable in GitHub light and dark modes).
- **Pareto plot** (`pareto.png`): x = p50 latency at batch 1 (ms), y = top-1 (%). One point per variant, labelled with the variant name. Horizontal whiskers from p50 to p90 latency.
- **Encoding:** colour = runtime, marker = precision.
  - Colours (Okabe-Ito, colour-blind safe): torch `#0072B2`, onnxruntime `#E69F00`, tflite `#009E73`, other `#CC79A7`.
  - Markers: FP32 `o` (circle), INT8 dynamic `^` (triangle), INT8 static `s` (square).
- Axis labels always include units: `Latency p50, batch 1 (ms)`, `Top-1 accuracy (%)`.
- Title: `<model> — <target>`; small caption with CPU model and run ID.
- Font: matplotlib default (DejaVu Sans), size 10; title 12. No custom fonts.
- Don't truncate the y-axis misleadingly: start at a round number below the minimum and state the range in the caption.
- Same colour/marker mapping in every plot in the repo.

## 5. README layout
1. Title + one-line tagline (from `PRD.md`).
2. CI badge.
3. **Why** (3–4 sentences: the problem from `PRD.md`).
4. **Results** — the headline table (subset of report columns) + `docs/pareto.png`, with the machine spec directly under the table.
5. **Architecture** — `docs/architecture.png` (export of the diagram in `Architecture.md` §1) + 5-line explanation.
6. **Quickstart** — at most 5 commands from clone to report.
7. **Methodology** — timing, tolerances, accuracy definition, data splits.
8. **Adding a backend / target** — short how-to (the extensibility story).
9. **Testing & CI** — how to run tests; what the gate checks and why latency isn't gated.
10. **Limitations** and **Roadmap** (stretch phases, clearly marked as not done).

Rules: real numbers only; every claim traceable to `results/sample/`; no marketing adjectives ("blazing fast", "state-of-the-art").

## 6. Accessibility
- Never rely on colour alone: markers differ by precision, and points are labelled.
- Tables use text status (`ok` / `error` / `skipped`), not symbols.
- Plots have descriptive alt text in the README.
