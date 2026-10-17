# PRD — Product Requirements Document

> Source of truth for **what** EdgeBench is and **why**. If any other doc conflicts with this one on scope, this one wins.

## App name
**EdgeBench**

## Tagline
A config-driven Python tool that converts PyTorch models to ONNX Runtime (FP32/INT8) and TFLite, checks the converted model's outputs match the original, benchmarks latency/size/accuracy, and fails CI when a conversion regresses.

## Problem
Engineers who deploy ML models to edge hardware convert every model into several runtimes and precisions. These conversions fail **silently** in three ways:

1. **Numerical drift** — export bugs, opset differences, or input-layout mismatches make the converted model produce different outputs than the original.
2. **Quantization accuracy loss** — INT8 models can lose accuracy, sometimes badly, and nobody notices until later.
3. **Misleading latency numbers** — single runs, no warmup and uncontrolled thread counts produce numbers that can't be trusted or reproduced.

Today this is often checked by hand in notebooks. EdgeBench makes it a repeatable, tested, CI-gated pipeline.

## Core value proposition
- **Functional correctness first:** every converted model is compared against the PyTorch reference before any speed number is reported.
- **Honest benchmarking:** percentiles, spread, warmup, fixed threads and a recorded environment — no single-number claims.
- **Regression gate:** a versioned baseline plus a `gate` command that fails CI when accuracy or parity regresses.
- **Extensible by design:** adding a runtime backend or a target profile does not require changing existing code.

## Target user persona
An ML deployment / validation engineer (e.g. on a team shipping models to mobile or embedded SoCs). They receive models from researchers or customers, convert them into several runtimes, and must sign off that each converted model is correct and fast enough. They live in Python, pytest and CI, and care more about reproducibility than about dashboards.

Secondary audience: **hiring managers and interviewers** reading the GitHub repo. The README and results must be understandable in 2 minutes and defensible in a 30-minute technical interview.

## Core features (Must have — MVP)
1. **Model loading:** torchvision `mobilenet_v3_small` and `resnet18` with pretrained weights.
2. **Dataset:** Imagenette (160px) with fixed, disjoint calibration (from train) and evaluation (from val) subsets.
3. **Exports:**
   - PyTorch → ONNX FP32
   - ONNX → ONNX INT8 dynamic quantization (ORT `quantize_dynamic`)
   - ONNX → ONNX INT8 static quantization, QDQ format, per-channel weights, calibrated on the train subset
4. **Backends:** PyTorch (reference), ONNX Runtime CPU — all behind one `Backend` interface.
5. **Parity check:** max abs error, mean abs error, cosine similarity, top-1 agreement vs the PyTorch reference.
6. **Accuracy evaluation:** top-1 on the fixed evaluation subset (definition in `Data_Schema.md`).
7. **Benchmark engine:** warmup, N timed runs, p50/p90/p99, mean, std, IQR, coefficient of variation, session load time, model file size, environment capture.
8. **Results:** one JSON file per run (schema in `Data_Schema.md`).
9. **Report:** markdown table + accuracy-vs-latency (Pareto) plot generated from results JSON.
10. **Regression gate:** compare a run against `baselines/*.json`; non-zero exit code on regression.
11. **CLI:** `edgebench run`, `edgebench report`, `edgebench gate`.
12. **CI:** GitHub Actions running ruff, pytest and a small end-to-end smoke test (no dataset download).

## Nice to have (only after MVP is done, in this order)
1. **TFLite backend:** PyTorch → TFLite FP32 via `ai-edge-torch` (separate virtualenv) + LiteRT interpreter backend. *Time-boxed; may slip to v2.*
2. **Per-layer quantization error analysis:** FP32-vs-INT8 activation comparison per layer, ranked; re-quantize with the worst layers excluded and measure accuracy recovery.
3. **Calibration sweep:** MinMax vs Entropy vs Percentile; per-tensor vs per-channel.
4. **Qualcomm AI Hub:** compile and profile on hosted Snapdragon devices (requires the user's own account and token).
5. Transformer model (e.g. DistilBERT) to contrast dynamic quantization behaviour.
6. Dockerfile; `.gitlab-ci.yml` mirror.

## Out of scope (this version does NOT do)
- No web UI, dashboard, Streamlit app or API server.
- No user accounts, authentication or database.
- No model training (pretrained weights only in MVP).
- No GPU / NPU execution in MVP — CPU only. **Results must never be described as "edge", "mobile" or "Snapdragon" numbers unless they were actually measured on that hardware.**
- No LLM / agent features.
- No gating CI on latency (shared CI runners are too noisy); latency is reported only.

## User stories
- As a deployment engineer, I want to run one command with a YAML config so that I get FP32 and INT8 variants of a model plus a results file, without writing glue code.
- As a deployment engineer, I want each converted model compared to the PyTorch reference so that I catch export bugs before trusting any benchmark.
- As a deployment engineer, I want latency reported with percentiles and variance so that I can tell a real speedup from noise.
- As a deployment engineer, I want to see accuracy loss per quantization method so that I can pick the right precision for a target.
- As a maintainer, I want CI to fail when parity or accuracy regresses against a baseline so that a dependency upgrade can't silently break conversions.
- As a maintainer, I want to add a new backend by implementing one interface so that existing backends and tests stay untouched.
- As a reviewer, I want a README with a results table, plot, methodology and limitations so that I can judge the work in minutes.

## Success metrics (for the project, not invented results)
- `edgebench run` produces results for **2 models × ≥4 variants** (PyTorch, ONNX FP32, ONNX INT8 dynamic, ONNX INT8 static) end-to-end on the user's machine.
- ONNX FP32 parity vs PyTorch is within the configured tolerance for both models (tolerance calibrated from observed values and documented).
- Every latency figure in the README has p50, p90 and CV, plus the CPU model and library versions.
- pytest suite passes locally and in CI; ruff reports zero issues.
- A deliberately broken baseline makes `edgebench gate` exit non-zero (demonstrated once).
- A fresh clone can reproduce the results table by following the README quickstart.
- **Every number in the README comes from a real run recorded in `results/`. No fabricated or estimated numbers.**
