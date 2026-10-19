# Architecture — Flow, Components and Folder Structure

> Replaces the PDF's "App Flow" document. EdgeBench has no screens, so this document maps **command flows** instead of page flows: what happens, in what order, for each CLI command.

## 1. Big picture

```
configs/<model>.yaml + configs/targets/<target>.yaml
                 │
                 ▼
          config.load_config()  ──► RunConfig (frozen dataclass, validated)
                 │
     ┌───────────┼──────────────────────────────┐
     ▼           ▼                              ▼
 models.py    data.py                     target profile
 (torchvision (Imagenette: calib subset    (threads, batch sizes,
  + weights)   from train, eval subset      warmup, runs)
               from val, shared
               preprocessing)
     │
     ▼
 export.py ──► artifacts/<model>/onnx_fp32.onnx
     │
     ▼
 quantize.py ─► artifacts/<model>/onnx_int8_dynamic.onnx
             └► artifacts/<model>/onnx_int8_static.onnx   (calibrated on calib subset)

 scripts/convert_tflite.py (runs in .venv-tflite, optional)
             └► artifacts/<model>/tflite_fp32.tflite

                 │  every variant is loaded through the same interface
                 ▼
     backends/registry.py ── get_backend(variant) ──► Backend
          ├─ TorchBackend   (reference)
          ├─ OrtBackend     (onnx_*)
          └─ TFLiteBackend  (tflite_*, optional)
                 │
     ┌───────────┼─────────────────┐
     ▼           ▼                 ▼
 parity.py   evaluate.py        bench.py
 (vs torch   (top-1 on eval     (latency stats,
  reference)  subset)            load time, size)
     └───────────┴────────┬────────┘
                          ▼
              env.py (CPU, OS, versions)
                          ▼
           results/<run_id>.json   (schema: Data_Schema.md)
                 │                       │
                 ▼                       ▼
            report.py               gate.py  ◄── baselines/<name>.json
   results/<run_id>/report.md      exit code 0 (pass) / 1 (regression)
   results/<run_id>/pareto.png
```

## 2. Command flows (the "user journeys")

### Journey 1 — `edgebench run --config configs/mobilenet_v3_small.yaml --target configs/targets/cpu_4t.yaml`
1. Load and validate the config. Invalid config → clear error message naming the bad field, exit code 2.
2. Check the dataset exists at `EDGEBENCH_DATA_DIR`. Missing → error with the exact download command, exit code 2.
3. Build the calibration and evaluation subsets (deterministic, seeded; disjoint by construction — calib from `train/`, eval from `val/`).
4. Load the PyTorch model with pretrained weights.
5. Export ONNX FP32 (skip if the artifact exists and `--force` is not set). Run `onnx.checker.check_model`.
6. Create INT8 variants requested in the config.
7. For each variant in the config:
   1. Load it via the backend registry (record session load time).
   2. **Parity**: run the parity batch, compare with the torch reference outputs.
   3. **Accuracy**: run the eval subset, compute top-1.
   4. **Benchmark**: for each batch size in the target, warmup then timed runs.
   5. If one variant fails (export error, runtime error), record `status: "error"` with the message and **continue with the other variants**.
8. Capture the environment, write `results/<run_id>.json`, print a short summary table and the results path.
9. Exit code 0 if all variants succeeded, 1 if any variant errored.

### Journey 2 — `edgebench report --results results/<run_id>.json`
1. Load and validate the results JSON.
2. Write `report.md` (markdown table + environment block + methodology notes) and `pareto.png` (accuracy vs p50 latency at batch 1) next to it.
3. Print the paths.

### Journey 3 — `edgebench gate --results results/<run_id>.json --baseline baselines/ci.json`
1. Load both files. Match entries by `(model, variant)`.
2. For each baseline entry: fail if the variant is missing, errored, its accuracy dropped more than `max_accuracy_drop_pts`, or its parity is outside tolerance.
3. Latency is **printed for information, never gated**.
4. Print a pass/fail line per check; exit 0 if all pass, 1 otherwise.

### Edge cases and error states
| Situation | Behaviour |
|---|---|
| Dataset folder missing | Error with download command; exit 2 |
| Weights can't be downloaded (offline) | Error naming the model and `TORCH_HOME`; exit 2 |
| ONNX export fails | Variant marked `error`; dependent INT8 variants marked `skipped` with reason |
| Quantization fails | That variant marked `error`; others continue |
| TFLite artifact missing (optional path) | Variant marked `skipped` with "run scripts/convert_tflite.py first" |
| Parity outside tolerance | Recorded as `parity.within_tolerance: false`; **benchmarks still run** (so you can see both) |
| Benchmark CV > `noisy_cv_threshold` | Recorded as `noisy: true`; a warning in the report |
| Results/baseline schema version mismatch | Error explaining which version is expected; exit 2 |

## 3. Component responsibilities

| Module | Responsibility | Must NOT |
|---|---|---|
| `config.py` | Load YAML (`safe_load`), validate, return frozen dataclasses | Touch models or files other than the config |
| `data.py` | Locate Imagenette, build deterministic subsets, label mapping, shared `preprocess()`, ORT `CalibrationDataReader` | Know about backends |
| `models.py` | Model registry: name → (torchvision constructor, weights enum) | Export or run inference |
| `export.py` | Torch → ONNX FP32; validation with `onnx.checker` | Quantize |
| `quantize.py` | ONNX FP32 → INT8 dynamic / static QDQ | Benchmark |
| `backends/base.py` | `Backend` protocol (see `Data_Schema.md`) | Contain runtime-specific code |
| `backends/registry.py` | Map variant name → backend class | Import optional runtimes at module import time |
| `backends/torch_backend.py` / `ort_backend.py` / `tflite_backend.py` | Load a model file, run `predict(np.ndarray) -> np.ndarray` | Compute metrics |
| `parity.py` | Pure numpy: compare two output arrays → `ParityReport` | Import torch/onnxruntime |
| `evaluate.py` | Run a backend over the eval subset → top-1 | Time anything |
| `bench.py` | Timing loop + pure stats functions → `LatencyStats` | Compute accuracy |
| `env.py` | Capture CPU, OS, Python and package versions | — |
| `results.py` | Build, validate, save, load results JSON | — |
| `report.py` | Results JSON → markdown + PNG | Re-run any model |
| `gate.py` | Results vs baseline → list of check outcomes + exit code | Re-run any model |
| `runner.py` | Orchestrates `run` (Journey 1): artifacts, per-variant parity/accuracy/latency with error isolation, results JSON. Added 2026-10-19 so `cli.py` stays free of business logic | Print (reports progress through a callback) |
| `cli.py` | argparse wiring, logging setup, exit codes | Contain business logic |

**Dependency rule:** `parity.py`, `bench.py` (stats part), `gate.py` and `results.py` must be importable **without** torch or onnxruntime, so their tests are instant.

## 4. Extensibility ("earlier chip variants" story)
- **New runtime** → add `backends/<name>_backend.py` implementing `Backend`, register it in `registry.py`, add a variant name. No other file changes.
- **New target** → add `configs/targets/<name>.yaml` (threads, batch sizes, warmup, runs). No code changes.
- **New model** → add one entry in `models.py` and one config file.
- Baselines are stored **per target** (`baselines/<target>.json`), so results from different machines are never compared against each other.

## 5. Folder structure

```
edgebench/
├── CLAUDE.md                     # Instructions Claude Code reads automatically
├── README.md
├── pyproject.toml                # deps, extras, ruff + pytest config, entry point
├── .gitignore                    # .venv*/, data/, artifacts/, results/* (except samples), __pycache__/
├── configs/
│   ├── mobilenet_v3_small.yaml
│   ├── resnet18.yaml
│   ├── smoke_tiny.yaml           # tiny synthetic model for CI
│   └── targets/
│       ├── cpu_1t.yaml
│       ├── cpu_4t.yaml
│       └── ci.yaml
├── src/edgebench/
│   ├── __init__.py               # __version__
│   ├── cli.py
│   ├── runner.py                 # `run` orchestration (Journey 1)
│   ├── errors.py                 # EdgeBenchError + subclasses (Rules.md §5)
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── export.py
│   ├── quantize.py
│   ├── parity.py
│   ├── evaluate.py
│   ├── bench.py
│   ├── env.py
│   ├── results.py
│   ├── report.py
│   ├── gate.py
│   └── backends/
│       ├── __init__.py
│       ├── base.py
│       ├── registry.py
│       ├── torch_backend.py
│       ├── ort_backend.py
│       └── tflite_backend.py     # optional, Phase 8
├── scripts/
│   └── convert_tflite.py         # runs in .venv-tflite only
├── tests/
│   ├── conftest.py               # tiny CNN fixture, synthetic image folder fixture
│   ├── test_parity.py
│   ├── test_bench_stats.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_export.py
│   ├── test_quantize.py
│   ├── test_backends.py
│   ├── test_results.py
│   ├── test_gate.py
│   ├── test_report.py
│   └── test_cli_smoke.py
├── baselines/
│   └── ci.json                   # baseline for the CI smoke config
├── docs/                         # these planning docs + architecture.png, pareto.png for README
├── data/                         # Imagenette (gitignored)
├── artifacts/                    # exported models (gitignored)
├── results/                      # run outputs (gitignored, except one committed sample run)
└── .github/workflows/ci.yml
```
