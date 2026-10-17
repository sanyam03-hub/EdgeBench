# TRD — Technical Requirements Document

> Locks in the technical choices so the coding agent never picks a stack on its own. Changing anything here requires the user's approval and a note in `Memory.md`.

## Project type
Python library + command-line tool. **No frontend, no backend server, no database, no auth, no hosting.**
(The PDF template's Frontend / Backend / Database / Auth / Hosting rows are intentionally "N/A".)

| Template row | EdgeBench answer |
|---|---|
| Frontend | N/A — CLI output + generated markdown report + PNG plots |
| Backend | N/A — Python package `edgebench`, invoked via CLI |
| Database | N/A — results stored as JSON files (see `Data_Schema.md`) |
| Auth | N/A |
| Hosting | GitHub repository + GitHub Actions CI. No deployment. |

## Development environment
> **Changed 2026-10-18 (user decision):** development is on native Windows 11 with Python 3.11, in the Desktop project folder, because WSL Ubuntu is not installed. The WSL plan below still applies to Phase 8 (TFLite). See `Memory.md` Decisions.

- **OS:** Windows 11 host, development inside **WSL2 Ubuntu** (22.04 or 24.04).
  Reason: `ai-edge-torch` (TFLite conversion) does not support native Windows; one Linux environment avoids split toolchains. **[verify current platform support]**
- **Editor:** VS Code with the **WSL** extension; the project folder lives in the Linux filesystem (`~/edgebench`), **not** under `/mnt/c` (slow I/O, permission issues).
- **Python:** 3.11 or 3.12 (`python3 -m venv .venv`). Pick one in Phase 1 and record it in `Memory.md`.
- **Package layout:** `src/` layout, installed in editable mode (`pip install -e ".[dev]"`).

## Runtime dependencies (main venv `.venv`)
| Package | Purpose | Notes |
|---|---|---|
| `torch`, `torchvision` | Reference models, ONNX export | Install **CPU wheels** from `https://download.pytorch.org/whl/cpu` |
| `onnx` | Model loading, inspection, checker | |
| `onnxruntime` | Inference backend + quantization (`onnxruntime.quantization`) | CPU package only |
| `numpy` | Arrays, metrics | |
| `pillow` | Image loading | Already a torchvision dependency |
| `pyyaml` | Config files | **Always `yaml.safe_load`** |
| `matplotlib` | Pareto plot | Agg backend (no display) |

## Dev dependencies (`[dev]` extra)
| Package | Purpose |
|---|---|
| `pytest` | Tests |
| `pytest-cov` | Coverage report |
| `ruff` | Linting + formatting (PEP 8) |

## Optional / later dependencies
| Package | Where | Purpose |
|---|---|---|
| `ai-edge-torch` | **Separate venv `.venv-tflite`** | PyTorch → TFLite conversion (offline artifact step) |
| `ai-edge-litert` | Main venv, `[tflite]` extra | LiteRT (TFLite) interpreter for benchmarking **[verify package name and that it doesn't conflict with the torch pin]** |
| `qai-hub` | `[aihub]` extra | Qualcomm AI Hub (stretch) |

**Explicitly not used:** pandas (not needed; avoid the dependency), typer/click (stdlib `argparse` is enough), Streamlit, FastAPI, any database, any LLM SDK, TensorFlow in the main venv.

## Version policy
- Pin exact versions in `pyproject.toml` once Phase 1 installs succeed; record them in `Memory.md`.
- Every results JSON records the installed versions of torch, torchvision, onnx, onnxruntime, numpy and Python.
- Upgrading a pinned dependency is a deliberate change: run the full test suite and the gate before committing.

## Key technical decisions
| Decision | Choice | Reason |
|---|---|---|
| ONNX export | `torch.onnx.export`, **opset 17**, dynamic batch axis, exporter mode passed **explicitly** (`dynamo=` argument) | Recent torch versions changed the default exporter. **[verify which mode works best for the pinned torch version; record choice in Memory.md]** |
| ONNX pre-processing for quantization | `onnxruntime.quantization.quant_pre_process` (shape inference + optimization) before `quantize_static` | Recommended by ORT docs |
| Static quantization | `quantize_static`, `QuantFormat.QDQ`, activations `QUInt8`, weights `QInt8`, `per_channel=True`, `CalibrationMethod.MinMax` (default) | Common CPU configuration; calibration method configurable |
| Dynamic quantization | `quantize_dynamic`, weights `QInt8` | Included as a contrast; may be **slower** for CNNs on CPU — report, don't hide |
| ORT session | `CPUExecutionProvider`, `intra_op_num_threads` and `inter_op_num_threads` set from the target profile, `graph_optimization_level=ORT_ENABLE_ALL` | Reproducible threading |
| PyTorch reference | `model.eval()`, `torch.inference_mode()`, `torch.set_num_threads()` from the target profile | |
| Preprocessing | One shared function using the torchvision weights' transforms, producing `float32` NCHW numpy arrays; **all backends consume the same arrays** | Removes preprocessing as a source of mismatch |
| Timing | `time.perf_counter_ns()` around the single inference call only | Excludes data loading from latency |
| CLI | `argparse` with subcommands `run`, `report`, `gate`; entry point `edgebench = "edgebench.cli:main"` | No extra dependency |
| Config | YAML → frozen `dataclasses` with validation | No pydantic dependency |
| Logging | stdlib `logging`, INFO default, `--verbose` for DEBUG | No `print` in library code |
| TFLite path | Conversion runs as a separate script in `.venv-tflite` that writes `.tflite` files; benchmarking uses only the interpreter in the main venv | Avoids torch version conflicts; mirrors compile-vs-profile separation |

## Third-party services
| Service | Purpose | Tier |
|---|---|---|
| GitHub + GitHub Actions | Repo + CI | Free |
| Imagenette download (fast.ai S3) | Dataset | Free, public |
| torchvision weight downloads | Pretrained weights | Free, public |
| Qualcomm AI Hub (stretch) | Real Snapdragon profiling | Free account required; user signs up themselves |

## Environment variables
| Name | Purpose | Default |
|---|---|---|
| `EDGEBENCH_DATA_DIR` | Root folder containing `imagenette2-160/` | `./data` |
| `EDGEBENCH_ARTIFACTS_DIR` | Where exported models are written | `./artifacts` |
| `EDGEBENCH_RESULTS_DIR` | Where run JSONs are written | `./results` |
| `TORCH_HOME` | torchvision weight cache (standard torch variable) | torch default |

Qualcomm AI Hub (stretch) stores its token via `qai-hub configure` in the user's home directory — **never** in the repo. See `Security.md`.

## Folder structure and naming
See `Architecture.md` for the full tree. Conventions:
- Modules and functions: `snake_case`; classes: `PascalCase`; constants: `UPPER_SNAKE_CASE`.
- Artifact files: `artifacts/<model>/<variant>.onnx` (e.g. `artifacts/mobilenet_v3_small/onnx_int8_static.onnx`).
- Variant names (fixed vocabulary): `torch_fp32`, `onnx_fp32`, `onnx_int8_dynamic`, `onnx_int8_static`, `tflite_fp32`, `tflite_int8_dynamic`.
- Run IDs: `YYYYMMDDTHHMMSSZ_<8-char config hash>`.

## Hard constraints
- CPU only. Must run on a laptop with ~8 GB RAM.
- Unit tests must run in < 60 seconds with **no network and no dataset** (use synthetic data and a tiny model).
- CI must not download Imagenette or pretrained weights.
- No result is reported without the environment it was measured on.
- All code passes `ruff check` and `ruff format --check`.
