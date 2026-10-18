# Phases — Implementation Plan

> Build in this order. Each phase has a goal, tasks, files, done criteria and a validation command. **Do one phase per session; update `Memory.md` at the end of each.**
> The PDF template's "Database" and "Auth" phases do not apply (no database, no users). Estimated total for Phases 1–7 + 9: ~9 focused hours.

---

## Phase 1 — Project setup (≈30 min)
**Goal:** an empty but correctly configured, installable, testable package.

Tasks
1. In WSL Ubuntu: `~/edgebench`, `git init`, create `.venv`, install CPU torch + torchvision, then the rest of `TRD.md` runtime + dev dependencies.
2. Create `pyproject.toml`: project metadata, dependencies (pinned to the installed versions), extras `[dev]`, `[tflite]`, `[aihub]`, entry point `edgebench = "edgebench.cli:main"`, ruff config (line length 100, rules `E,F,W,I,B,UP,N`), pytest config (`testpaths = ["tests"]`, markers `data`, `slow`).
3. Create `src/edgebench/__init__.py` (`__version__ = "0.1.0"`), `errors.py` (exception classes from `Rules.md` §5), `cli.py` with a `main()` that prints help.
4. `.gitignore`: `.venv*/`, `data/`, `artifacts/`, `results/*` with `!results/sample/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.onnx`, `*.tflite`, `.coverage`.
5. `.github/workflows/ci.yml` skeleton: checkout, Python setup, install CPU torch + `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `pytest -m "not data and not slow"`.
6. Download Imagenette into `data/` (see `CLAUDE.md` setup commands).

Done when
- `pip install -e ".[dev]"` succeeds; `edgebench --help` prints usage.
- `pytest -q` runs (0 tests or 1 trivial import test) and `ruff check .` passes.
- `data/imagenette2-160/train` and `val` exist with 10 class folders each.
- Python and package versions recorded in `Memory.md`.

Validation
```bash
edgebench --help && pytest -q && ruff check . && ls data/imagenette2-160/val | wc -l
```

Commit: `chore: scaffold edgebench package, tooling and CI skeleton`

---

## Phase 2 — Pure core: parity, latency stats, results schema (≈60 min, TDD)
**Goal:** the numeric heart of the project, fully tested, with no torch/ORT imports.

Tasks
1. `parity.py`: `compare(reference, candidate, tolerance) -> ParityReport` (fields in `Data_Schema.md`). Validate shapes (raise `ValueError` on mismatch) and dtypes (cast to float64 for metrics).
2. `bench.py` (stats part only): `summarize_latencies(samples_ms: Sequence[float], noisy_cv_threshold: float) -> LatencyStats` — p50/p90/p99 (`numpy.percentile`, linear), mean, std (ddof=1), IQR, CV, n, noisy flag.
3. `results.py`: dataclasses for the results file, `to_dict` / `from_dict`, `save_results`, `load_results` with `schema_version` check → `SchemaError`.

Done when
- Tests from `Testing.md` §3 for these modules pass, including edge cases (identical arrays, known perturbation, one flipped argmax, shape mismatch, single-sample latency list, constant latencies → CV 0).
- `python -c "import edgebench.parity, edgebench.results"` works in a venv **without** torch (CI checks this with an import-isolation test).

Validation
```bash
pytest tests/test_parity.py tests/test_bench_stats.py tests/test_results.py -v
```

Commit: `feat: add parity metrics, latency statistics and results schema`

---

## Phase 3 — Config and data (≈60 min)
**Goal:** deterministic, leakage-free data subsets and validated configs.

Tasks
1. `config.py`: load model config + target config (`yaml.safe_load`) into frozen dataclasses (`ModelConfig`, `TargetConfig`, `RunConfig`); validate required fields, allowed variant names, positive ints; raise `ConfigError` naming the field.
2. Create `configs/mobilenet_v3_small.yaml`, `configs/resnet18.yaml`, `configs/targets/cpu_1t.yaml`, `cpu_4t.yaml`, `ci.yaml` (schemas in `Data_Schema.md`).
3. `data.py`:
   - `IMAGENETTE_WNID_TO_IMAGENET_IDX` mapping (n01440764→0, n02102040→217, n02979186→482, n03000684→491, n03028079→497, n03394916→566, n03417042→569, n03425413→571, n03445777→574, n03888257→701). Folder names verified against the downloaded dataset on 2026-10-18 (was wrongly `n03888605`).
   - `list_images(split_dir)` sorted by path; `sample_subset(paths, n, seed)` deterministic.
   - `preprocess(path, transforms) -> np.ndarray float32 (3,H,W)` using the torchvision weights' transforms.
   - `load_batch(paths, transforms) -> (np.ndarray (N,3,H,W), np.ndarray labels)`.
   - `ImagenetteCalibrationReader` implementing ORT's `CalibrationDataReader` (`get_next` returns `{input_name: batch}` then `None`).
4. Raise `DatasetError` with the download command if the dataset is missing.

Done when
- Config tests pass (valid config loads; each invalid case raises `ConfigError` with the field name).
- Data tests on a **synthetic** image folder pass: determinism (same seed → same subset), calib and eval subsets disjoint, labels mapped correctly, output shape/dtype correct.
- Manual check on the real dataset: eval subset of 1000 images loads; label distribution printed.

Validation
```bash
pytest tests/test_config.py tests/test_data.py -v
python -c "from edgebench.data import list_images; print(len(list_images('data/imagenette2-160/val')))"
```

Commit: `feat: add config loading and deterministic Imagenette subsets`

---

## Phase 4 — Models, ONNX export, backends (≈75 min)
**Goal:** PyTorch reference and ONNX FP32 behind one interface, with parity verified.

Tasks
1. `models.py`: registry `{"mobilenet_v3_small": ..., "resnet18": ...}` → model + weights enum + transforms; also `tiny_cnn` (for tests/CI, random init with fixed seed, no download).
2. `export.py`: `export_onnx(model, sample_input, path, opset=17)` with dynamic batch axis and explicit exporter mode; run `onnx.checker.check_model`; raise `ExportError` with the original message.
3. `backends/base.py`: `Backend` protocol; `backends/registry.py`; `torch_backend.py`; `ort_backend.py` (session options from target profile).
4. Each backend validates input dtype `float32` and rank 4 and returns `float32` logits `(N, 1000)` (or `(N, num_classes)` for tiny_cnn).

Done when
- `test_export.py`: tiny_cnn exports, loads in ORT, parity within FP32 tolerance on random input — **no network**.
- `test_backends.py`: both backends return the same shape/dtype; registry raises `BackendError` on unknown variant.
- Manual: mobilenet_v3_small and resnet18 export; parity vs torch on a 32-image real batch printed and recorded in `Memory.md`. Set the FP32 tolerance in configs from these observed values (with margin) and note the reasoning.

Validation
```bash
pytest tests/test_export.py tests/test_backends.py -v
```

Commit: `feat: add ONNX export and torch/ORT backends behind a common interface`

---

## Phase 5 — Quantization (≈60 min)
**Goal:** INT8 dynamic and static QDQ variants.

Tasks
1. `quantize.py`:
   - `quantize_dynamic_int8(fp32_path, out_path)`.
   - `quantize_static_int8(fp32_path, out_path, calibration_reader, per_channel=True, calibrate_method="minmax")`, running `quant_pre_process` first into a temp file.
   - Raise `QuantizationError` with the cause.
2. Before coding, **explain** QDQ format, per-channel vs per-tensor, and calibration (Rules §1.8).

Done when
- `test_quantize.py` on tiny_cnn: INT8 files exist, are smaller than FP32, contain `QuantizeLinear`/`DequantizeLinear` nodes (static) and run in ORT with the right output shape.
- Manual: both real models quantized; INT8 static top-1 agreement vs torch on 32 images printed and recorded in `Memory.md`.

Validation
```bash
pytest tests/test_quantize.py -v
```

Commit: `feat: add ORT dynamic and static QDQ INT8 quantization`

---

## Phase 6 — Evaluation, benchmark runner and `edgebench run` (≈75 min)
**Goal:** one command produces a complete results JSON.

Tasks
1. `evaluate.py`: `top1_accuracy(backend, batches, label_indices) -> float` — accuracy definition from `Data_Schema.md` §5.
2. `bench.py` (runner part): `benchmark(backend, input_batch, warmup, runs) -> LatencyStats`; model file size; load time.
3. `env.py`: capture OS, kernel, CPU model (`/proc/cpuinfo`), logical cores, Python and package versions.
4. `cli.py run`: orchestrate per `Architecture.md` Journey 1, with per-variant error isolation and exit codes.
5. `configs/smoke_tiny.yaml` using `tiny_cnn` + synthetic data so `run` works without the dataset.

Done when
- `edgebench run --config configs/smoke_tiny.yaml --target configs/targets/ci.yaml` succeeds offline in < 60 s and writes a valid results JSON.
- `edgebench run` for both real models on `cpu_4t` writes results with all 4 variants `status: "ok"` (or an honest `error` with message).
- CV printed for each latency measurement; any noisy measurements flagged.

Validation
```bash
pytest tests/test_cli_smoke.py -v
edgebench run --config configs/mobilenet_v3_small.yaml --target configs/targets/cpu_4t.yaml
```

Commit: `feat: add evaluation, benchmark runner and run command`

---

## Phase 7 — Report, gate and CI (≈60 min)
**Goal:** human-readable output and an automated regression gate.

Tasks
1. `report.py`: markdown table (columns in `Design.md`), environment block, methodology notes, `pareto.png`.
2. `gate.py`: checks per `Architecture.md` Journey 3; returns list of `GateCheck`; `cli.py gate` maps to exit code.
3. Create `baselines/ci.json` from a smoke run (`edgebench gate --write-baseline` or a documented manual copy).
4. CI: add a step running the smoke config, then `edgebench gate` against `baselines/ci.json`.

Done when
- `test_gate.py`: passing case, accuracy-drop case, parity-fail case, missing-variant case, errored-variant case, latency never fails the gate.
- CI is green on GitHub.
- **Demonstrated failure:** edit `baselines/ci.json` to an impossible accuracy, confirm CI/gate goes red, then revert. Record in `Memory.md`.

Validation
```bash
pytest tests/test_gate.py tests/test_report.py -v
edgebench report --results results/<run_id>.json
edgebench gate --results results/<run_id>.json --baseline baselines/ci.json; echo "exit=$?"
```

Commit: `feat: add markdown/plot report, regression gate and CI smoke test`

---

## Phase 8 — TFLite (optional, HARD TIMEBOX 75 min)
**Goal:** one model's TFLite FP32 variant in the results.

Tasks
1. Create `.venv-tflite`; `pip install ai-edge-torch` **[verify install command and supported Python]**.
2. `scripts/convert_tflite.py --model mobilenet_v3_small --out artifacts/mobilenet_v3_small/tflite_fp32.tflite` (uses only `edgebench.models`, not the rest of the package).
3. `backends/tflite_backend.py` using the LiteRT interpreter; read the input shape from the interpreter and **assert the layout** (NCHW vs NHWC) instead of assuming.
4. Add `tflite_fp32` to the mobilenet config.

Done when
- TFLite variant appears in results with parity vs torch.
- **If not working at 75 min:** stop, record what failed in `Memory.md`, mark Phase 8 "deferred", go to Phase 9.

Commit: `feat: add TFLite conversion script and LiteRT backend`

---

## Phase 9 — README, results and polish (≈45 min)
**Goal:** a repo a reviewer understands in 2 minutes and can reproduce.

Tasks
1. Run both models on `cpu_1t` and `cpu_4t`; generate reports; commit one sample run to `results/sample/`.
2. README per `Design.md` §4 with **real numbers only**, the Pareto plot, the architecture diagram, methodology, limitations.
3. Fresh-clone test: clone into `/tmp`, follow the quickstart exactly, confirm it works.
4. Coverage report: `pytest --cov=edgebench`; record the number in `Memory.md` (do not inflate).

Done when
- README contains no placeholder `[TBD]` values; every number traces to `results/sample/`.
- Fresh-clone quickstart works.

Commit: `docs: add README with measured results and methodology`

---

## Stretch phases (only when the user asks)
- **S1 — Per-layer quantization error analysis:** FP32 vs INT8 per-layer activation error (ORT `onnxruntime.quantization.qdq_loss_debug` **[verify API]**), ranked bar chart, re-quantize with top-k layers excluded (`nodes_to_exclude`), measure accuracy recovery vs a random-exclusion baseline. Calibration from train, search on half of eval, report on the other half.
- **S2 — Calibration sweep:** MinMax / Entropy / Percentile × per-tensor / per-channel on the Pareto plot.
- **S3 — Qualcomm AI Hub:** compile + profile on 2–3 hosted Snapdragon devices; separate `results/aihub/`; label clearly as on-device numbers.
- **S4 — Second model family** (DistilBERT) to contrast dynamic quantization.
- **S5 — Dockerfile + `.gitlab-ci.yml` mirror.**

## Final done criteria (MVP)
- `edgebench run`, `report`, `gate` work end-to-end for both models on the user's machine.
- CI green: lint, unit tests, smoke run + gate.
- README with measured results, methodology, limitations and reproducible quickstart.
- `Memory.md` up to date.
