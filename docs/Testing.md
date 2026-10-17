# Testing — Strategy, Test List and Commands

> Tests are part of the deliverable: the JD asks for testable code and pytest. Every phase in `Phases.md` lists which tests must pass before it is done.

## 1. Principles
- **Fast by default:** the default test run (`pytest`) needs no network, no dataset and no pretrained weights, and finishes in < 60 s.
- **Test the logic, not the libraries:** don't test that ONNX Runtime works; test that *our* code calls it correctly and handles its outputs.
- **Tests first for pure logic** (`parity`, `bench` stats, `config`, `results`, `gate`).
- **Deterministic:** fixed seeds; no test depends on timing values (only on the structure of timing results).
- **A failing test is information.** Never weaken an assertion to make it pass without writing down why the original was wrong.

## 2. Test layers and markers

| Layer | What | Marker | Runs in CI |
|---|---|---|---|
| Unit | Pure functions with hand-computed expected values | none | Yes |
| Component | Real torch/ORT calls on `tiny_cnn` + synthetic data | none | Yes |
| Smoke / E2E | `edgebench run` → `report` → `gate` on `configs/smoke_tiny.yaml` | none | Yes |
| Data | Uses the real Imagenette folder | `@pytest.mark.data` | No (local only) |
| Slow | Real pretrained models (downloads weights) | `@pytest.mark.slow` | No (local only) |

Register markers in `pyproject.toml`. Data/slow tests skip automatically with a clear reason if the dataset/weights aren't present.

## 3. Test list by module

### `test_parity.py`
- Identical arrays → `max_abs_err == 0`, `cosine_similarity == 1`, `top1_agreement == 1.0`, `within_tolerance is True`.
- Known perturbation (+1e-3 everywhere) → `max_abs_err ≈ 1e-3` (use `pytest.approx`), agreement still 1.0.
- One of 10 rows with swapped argmax → `top1_agreement == 0.9`.
- Shape mismatch → `ValueError` mentioning both shapes.
- Zero-vector rows don't produce NaN cosine (define behaviour and test it).
- float16 / float32 inputs are handled (metrics computed in float64).

### `test_bench_stats.py`
- Known list `[1..100]` → p50, p90, p99 match `numpy.percentile` (linear).
- Constant latencies → std 0, CV 0, IQR 0, not noisy.
- Highly variable list → `noisy is True` when CV above threshold.
- Empty list → `ValueError`; single sample → std reported as 0.0 (documented and tested).

### `test_results.py`
- Round trip: build → `save_results` → `load_results` → equal.
- Wrong `schema_version` → `SchemaError`.
- Missing required field → `SchemaError` naming the field.
- JSON is valid UTF-8 with sorted keys and 2-space indent (stable diffs).

### `test_config.py`
- Valid model + target config loads into frozen dataclasses.
- Each invalid case raises `ConfigError` naming the field: unknown variant, missing model name, non-positive `runs`, empty `batch_sizes`, unknown keys.
- Configs are loaded with `safe_load` (a YAML file with a `!!python/object` tag must fail).

### `test_data.py` (synthetic image folder fixture)
- `list_images` is sorted and only returns image files.
- Same seed → same subset; different seed → different subset.
- Calibration subset (train) and eval subset (val) are disjoint.
- WNID → ImageNet index mapping correct for all 10 classes.
- `preprocess` returns `float32`, shape `(3, H, W)`.
- Missing dataset → `DatasetError` whose message contains the download command.
- `[data]` Real dataset: val has 10 classes; 1000-image subset loads.

### `test_export.py`
- `tiny_cnn` exports to ONNX; `onnx.checker` passes; dynamic batch works (batch 1 and 4).
- ORT output vs torch output within FP32 tolerance.
- Export failure raises `ExportError` (monkeypatch `torch.onnx.export` to raise).

### `test_quantize.py`
- Dynamic and static INT8 files are produced and smaller than FP32.
- Static model contains `QuantizeLinear` and `DequantizeLinear` nodes.
- Quantized models run and return the expected output shape.
- Calibration reader yields the configured number of batches and then `None`.

### `test_backends.py`
- Torch and ORT backends: same output shape and dtype for the same input.
- `predict` rejects wrong dtype / rank with `BackendError`.
- Registry returns the right class per variant; unknown variant → `BackendError`.
- Optional backend (TFLite) missing package → clear `BackendError`, not `ImportError` at import time.

### `test_gate.py`
- All checks pass → exit code 0.
- Accuracy drop above threshold → that check fails, exit 1.
- Parity outside tolerance → fail.
- Variant missing in results → fail; variant with `status: "error"` → fail.
- Latency 10× worse → **still passes** (latency is not gated) but is reported.

### `test_report.py`
- Markdown contains one row per variant and the environment block.
- Errored variants show status and `—` in numeric cells.
- `pareto.png` file is created and non-empty (don't compare pixels).

### `test_cli_smoke.py`
- `edgebench --help` exit code 0.
- `run` on `smoke_tiny` + `ci` target → exit 0, results JSON valid.
- `report` on that result → files created.
- `gate` against `baselines/ci.json` → exit 0; against a deliberately impossible baseline (temp file) → exit 1.
- Bad config path → exit 2 with an `ERROR:` message.

### Import isolation
- A subprocess test that runs `python -c "import edgebench.parity, edgebench.results, edgebench.gate"` with `sys.modules['torch'] = None` and `sys.modules['onnxruntime'] = None` set first, proving these modules don't import heavy dependencies.

## 4. Fixtures (`tests/conftest.py`)
- `tiny_cnn`: 2 conv layers + pooling + linear, 10 outputs, `torch.manual_seed(0)`, eval mode.
- `synthetic_imagenette(tmp_path)`: creates `train/` and `val/` with the 10 real WNID folder names and a few small random RGB JPEGs each (fixed seed).
- `sample_results`: a small valid results object for gate/report tests.
- `tmp_artifacts_dir`, `tmp_results_dir` via `tmp_path`.

## 5. Commands
```bash
pytest                                   # fast suite (what CI runs)
pytest -m "data or slow"                 # local-only tests with real data/weights
pytest --cov=edgebench --cov-report=term-missing
ruff check . && ruff format --check .
```

## 6. Coverage target
- Aim for ≥ 85% line coverage on `parity`, `bench`, `results`, `config`, `gate`; ≥ 70% overall.
- Report the actual number in `Memory.md` and the README. Never state a coverage number that wasn't measured.

## 7. CI (`.github/workflows/ci.yml`)
- Trigger: push and pull request on `main`.
- `permissions: contents: read`.
- Steps: checkout → setup Python (pinned version) → pip cache → install CPU torch → `pip install -e ".[dev]"` → `ruff check .` → `ruff format --check .` → `pytest` → `edgebench run --config configs/smoke_tiny.yaml --target configs/targets/ci.yaml` → `edgebench gate ... --baseline baselines/ci.json`.
- Upload the smoke results JSON + report as a build artifact.
- Must not download Imagenette or pretrained weights.

## 8. Manual checks per phase
Each phase's "Validation" block in `Phases.md` is the manual check. Paste its output (or a summary) into `Memory.md` when the phase completes.
