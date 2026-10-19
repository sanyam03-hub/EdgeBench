# Memory — Project State

> The AI reads this first in every session and updates it at the end of every phase and before the session ends. Keep it short and factual: current state, decisions, measured values, open issues. Newest entries at the top of each section. Never write numbers here that weren't measured.

## Current status
- **Current phase:** Phase 8 — TFLite **deferred** 2026-10-19 (no Linux; user declined WSL). Phase 9 not started.
- **Last updated:** 2026-10-19
- **Next step:** user commits the CI-made `baselines/ci.json` + this Memory.md. Then Phase 9 (README; re-run real models on an idle machine first — see Open issues). README must list TFLite as deferred, not as a feature.
- **Blockers:** none. Phases 5–7 committed+pushed by user as `21818a2` ("Till phase 7"); CI run 39218972275: success (59 s). Phase 4 = `15ae622` (CI success). Phases 1–3 = `1c6b25b` (CI success).

## Phase checklist
- [x] Phase 1 — Project setup (2026-10-18)
- [x] Phase 2 — Parity, latency stats, results schema (2026-10-18)
- [x] Phase 3 — Config and data (2026-10-18)
- [x] Phase 4 — Models, ONNX export, backends (2026-10-18)
- [x] Phase 5 — Quantization (2026-10-18; MobileNetV3-Small INT8 finding open)
- [x] Phase 6 — Evaluation, benchmark runner, `run` command (2026-10-19)
- [x] Phase 7 — Report, gate, CI (2026-10-19)
- [~] Phase 8 — TFLite (optional, 75-min timebox) — **deferred 2026-10-19**: converter has no Windows build; user declined Linux/WSL
- [ ] Phase 9 — README and results

## Environment (fill in during Phase 1)
- OS / WSL distro: native Windows 11 Home 10.0.26200 (no WSL Ubuntu — see Decisions); project at `C:\Users\sanyam jain\Desktop\EdgeBench`, venv `.venv` (`.\.venv\Scripts\Activate.ps1`)
- CPU model / logical cores: AMD Ryzen 5 5600H with Radeon Graphics / 6 physical, 12 logical
- Python: 3.11.4
- torch / torchvision: 2.14.1+cpu / 0.29.1+cpu
- onnx / onnxruntime: 1.23.2 / 1.31.0 (providers: CPUExecutionProvider, AzureExecutionProvider)
- numpy: 2.4.6
- Others: pillow 12.3.0, pyyaml 6.0.3, matplotlib 3.11.2, pytest 9.1.1, pytest-cov 7.1.0, ruff 0.17.1
- Imagenette tarball SHA-256: 64D0C4859F35A461889E0147755A999A48B49BF38A7E0F9BD27003F10DB02FE5 (99003388 bytes)
- Imagenette class folder image counts (train / val), measured 2026-10-18: n01440764 963/387, n02102040 955/395, n02979186 993/357, n03000684 858/386, n03028079 941/409, n03394916 956/394, n03417042 961/389, n03425413 931/419, n03445777 951/399, n03888257 960/390

## Decisions log
| Date | Decision | Reason |
|---|---|---|
| 2026-10-19 | `run` progress/summary printed with `flush=True` (`cli._say`) | Redirected stdout is block-buffered: background run output file stayed empty for 14 min. Verified: flushed line arrived in a pipe after 0.3 s vs 5.0 s (child exit) for plain print |
| 2026-10-19 | Gate chain per entry stops at first failure: present → status → accuracy (`top1 ≥ baseline − max_drop − 1e-9`) → parity (if `require_parity`) → `latency_info` (always passes, reports current p50 only — baseline stores no latency) | Data_Schema §7; epsilon so "exactly at threshold" passes despite float rounding |
| 2026-10-19 | Gate raises `SchemaError` (exit 2) if results' `config.target.name` ≠ baseline `target` | Architecture §4: baselines per target, never compared across machines |
| 2026-10-19 | `edgebench gate --write-baseline [--max-accuracy-drop 0.5]` creates a baseline from a results file: one entry per `ok` variant; `require_parity` = parity was measured (false for torch_fp32) | Phases "`--write-baseline` or documented manual copy"; 0.5 = Data_Schema §7 example |
| 2026-10-19 | `baselines/ci.json` regenerated from the **CI runner's** smoke results (`20261019T075446Z_7275bc9f`, artifact of run 39218972275) — not yet committed. Entries identical to the provisional local baseline (all top-1 10.9375, same `require_parity`); only `created_from_run_id`/`created_at` changed | Data_Schema §7. Gate passes against both CI and local results |
| 2026-10-19 | `report` writes to `results/<run_id>/` (folder named after the JSON); `load_results` raises `SchemaError` for a missing file (exit 2) | Architecture Journey 2; clean error instead of traceback |
| 2026-10-19 | Pareto plot: legend outside axes; labels alternate above/below in latency order, below in the top 10 % of the range; y from a round number ≥5 pts below the min, 3 % headroom above 100 (ticks stop at 100) | First renders had labels overlapping, a point on the axis edge, legend over a data point (checked visually on smoke, MobileNet, ResNet plots) |
| 2026-10-19 | CI: after tests → smoke `run` → `report` → `gate` vs `baselines/ci.json`; `results/` uploaded as artifact `smoke-results` with `if: always()` | Testing.md §7 |
| 2026-10-19 | New module `runner.py` holds `run` orchestration (Architecture.md updated); `cli.py` only parses args, prints, maps exit codes | Architecture says cli has no business logic; Phases says "cli.py run: orchestrate" — resolved this way |
| 2026-10-19 | `run` exit codes: 0 if no variant has status `error`; 1 if any does (parity failure alone is **not** an error — the gate judges parity); 2 for Config/Dataset/Schema errors | Architecture Journey 1 step 9; Rules §5 |
| 2026-10-19 | Dataset existence checked before loading any model (fail fast, no results file written) | Journey 1 step 2 |
| 2026-10-19 | `dataset: synthetic` = seeded random arrays in memory (`data.synthetic_split`, RNG seeded with `[seed, split]`), labels cycling the 10 Imagenette ImageNet indices; `ArrayCalibrationReader` for calibration | Not defined in docs; offline, no files; train/val never identical even with the same seed |
| 2026-10-19 | Accuracy: 1000-logit models restricted to the 10 Imagenette columns; a 10-logit model's output j = class `LABEL_INDICES[j]` | Data_Schema §5; lets `tiny_cnn` use the same code |
| 2026-10-19 | Parity batch = first `parity.n_images` of the eval subset; benchmark input for batch b = first b eval images; eval streamed per `eval.batch_size` | Data_Schema §2; RAM limit |
| 2026-10-19 | ONNX FP32 export reused if the file exists (re-export with `--force`); failed export's partial file is deleted; INT8 variants always re-quantized | Architecture Journey 1 step 5; INT8 must never silently use stale calibration settings |
| 2026-10-19 | `torch_fp32` row: `artifact` and `size_mb` = null (no file) | Schema allows null; no artifact exists |
| 2026-10-19 | Logging: root logger WARNING (silences ORT quantization INFO noise), `edgebench` logger INFO / DEBUG with `--verbose` | ORT logs via the root logger |
| 2026-10-19 | Terminal output ASCII-only ("Delta", not "Δ"); noisy CV marked `*` with a footnote | Windows cp1252 can't encode "Δ" when output is piped |
| 2026-10-19 | CPU model: `/proc/cpuinfo` on Linux, registry `ProcessorNameString` (stdlib `winreg`) on Windows, else `platform.processor()` | Phases says `/proc/cpuinfo` (Linux only); dev machine is Windows |
| 2026-10-18 | Static INT8 = `quant_pre_process` (temp file) → `quantize_static` QDQ, settings from config (u8 activations passed explicitly — ORT 1.31 defaults activations to QInt8); dynamic INT8 = `quantize_dynamic`, QInt8 weights, no pre-processing | TRD key decisions |
| 2026-10-18 | Plain `quantize_static` (no chunked calibration) | Checked ORT 1.31 source: MinMax calibrator stores only `ReduceMin`/`ReduceMax` outputs per batch, so memory ≈ one batch through the model. (My earlier 3.1 GB / 5.9 GB estimate assumed full activations — wrong for MinMax; it would matter for Entropy/Percentile, stretch S2) |
| 2026-10-18 | Never use ORT `CalibMaxIntermediateOutputs` | Read in ORT 1.31.0 `calibrate.py` (`MinMaxCalibrater.collect_data`): when the limit is hit it calls `clear_collected_data()` **without** `compute_data()`, discarding those ranges (observed in source, not run). The supported chunking path is `CalibStridedMinMax` (needs reader `set_range`) |
| 2026-10-18 | `quantize.py` validates the FP32 input with `onnx.checker.check_model` first | Corrupt input otherwise surfaces as `google.protobuf...DecodeError` (protobuf not a TRD dependency); checker raises `ValidationError` |
| 2026-10-18 | `tiny_cnn` widened to 16/32 channels (~5.4k params) | With 8/16 channels (~1.6k params) the static QDQ file (10,018 B) was larger than FP32 (7,858 B): Q/DQ node + scale overhead outweighed the int8 weight saving. Testing.md's "INT8 smaller than FP32" assertion kept unchanged |
| 2026-10-18 | `ORT_ERRORS` (was `_ORT_ERRORS`) made public in `ort_backend.py` | Reused by `quantize.py` |
| 2026-10-18 | ONNX export uses the **legacy TorchScript exporter** (`dynamo=False`, explicit) — user choice | torch 2.14.1 defaults to `dynamo=True`, which needs `onnxscript` (not installed, not in TRD). Legacy path works but emits `DeprecationWarning` ("will be removed"); torch is pinned so this is stable here. Revisit before any torch upgrade |
| 2026-10-18 | `create_backend` takes an extra optional `weights="DEFAULT"` kwarg (not in Data_Schema §8 signature) | Torch reference must use the same weights as the exported model; otherwise a non-DEFAULT config silently compares against DEFAULT |
| 2026-10-18 | Pretrained-weight download failure → `DatasetError` (exit 2); unknown model/weights → `ConfigError` | Architecture says "weights can't be downloaded → exit 2" but names no exception; weights treated as data |
| 2026-10-18 | `TorchBackend` sets only `torch.set_num_threads(intra_op_threads)` (process-wide); inter-op threads not set for torch | `torch.set_num_interop_threads` raises once parallel work has run; torch has no per-model thread setting |
| 2026-10-18 | No broad `except Exception` outside the run loop: export catches `(RuntimeError, TypeError, ValueError)`; ORT backend catches exactly the exception classes defined in `onnxruntime.capi.onnxruntime_pybind11_state` | Rules §5; verified ORT errors (e.g. `InvalidProtobuf`) derive directly from `Exception` |
| 2026-10-18 | `tiny_cnn`: 2 conv + pool + linear, 10 classes, 32×32 input, seeded inside `torch.random.fork_rng()` | Deterministic without changing global RNG state (test checks this) |
| 2026-10-18 | Config errors name the field as a quoted dotted path (`'eval.n_images'`); unknown keys rejected at every level; `bool` rejected where an int/float is expected | Tests can match the exact field; YAML `true` would otherwise pass as `1` |
| 2026-10-18 | `model` and target `name` must match `[A-Za-z0-9_]+` | They become path parts (`artifacts/<model>/`, `baselines/<target>.json`) — stricter than Security.md's "no `/`, `\`, `..`" |
| 2026-10-18 | `warmup` may be 0; all other counts ≥ 1; seeds ≥ 0; `noisy_cv_threshold` > 0; `dataset` ∈ {`imagenette2-160`, `synthetic`} | Data_Schema §2/§3 |
| 2026-10-18 | `quantization` section always required, even if no INT8 variant is listed | Simpler, explicit configs |
| 2026-10-18 | Labels returned by `data.load_batch` are **ImageNet indices** (e.g. 701), not 0–9 | 10-way accuracy (Data_Schema §5) is computed in `evaluate.py` (Phase 6); `tiny_cnn` smoke data will need its own label handling then |
| 2026-10-18 | `data.py` doesn't import torch or onnxruntime; `ImagenetteCalibrationReader` is duck-typed | Verified in ORT 1.31.0: `CalibrationDataReader` only abstracts `get_next` and its `__subclasshook__` accepts any class with it (test asserts `isinstance`) |
| 2026-10-18 | `sample_subset` = `default_rng(seed).choice(..., replace=False)`, result sorted | Deterministic and order-stable |
| 2026-10-18 | Real-dataset `[data]` test loads the 1000-image eval subset per batch (32) | 7.4 GB machine with ~0.9 GB free OOM'd loading 1000×3×224×224 float32 + `np.stack` copy at once; assertion (all 1000 load, 10 classes) unchanged |
| 2026-10-18 | Parity: `compare` requires 2-D non-empty equal shapes; checks only the tolerance keys given (`max_abs_err`, `min_cosine`, `min_top1_agreement`; unknown keys → `ValueError`); metrics in float64 | INT8 passes only `min_top1_agreement` (Data_Schema §2) |
| 2026-10-18 | Cosine of a zero row: both zero → 1.0, one zero → 0.0 | Never NaN; identical-zero outputs count as agreement |
| 2026-10-18 | Latency stats: `noisy` = CV strictly > threshold; single sample → std/CV 0.0; empty, negative or non-finite samples → `ValueError`; CV 0.0 when mean is 0 | Testing.md §3; bad timings are bugs, not data |
| 2026-10-18 | Results dataclasses named `RunResults`, `VariantResult`, `Environment`, `AccuracyResult` (Data_Schema fixes field names only); `variants` is a tuple in code, list in JSON | Frozen, typed; JSON only at the boundary |
| 2026-10-18 | `save_results` writes with `newline="\n"` | On Windows `write_text` would emit CRLF and break byte-stable diffs |
| 2026-10-18 | Unknown extra fields in results JSON are ignored (not rejected) | Not specified; only required fields + `schema_version` + `status` are validated |
| 2026-10-18 | Develop on **native Windows** with Python 3.11 (supersedes the WSL2 decision below) | No WSL Ubuntu distro installed; user chose speed. CI still runs on Linux. Phase 8 (ai-edge-torch) will need WSL/Linux when reached |
| 2026-10-18 | Ruff excludes `docs/` (`extend-exclude`) | ruff 0.17 formats Python blocks inside Markdown; planning docs aren't source |
| 2026-10-18 | CI runs `pytest -m "not data and not slow"` (Phases.md), not bare `pytest` (Testing.md §7) | Explicit exclusion; doesn't rely on auto-skip |
| 2026-10-18 | `[tflite]`/`[aihub]` extras listed unpinned, not installed | Package names/compat still unverified |
| 2026-10-18 | `CLAUDE.md` copied to project root (original stays in `docs/`) | Architecture.md expects it at root for Claude Code to auto-read |
| 2026-10-18 | Project = EdgeBench; MVP = torch + ONNX FP32/INT8 dynamic/INT8 static, 2 models, CPU only | Best fit for Qualcomm ML benchmarking JD (see PRD) |
| 2026-10-18 | Develop in WSL2 Ubuntu, not native Windows | ai-edge-torch (TFLite) needs Linux; one environment for everything |
| 2026-10-18 | No database; results as JSON files | Few runs; diffable in git (Data_Schema §1) |
| 2026-10-18 | Accuracy = 10-way argmax over Imagenette's ImageNet indices | Pretrained 1000-class models, no training (Data_Schema §5) |
| 2026-10-18 | CI gates on accuracy + parity only; latency reported, not gated | Shared CI runners are too noisy for latency gates |

## Measured values (copy from real runs only)
| Date | Run ID | Model | Variant | Metric | Value | Notes |
|---|---|---|---|---|---|---|
| 2026-10-18 | (manual, no run ID) | mobilenet_v3_small | onnx_fp32 vs torch_fp32 | parity on 32 eval images (seed 0), cpu_4t | max_abs 6.497e-05, mean_abs 6.902e-06, cosine 1.000000000, top-1 agree 1.0 | max \|logit\| 27.481; ONNX file 10.18 MB; legacy exporter, opset 17 |
| 2026-10-19 | 20261019T071148Z_e0e28630 | resnet18 | torch_fp32 | top-1 (10-way, 1000 val) / p50 b1 / p50 b8 | 97.70% / 33.39 ms (CV 13.5% noisy) / 156.91 ms (CV 7.5%) | cpu_4t; run took 833.6 s |
| 2026-10-19 | 20261019T071148Z_e0e28630 | resnet18 | onnx_fp32 | top-1 / Δ / agree / p50 b1 / p50 b8 / size | 97.70% / +0.00 / 1.000 (max_abs 1.7e-05) / 13.84 ms (CV 16.6% noisy) / 103.00 ms (CV 7.8%) / 46.75 MB | |
| 2026-10-19 | 20261019T071148Z_e0e28630 | resnet18 | onnx_int8_dynamic | top-1 / Δ / agree / p50 b1 / p50 b8 / size | 97.60% / −0.10 / 0.969 (max_abs 1.5) / 199.00 ms (CV 9.2%) / 1639.41 ms (CV 18.9% noisy) / 11.74 MB | ~14× slower than ONNX FP32 at b1 |
| 2026-10-19 | 20261019T071148Z_e0e28630 | resnet18 | onnx_int8_static | top-1 / Δ / agree / p50 b1 / p50 b8 / size | 96.20% / −1.50 / 1.000 (max_abs 12) / 10.39 ms (CV 11.1% noisy) / 74.71 ms (CV 8.7%) / 11.82 MB | ~1.33× faster than ONNX FP32 at b1 (13.84→10.39), 4× smaller |
| 2026-10-19 | 20261019T070905Z_e158a466 | mobilenet_v3_small | torch_fp32 | top-1 / p50 b1 / p50 b8 | 97.60% / 7.60 ms (CV 15.3% noisy) / 26.57 ms (CV 15.3% noisy) | cpu_4t; run took 149.0 s |
| 2026-10-19 | 20261019T070905Z_e158a466 | mobilenet_v3_small | onnx_fp32 | top-1 / Δ / agree / p50 b1 / p50 b8 / size | 97.60% / +0.00 / 1.000 (max_abs 6.5e-05) / 1.55 ms (CV 17.7% noisy) / 12.04 ms (CV 14.4% noisy) / 10.18 MB | |
| 2026-10-19 | 20261019T070905Z_e158a466 | mobilenet_v3_small | onnx_int8_dynamic | top-1 / Δ / agree / p50 b1 / p50 b8 / size | **14.10%** / −83.50 / **0.000** (parity FAIL) / 25.39 ms (CV 10.0% noisy) / 207.43 ms (CV 13.5% noisy) / 2.72 MB | INT8 collapse (Open issues) |
| 2026-10-19 | 20261019T070905Z_e158a466 | mobilenet_v3_small | onnx_int8_static | top-1 / Δ / agree / p50 b1 / p50 b8 / size | **25.20%** / −72.40 / **0.000** (parity FAIL) / 2.36 ms (CV 19.3% noisy) / 14.38 ms (CV 17.5% noisy) / 2.91 MB | INT8 collapse; also **slower** than ONNX FP32 (1.55 ms) |
| 2026-10-19 | 20261019T070750Z_7275bc9f | tiny_cnn (smoke) | all 4 | status / run time | all ok; INT8 agree 1.000; top-1 10.94% (chance, random model) | ci target; CLI run took 12.2 s (< 60 s criterion) |
| 2026-10-18 | (manual, no run ID) | mobilenet_v3_small | onnx_int8_dynamic vs torch_fp32 | agreement on 32 eval images, cpu_4t | top-1 agree **0/32**, cosine −0.1596, max_abs 29.04 | file 2.72 MB (FP32 10.18 MB); quantize time 0.7 s |
| 2026-10-18 | (manual, no run ID) | mobilenet_v3_small | onnx_int8_static vs torch_fp32 | agreement on 32 eval images, cpu_4t | top-1 agree **0/32**, cosine −0.0439, max_abs 26.61 | file 2.91 MB; 256 calib images (train/), minmax, per-channel, u8s8; quantize time 8.8 s |
| 2026-10-18 | (manual, no run ID) | resnet18 | onnx_int8_dynamic vs torch_fp32 | agreement on 32 eval images, cpu_4t | top-1 agree 31/32 (0.9688), cosine 0.9954, max_abs 1.463 | file 11.74 MB (FP32 46.75 MB); quantize time 1.0 s |
| 2026-10-18 | (manual, no run ID) | resnet18 | onnx_int8_static vs torch_fp32 | agreement on 32 eval images, cpu_4t | top-1 agree 32/32 (1.0), cosine 0.9549, max_abs 12.39 | file 11.82 MB; 256 calib images, minmax, per-channel, u8s8; quantize time 16.6 s |
| 2026-10-18 | (manual, no run ID) | resnet18 | onnx_fp32 vs torch_fp32 | parity on 32 eval images (seed 0), cpu_4t | max_abs 1.717e-05, mean_abs 2.062e-06, cosine 1.000000000, top-1 agree 1.0 | max \|logit\| 32.224; ONNX file 46.75 MB; legacy exporter, opset 17 |

## Tolerances (set in Phase 4 from observed values)
- FP32 parity `max_abs_err` tolerance: **1e-3 (kept, calibrated 2026-10-18)** — reason: observed max 6.5e-5 (mobilenet) / 1.7e-5 (resnet18) on one 32-image batch at cpu_4t → ~15× / ~58× headroom for other thread counts, batch sizes and CPUs (CI); real conversion bugs (wrong preprocessing, broken op) produce errors ≳1e-2, so 1e-3 still catches them. `min_cosine` 0.9999 and `min_top1_agreement` 0.99 kept (observed 1.0; at n=32, 0.99 means any flip fails, correct for FP32).
- INT8 `min_top1_agreement`: 0.90 (unchanged) — observed resnet18 0.97 (dynamic) / 1.0 (static) on 32 images; mobilenet_v3_small 0.0 for both — a real failure, **not** a tolerance problem; do not lower the threshold to hide it.

## Unverified assumptions (check, then move to Decisions)
- [x] `ai-edge-torch` has no native Windows support (TRD) — **checked 2026-10-19: true, and renamed.** `ai-edge-torch` 0.7.2 is a 1.8 KB deprecated stub ("renamed to litert-torch") that depends on `litert-torch`. `litert-torch` 0.9.4 (pure Python) requires `torch<2.14.0,>=2.4.0`, `ai-edge-litert[model-utils]>=2.2,<2.3`, `litert-converter==0.4.*`, plus jax, transformers, torchao, kagglehub, ai-edge-quantizer. **`litert-converter` has no Windows wheel in any version** (PyPI JSON: only `manylinux_2_27_x86_64` and `macosx_12_0_arm64`, cp310–cp314) ⇒ conversion needs Linux (WSL/CI) or macOS; and a separate venv with torch ≤2.13.
- [x] `ai-edge-litert` is the correct interpreter package name and doesn't conflict with the torch pin — **checked 2026-10-19: yes.** 2.3.0 has a `cp311-win_amd64` wheel (20.7 MB); deps: flatbuffers, numpy, protobuf, ml_dtypes, tqdm, typing-extensions, backports.strenum — no torch. Converter pins 2.2.x; interpreter version for the main venv still to choose (`[tflite]` extra).
- [x] Best `torch.onnx.export` exporter mode for torch 2.14.1 — **checked: `dynamo=False`** (see Decisions)
- [x] Imagenette folder names match the WNID mapping in Phases.md Phase 3 — **checked: 9/10 match; mismatch logged in Open issues**
- [ ] `onnxruntime.quantization.qdq_loss_debug` API (stretch S1)
- [ ] Imagenette licence wording for the README

## Open issues / known problems
- ~~Phases.md WNID mapping wrong for parachute~~ — fixed 2026-10-18 with user approval (`n03888605` → `n03888257`).
- **MobileNetV3-Small INT8 collapses (dynamic and static, 0/32 top-1 agreement vs torch); ResNet-18 INT8 is fine.** Diagnosis 2026-10-18 (32 eval images, agreement vs ONNX FP32, one setting changed per row): static baseline 0/32 (cos −0.04); `reduce_range=True` 0/32 (+0.27); s8s8 activations 0/32 (+0.18); per-tensor weights 0/32 (−0.21); dynamic baseline 0/32 (−0.16); dynamic `reduce_range` 0/32 (+0.35); **dynamic with only MatMul/Gemm quantized (convs FP32) 31/32 (+0.9998)**. ⇒ quantizing MobileNetV3's **conv layers** breaks it in every mode tried; not u8s8 saturation (reduce_range doesn't fix), not calibration (dynamic also fails). Root cause not yet found — candidates: a few layers with extreme activation/weight ranges (e.g. first conv, SE blocks, Hardswish/Hardsigmoid paths). Investigation stopped after the same failure repeated (user rule). **User decision 2026-10-19: record and move on** — configs unchanged; Phase 6 `run` will report MobileNet INT8 as-is (parity fail + measured accuracy); root cause to be found in stretch S1 (per-layer FP32 vs INT8 error analysis).
- **Latency noise:** in both real cpu_4t runs most measurements had CV > 10% (range 7.5–19.3%) — laptop not idle (VS Code, etc.). Before README numbers (Phase 9): close other apps, plug in power, re-run; consider cpu_1t too.
- **INT8 dynamic is much slower than FP32 on these CNNs** (resnet18 b1: 199.00 vs 13.84 ms; mobilenet b1: 25.39 vs 1.55 ms). Expected for dynamic quantization of convs on CPU (runtime activation quantization + ConvInteger path) — report, don't hide. ORT also warns "consider pre-processing" for dynamic (not pre-processed per TRD).
- **MobileNetV3-Small INT8 static is slower than ONNX FP32** (b1 2.36 vs 1.55 ms) on top of the accuracy collapse; ResNet-18 INT8 static is faster (10.39 vs 13.84 ms). Possible causes: small depthwise convs where Q/DQ overhead dominates; no VNNI on Zen 3 (unverified). Investigate with S1.
- Import-isolation test now covers parity, bench, results and gate (done 2026-10-19).
- **Gate failure demonstrated locally 2026-10-19:** provisional `baselines/ci.json` → GATE PASSED, exit 0; `onnx_fp32` top1 set to 101 → 1 FAIL (accuracy, "minimum 100.50%"), GATE FAILED (1 checks), exit 1; reverted (byte-identical to backup) → GATE PASSED, exit 0. **CI green confirmed 2026-10-19** (run 39218972275 on `21818a2`): 205 passed, smoke run → report → gate `GATE PASSED`, artifact `smoke-results` uploaded (3 files, 53,488 B). CI env: Linux 6.17.0-1022-azure, Intel Xeon Platinum 8573C, 4 cores, Python 3.11.17. Smoke config hash identical to local (`7275bc9f`); top-1 identical (10.94%); INT8 max abs err differs by platform (CI dynamic 9.7e-04 / static 1.6e-03 vs local 5.2e-03 / 8.3e-03) — kernel differences, agreement 1.000 on both. **CI going red demonstrated 2026-10-19** (user-approved): temp branch `demo/gate-red-check` (from `21818a2`, commit `d51d11f`: `onnx_fp32` baseline top-1 → 101) via draft PR #1 → CI run 39231885225 **failed at the Regression gate step only** (Lint, Test, Smoke run, Report success; Upload smoke results still ran): `FAIL tiny_cnn/onnx_fp32 accuracy: top-1 10.94% vs baseline 101.00% (minimum 100.50%, change -90.06 pts)`, `GATE FAILED (1 checks)`, exit 1. PR closed, branch deleted (remote + local), worktree removed; `main` unchanged. Note: CI triggers only on push to `main` and PRs into `main`, so the demo needed a PR. Pushing from this shell needs `-c credential.helper="!gh auth git-credential"` (git has no stored credentials; global config not changed).
- **Low RAM:** 7.4 GB total, ~0.9 GB free during Phase 3. Evaluation (Phase 6) must stream batches, never hold the full eval set; close other apps before benchmark runs (also reduces latency noise).
- `DOWNLOAD_COMMAND` in `data.py` uses `curl ... && tar ...` — works in bash and cmd, not Windows PowerShell 5.1 (no `&&`).
- Doc commands in `CLAUDE.md` / `Phases.md` are bash/WSL; on this machine use PowerShell equivalents (e.g. `.\.venv\Scripts\Activate.ps1`).
- **Phase 8 (TFLite) deferred 2026-10-19** (~10 min of the 75-min timebox used, on package checks only; no code written). Reason: `ai-edge-torch` is deprecated → renamed `litert-torch` (0.9.4, needs torch<2.14); its `litert-converter` dependency has **no Windows wheel in any version** (Linux x86_64 + macOS arm64 only). User first chose WSL Ubuntu, then **declined switching to Linux** (2026-10-19). The interpreter (`ai-edge-litert` 2.3.0) does have a Windows cp311 wheel, so only conversion is blocked. Existing hooks stay: registry `tflite_fp32` lazily imports `backends/tflite_backend.py` (not written) and raises `BackendError`; runner skips tflite variants without an artifact. To resume later: any Linux/macOS machine or CI job runs the conversion into `.venv-tflite`; TRD must then be updated for the `litert-torch` rename.

## Session log (one line per session, newest first)
- 2026-10-19 — Phase 8 deferred: TFLite converter (`litert-torch`/`litert-converter`) has no Windows build; user declined WSL/Linux. No code changed; fast suite unchanged (205 passed, 3 deselected).
- 2026-10-19 — Phase 7 done: user pushed `21818a2`; CI run 39218972275 green with smoke+report+gate; `baselines/ci.json` regenerated from the CI artifact (uncommitted); CI red demo via draft PR #1 (run 39231885225, failed at gate only), then PR closed + branch deleted.
- 2026-10-19 — Phase 7 local: flush fix; `gate.py` (explained first) + `report.py`, tests first; `cli report`/`gate` (+`--write-baseline`); provisional `baselines/ci.json`; CI smoke/report/gate/upload steps. Validation `test_gate.py test_report.py`: 24 passed; fast suite 205 passed (3 deselected); ruff clean; measured coverage total 95% (gate 93%, report 98%, cli 94%, runner 93%, env 82%). Reports + plots generated for smoke, MobileNet and ResNet runs.
- 2026-10-19 — Phase 6 done: `evaluate.py`, `bench.benchmark`/`file_size_mb`, `env.py`, `runner.py`, `cli.py run`, synthetic data + `smoke_tiny.yaml`; tests first. Fast suite 176 passed (3 deselected); ruff clean. Smoke CLI run 12.2 s, exit 0. Real runs on cpu_4t: mobilenet_v3_small (149 s) and resnet18 (833.6 s), all variants `ok`, exit 0 (results in Measured values). Latency CVs 7.5–19.3% (most flagged noisy) — machine not idle during runs.
- 2026-10-18 — Phase 5 done: `quantize.py` (dynamic + static QDQ), tests first on `tiny_cnn` (widened). Validation `test_quantize.py`: 9 passed; fast suite 142 passed (3 deselected); ruff clean; measured coverage quantize 90%. Real models quantized: resnet18 INT8 OK; **mobilenet_v3_small INT8 0/32** (see Open issues, Measured values).
- 2026-10-18 — Phase 4 done: `models.py` (+`tiny_cnn`), `export.py` (legacy exporter), `backends/` (base, torch, ort, registry); tests first. Validation `test_export.py test_backends.py`: 27 passed; fast suite 133 passed (3 deselected); slow model tests 2 passed; ruff clean. Measured coverage: backends 97–100%, export 100%, models 86% (uncovered = pretrained path, covered by slow tests). Real-model FP32 parity measured (see Measured values); FP32 tolerance kept at 1e-3.
- 2026-10-18 — Phases 1–3 pushed to GitHub by user as `1c6b25b` (also contains `.ignore` and `.serena/` tool files); GitHub Actions `ci` on that push: success (54 s).
- 2026-10-18 — Phase 3 done: `config.py` + 5 configs, `data.py`, `conftest.py` synthetic dataset; `.gitignore` += `*.pt`, `.env`. Validation: config+data 59 passed (1 data test deselected); real `[data]` test passed; `list_images` val = 3925, train = 9469; fast suite 100 passed; ruff clean. Measured coverage: config 98%, data 96%. Eval subset (seed 0, n=1000) label counts by ImageNet idx: 0:97, 217:94, 482:81, 491:95, 497:98, 566:107, 569:109, 571:110, 574:95, 701:114.
- 2026-10-18 — Phase 2 done: `parity.py`, `bench.py` (stats), `results.py`, tests written first, plus import-isolation test. Validation: 31/31 passed; full suite 41 passed; ruff clean. Measured coverage: bench 100%, parity 100%, results 98% (uncovered: non-object JSON branches, lines 119/149).
- 2026-10-18 — Phase 1 committed (`1325534`) on branch `main` (renamed from `master`).
- 2026-10-18 — Phase 1 done on native Windows: venv, pinned `pyproject.toml`, `__init__`/`errors`/`cli` + 9 tests, `.gitignore`, CI skeleton, Imagenette downloaded. Validation: `edgebench --help` OK, 9 passed, ruff check + format clean, val/train 10 class folders each.
- 2026-10-18 — Planning docs written (PRD, TRD, Architecture, Rules, Phases, Design, Testing, Security, Data_Schema, Memory, CLAUDE.md). No code yet.
