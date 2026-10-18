# Memory — Project State

> The AI reads this first in every session and updates it at the end of every phase and before the session ends. Keep it short and factual: current state, decisions, measured values, open issues. Newest entries at the top of each section. Never write numbers here that weren't measured.

## Current status
- **Current phase:** Phase 3 — Config and data (done; Phase 4 not started)
- **Last updated:** 2026-10-18
- **Next step:** Phase 4 — `models.py` (incl. `tiny_cnn`), `export.py`, backends; verify `torch.onnx.export` `dynamo=` mode for torch 2.14.1 first (see `Phases.md` Phase 4).
- **Blockers:** none. Phases 2 and 3 are not committed yet (user has not asked).

## Phase checklist
- [x] Phase 1 — Project setup (2026-10-18)
- [x] Phase 2 — Parity, latency stats, results schema (2026-10-18)
- [x] Phase 3 — Config and data (2026-10-18)
- [ ] Phase 4 — Models, ONNX export, backends
- [ ] Phase 5 — Quantization
- [ ] Phase 6 — Evaluation, benchmark runner, `run` command
- [ ] Phase 7 — Report, gate, CI
- [ ] Phase 8 — TFLite (optional, 75-min timebox)
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
| | | | | | | |

## Tolerances (set in Phase 4 from observed values)
- FP32 parity `max_abs_err` tolerance: (initial 1e-3, not yet calibrated) — reason:
- INT8 `min_top1_agreement`: (initial 0.90, not yet calibrated) — reason:

## Unverified assumptions (check, then move to Decisions)
- [ ] `ai-edge-torch` has no native Windows support (TRD)
- [ ] `ai-edge-litert` is the correct interpreter package name and doesn't conflict with the torch pin
- [ ] Best `torch.onnx.export` exporter mode (`dynamo=True/False`) for the pinned torch version
- [x] Imagenette folder names match the WNID mapping in Phases.md Phase 3 — **checked: 9/10 match; mismatch logged in Open issues**
- [ ] `onnxruntime.quantization.qdq_loss_debug` API (stretch S1)
- [ ] Imagenette licence wording for the README

## Open issues / known problems
- ~~Phases.md WNID mapping wrong for parachute~~ — fixed 2026-10-18 with user approval (`n03888605` → `n03888257`).
- Import-isolation test covers parity, bench, results; add `edgebench.gate` in Phase 7.
- **Low RAM:** 7.4 GB total, ~0.9 GB free during Phase 3. Evaluation (Phase 6) must stream batches, never hold the full eval set; close other apps before benchmark runs (also reduces latency noise).
- `DOWNLOAD_COMMAND` in `data.py` uses `curl ... && tar ...` — works in bash and cmd, not Windows PowerShell 5.1 (no `&&`).
- Doc commands in `CLAUDE.md` / `Phases.md` are bash/WSL; on this machine use PowerShell equivalents (e.g. `.\.venv\Scripts\Activate.ps1`).
- Phase 8 (TFLite) needs a Linux environment (WSL Ubuntu not installed).

## Session log (one line per session, newest first)
- 2026-10-18 — Phase 3 done: `config.py` + 5 configs, `data.py`, `conftest.py` synthetic dataset; `.gitignore` += `*.pt`, `.env`. Validation: config+data 59 passed (1 data test deselected); real `[data]` test passed; `list_images` val = 3925, train = 9469; fast suite 100 passed; ruff clean. Measured coverage: config 98%, data 96%. Eval subset (seed 0, n=1000) label counts by ImageNet idx: 0:97, 217:94, 482:81, 491:95, 497:98, 566:107, 569:109, 571:110, 574:95, 701:114.
- 2026-10-18 — Phase 2 done: `parity.py`, `bench.py` (stats), `results.py`, tests written first, plus import-isolation test. Validation: 31/31 passed; full suite 41 passed; ruff clean. Measured coverage: bench 100%, parity 100%, results 98% (uncovered: non-object JSON branches, lines 119/149).
- 2026-10-18 — Phase 1 committed (`1325534`) on branch `main` (renamed from `master`).
- 2026-10-18 — Phase 1 done on native Windows: venv, pinned `pyproject.toml`, `__init__`/`errors`/`cli` + 9 tests, `.gitignore`, CI skeleton, Imagenette downloaded. Validation: `edgebench --help` OK, 9 passed, ruff check + format clean, val/train 10 class folders each.
- 2026-10-18 — Planning docs written (PRD, TRD, Architecture, Rules, Phases, Design, Testing, Security, Data_Schema, Memory, CLAUDE.md). No code yet.
