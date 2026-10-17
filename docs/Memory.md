# Memory — Project State

> The AI reads this first in every session and updates it at the end of every phase and before the session ends. Keep it short and factual: current state, decisions, measured values, open issues. Newest entries at the top of each section. Never write numbers here that weren't measured.

## Current status
- **Current phase:** Phase 1 — Project setup (done; Phase 2 not started)
- **Last updated:** 2026-10-18
- **Next step:** Phase 2 — write tests first for `parity.py`, `bench.py` stats, `results.py` (see `Phases.md` Phase 2). Fix the WNID mapping error (Open issues) before Phase 3.
- **Blockers:** none

## Phase checklist
- [x] Phase 1 — Project setup (2026-10-18)
- [ ] Phase 2 — Parity, latency stats, results schema
- [ ] Phase 3 — Config and data
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
- **Phases.md Phase 3 WNID mapping is wrong for parachute:** dataset folder is `n03888257`, doc says `n03888605→701`. ImageNet index 701 = parachute = `n03888257` (`n03888605` is "parallel bars", 702). Fix the doc (with user approval) before writing `data.py`.
- Doc commands in `CLAUDE.md` / `Phases.md` are bash/WSL; on this machine use PowerShell equivalents (e.g. `.\.venv\Scripts\Activate.ps1`).
- Phase 8 (TFLite) needs a Linux environment (WSL Ubuntu not installed).

## Session log (one line per session, newest first)
- 2026-10-18 — Phase 1 done on native Windows: venv, pinned `pyproject.toml`, `__init__`/`errors`/`cli` + 9 tests, `.gitignore`, CI skeleton, Imagenette downloaded. Validation: `edgebench --help` OK, 9 passed, ruff check + format clean, val/train 10 class folders each.
- 2026-10-18 — Planning docs written (PRD, TRD, Architecture, Rules, Phases, Design, Testing, Security, Data_Schema, Memory, CLAUDE.md). No code yet.
