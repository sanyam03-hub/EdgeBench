# Rules — How the AI must work on this project

> These rules apply to every session. If a rule conflicts with a request in chat, **stop and ask** instead of picking one.

## 1. Workflow rules
1. **Read first.** At the start of every session read `CLAUDE.md`, `docs/Memory.md`, and the current phase in `docs/Phases.md`. Read other docs only when the task needs them.
2. **One phase at a time.** Work only on the phase named in `Memory.md` as current. Do not start the next phase until the user says so.
3. **Small steps.** Within a phase, work in steps of at most ~150 lines of new code. After each step: run the relevant tests, show the result, then continue.
4. **Tests first for pure logic.** For `parity.py`, `bench.py` (stats), `config.py`, `results.py` and `gate.py`: write the failing test, then the implementation.
5. **Definition of done** for any step: tests pass, `ruff check .` and `ruff format --check .` pass, and the phase's own validation (in `Phases.md`) has been run and its output shown.
6. **Update `Memory.md`** at the end of every phase and before the session ends: what was done, decisions made, measured values, open issues, next step.
7. **Commits:** suggest a commit message at the end of each phase in Conventional Commits style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Do **not** commit or push unless the user asks.
8. **Explain the important parts.** The user must be able to defend this project in interviews. For `parity.py`, `bench.py`, `quantize.py` and `gate.py`, explain the approach in 3–6 sentences **before** writing code, and list the concepts the user should understand.

## 2. Scope rules
- Build only what is in `PRD.md` "Must have" until Phase 7 is done. Nice-to-haves only when the user asks.
- Do **not** add: web UI, Streamlit, API servers, databases, Docker (until asked), LLM/agent features, GPU code.
- Do **not** add dependencies that are not listed in `TRD.md`. If one seems necessary, stop and ask, explaining why stdlib or an existing dependency won't do.
- Do **not** rename modules, variant names, or JSON fields defined in `Architecture.md` / `Data_Schema.md` without asking. If a change is approved, update the docs in the same step.

## 3. Honesty rules (most important)
- **Never invent numbers.** No accuracy, latency, speedup, size or coverage figure may appear in the README, docs or commit messages unless it came from a real run whose results JSON exists in `results/`.
- Use placeholders like `[TBD — from run <id>]` until a real number exists.
- Never describe CPU results as "edge", "mobile", "on-device" or "Snapdragon" performance.
- If a result is surprising or bad (e.g. INT8 slower than FP32, accuracy drop larger than expected), **report it as-is** and suggest likely causes. Never tune the evaluation to make a number look better.
- If a library API is uncertain, say so and check it (inspect the installed package with `python -c "help(...)"` or read its source) instead of guessing. Mark unverified assumptions in `Memory.md`.
- Never claim a step is done if its tests were not run. If a test was skipped, say which and why.

## 4. Code style rules
- Python 3.11+ syntax, PEP 8 via ruff (line length 100).
- Type hints on all public functions; `from __future__ import annotations` at the top of every module.
- Docstrings (one line + Args/Returns when non-obvious) on public functions and classes. No comments that restate the code.
- Use `dataclasses` (frozen where possible) for structured data; no untyped dicts passed between modules except at the JSON boundary (`results.py`).
- Use `pathlib.Path`, never string path concatenation.
- No `print` in library code; use `logging.getLogger(__name__)`. Only `cli.py` prints user-facing output.
- No global mutable state. Seeds and thread counts come from the config, passed explicitly.
- Keep functions short (aim < 40 lines). Pure computation separated from I/O.
- Heavy imports (`torch`, `onnxruntime`, `ai_edge_litert`) only inside the modules that need them; `parity.py`, `gate.py`, `results.py` and the stats in `bench.py` must not import them.

## 5. Error-handling rules
- Define project exceptions in `src/edgebench/errors.py`: `EdgeBenchError` (base), `ConfigError`, `DatasetError`, `ExportError`, `QuantizationError`, `BackendError`, `SchemaError`.
- Raise specific exceptions with actionable messages: what failed, the input that caused it, and what to do next.
- Never use bare `except:` or `except Exception: pass`. Catch broad exceptions only at the per-variant boundary in the run loop, where the error is recorded in the results JSON (`status: "error"`, `error: "<type>: <message>"`) and the run continues.
- `cli.py` maps exceptions to exit codes: `0` success, `1` regression or variant failure, `2` usage/config/data error.
- Validate inputs at boundaries (config load, results load, backend `predict` input shape/dtype); trust internal calls.

## 6. Measurement rules
- Same preprocessed input arrays for every backend (`data.preprocess`).
- Latency: warmup runs are never recorded; time only the inference call; record every timed run before computing stats.
- Thread counts always set explicitly for both torch and ORT from the target profile.
- Calibration data comes only from `train/`; evaluation only from `val/`. Never mix.
- Every results JSON includes the environment block (see `Data_Schema.md`).

## 7. What the AI should do when stuck
- After **two** failed attempts at the same error, stop, summarize what was tried and the exact error, and propose 2 options to the user.
- For the TFLite phase, respect the **75-minute timebox**: if conversion isn't working, document what failed in `Memory.md` and mark the phase as deferred.
- Never "fix" a failing test by weakening its assertion without explaining why the original assertion was wrong.

## 8. Things the AI must never do
- Commit secrets, tokens, datasets, model weights or exported artifacts.
- Run `git push`, `git reset --hard`, or delete files outside `artifacts/` and `results/` without asking.
- Use `pickle` / `torch.load` on files not produced by this project; if `torch.load` is ever needed, use `weights_only=True`.
- Use `yaml.load` (only `yaml.safe_load`), `eval`, `exec`, or `subprocess` with `shell=True`.
- Edit files under `docs/` other than `Memory.md` without telling the user what changed.
