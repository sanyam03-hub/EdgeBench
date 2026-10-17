# EdgeBench — instructions for Claude Code

EdgeBench is a Python CLI that converts PyTorch models to ONNX Runtime (FP32/INT8) and TFLite, checks the converted outputs match the PyTorch reference, benchmarks latency/size/accuracy, and fails CI on regressions. It is a portfolio project for an ML benchmarking role; the user must be able to explain every part of it in interviews.

## Start of every session
1. Read `docs/Memory.md` (current phase, decisions, open issues).
2. Read the current phase in `docs/Phases.md`.
3. Read `docs/Rules.md` once per session. Read other docs only when the task needs them.
4. Say which phase you are on and what you will do next, then start.

## Source-of-truth documents (`docs/`)
| Doc | Use it for |
|---|---|
| `PRD.md` | Scope: what's in, what's out |
| `TRD.md` | Stack, dependencies, environment, key technical decisions |
| `Architecture.md` | Command flows, module responsibilities, folder structure |
| `Data_Schema.md` | Config/results/baseline JSON fields, accuracy definition, Python interfaces |
| `Phases.md` | Build order, done criteria, validation commands |
| `Rules.md` | How to work: workflow, honesty, style, errors |
| `Testing.md` | Test list per module, markers, CI |
| `Security.md` | What not to do with files, YAML, subprocess, secrets |
| `Design.md` | Terminal output, report table, plot and README conventions |
| `Memory.md` | Current state — **update it at the end of every phase** |

If two docs conflict, stop and ask. Precedence: PRD > Data_Schema > Architecture > TRD > others.

## Non-negotiables (details in Rules.md)
- One phase at a time; small steps; run tests after each step.
- Never invent numbers. Placeholders until a real run exists.
- No dependencies outside `TRD.md` without asking.
- Explain the approach before writing `parity.py`, `bench.py`, `quantize.py`, `gate.py`.
- Don't commit or push unless asked; suggest a commit message at the end of each phase.

## Environment
WSL2 Ubuntu, project at `~/edgebench`, virtualenv `.venv` (activate with `source .venv/bin/activate`). TFLite conversion only in `.venv-tflite`.

## Setup commands (Phase 1)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy pyyaml matplotlib pytest pytest-cov ruff
mkdir -p data && curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz | tar xz -C data
```

## Everyday commands
```bash
pytest                       # fast suite, no network/dataset
pytest -m "data or slow"     # local-only tests
ruff check . && ruff format --check .
edgebench run --config configs/mobilenet_v3_small.yaml --target configs/targets/cpu_4t.yaml
edgebench report --results results/<run_id>.json
edgebench gate --results results/<run_id>.json --baseline baselines/ci.json
```
