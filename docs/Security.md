# Security — Threats and Rules for a Local ML Tool

> EdgeBench has no users, auth, server or database, so web-app security (RLS, sessions, payments) does not apply. The real risks for an ML tooling repo are below. Keep this proportionate: follow the rules, don't build security infrastructure.

## 1. Threat model (what can actually go wrong)
| Risk | Where | Mitigation |
|---|---|---|
| Arbitrary code execution from untrusted model/pickle files | `torch.load`, pickle, `.pt` files | Never load pickled files we didn't create. If `torch.load` is ever needed: `weights_only=True`. Pretrained weights only via torchvision's weights enums (they verify file hashes). |
| Code execution via YAML | Config loading | `yaml.safe_load` only. Test that a `!!python/object` tag is rejected. |
| Malicious or huge ONNX files | Loading models from outside the project | Only load ONNX/TFLite files produced by this project under `artifacts/`. Don't load models with external-data files from elsewhere. |
| Tampered dataset download | Imagenette tarball | Download only from the official URL in `CLAUDE.md`; record the tarball's SHA-256 in `Memory.md` on first download; extract into `data/` only. Never run scripts from the dataset. |
| Leaked secrets | Qualcomm AI Hub token (stretch), any future API key | Never in code, configs, results, logs or git. AI Hub token lives in the user's home config via `qai-hub configure`. `.gitignore` covers `.env`. |
| Committing large/sensitive files | Dataset, weights, artifacts | `.gitignore`: `data/`, `artifacts/`, `*.onnx`, `*.tflite`, `*.pt`, `results/*` (except `results/sample/`). |
| Shell injection | Running the TFLite conversion script or any subprocess | `subprocess.run([...], check=True)` with an argument list; never `shell=True`; never build commands from config strings. |
| Path traversal / overwrite | Output paths built from config values | Build paths with `pathlib` from validated names (model/variant from a fixed vocabulary); refuse names containing `/`, `\` or `..`. Write only under `artifacts/` and `results/`. |
| Supply chain | Python dependencies, GitHub Actions | Pin package versions in `pyproject.toml`; pin Actions to a major version tag at minimum (SHA pinning is better); optional `pip-audit` locally before release. |
| CI privilege misuse | GitHub Actions | `permissions: contents: read`; trigger on `push` and `pull_request` only (never `pull_request_target`); no secrets needed for CI. |
| Leaking machine details | `env.py` capture in committed results | Record only CPU model, core count, OS/kernel version, Python and package versions. **Do not** record hostname, username, home paths, IP addresses or environment variables. |
| Misleading claims (integrity) | README, results | See `Rules.md` §3: no invented numbers; CPU results never labelled as on-device. |

## 2. Rules for the coding agent
1. No `eval`, `exec`, `pickle.load`, `yaml.load`, `shell=True`.
2. No network calls in library code except torchvision weight downloads (explicitly triggered by `run`), and none in tests.
3. No secrets in any file. If a feature needs a credential, read it from the environment or the tool's own config and fail with a clear message if absent.
4. No telemetry, analytics or uploading of results anywhere.
5. Absolute paths and usernames never appear in committed files (results JSONs store paths relative to the project root).
6. Before the first `git push`, run: `git ls-files | grep -Ei "\.(onnx|tflite|pt|tgz|env)$"` — it must print nothing.

## 3. Data and licences
- **Imagenette:** public dataset from fast.ai. Don't redistribute it in the repo; link to the source. **[verify licence before mentioning it in README]**
- **torchvision pretrained weights:** downloaded at runtime; not redistributed.
- **Qualcomm AI Hub (stretch):** follow its terms of service. Label results as AI Hub measurements with device names and date.

## 4. Responsible reporting
- Results JSONs committed to the repo are public — review them before committing (environment block, no personal paths).
- If a bug is found in a third-party library, report it upstream with a minimal reproduction.
